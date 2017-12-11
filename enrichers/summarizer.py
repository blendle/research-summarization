import json
import pickle
import torch

from gensim.models.word2vec import Word2Vec, KeyedVectors
from sklearn.externals import joblib as sk_joblib
import joblib

import representations as repr
from sent_selectors.modified_greedy import modified_greedy
from sent_selectors.textrank import textrank
from sent_selectors.msv import msv
from sent_selectors import greedy_objectives as objectives
from sent_selectors.rnn_ranker import rnn_ranker
from rnn.model import EncoderRNN, DecoderRNN, Attention

from enrichers.base import Enricher


# Define model class
class W2VModel(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.total_words = sum(vocab_obj.count for vocab_obj in model.vocab.values())
        self.freq_dict = {word: vocab_obj.count / self.total_words
                          for word, vocab_obj in model.vocab.items()}
        self.sif_svd_model = joblib.load('models/sif_svd_{n}.pkl'.format(n=self.name))
        self.pc = self.sif_svd_model.components_

class RNNModel(object):
    def __init__(self,
                 USE_CUDA,
                 parameters=None):
        embedding_size = 300
        rnn_hidden_size = 500
        attn_hidden_size = 500
        dropout_p = .3
        output_size = 2

        self.encoder = EncoderRNN(embedding_size, rnn_hidden_size, dropout_p=dropout_p)
        self.attn = Attention(rnn_hidden_size, attn_hidden_size, output_size)
        self.decoder = DecoderRNN(embedding_size + output_size, rnn_hidden_size,
                                  dropout_p=dropout_p)

        # If a savestate is given, then load it
        if parameters:
            self.encoder.load_state_dict(parameters['enc_state_dict'])
            self.decoder.load_state_dict(parameters['dec_state_dict'])
            self.attn.load_state_dict(parameters['att_state_dict'])

        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()
            self.attn = self.attn.cuda()

        # Set all model elements in eval mode
        self.encoder.eval()
        self.decoder.eval()
        self.attn.eval()

class Summarizer(Enricher):
    name = 'summarizer'
    persistent = True
    requires = ('sentence_splitter', 'tokenizer', 'stemmer', 'postagger')
    # list of pos-tags for filter (nouns, numbers and adjectives)
    tagfilter = ('NN', 'NNP', 'NNS', 'NNPS',
                 'CD',
                 'JJ', 'JJR', 'JJS')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        files = kwargs.get('files', {})
        self.dataset_name = kwargs.get('dataset_name')
        # summary length restriction
        self.max_length = 100
        # Set this to 0 is every sentence length is allowed
        self.min_sentence_length = 0
        # initiate stopword lists & w2v models
        self.stopwords = {}
        self.w2v_models = {}
        model_path = 'models/word2vec/blendle/word2vec_blendle'
        self.w2v_models['blendle'] = W2VModel('blendle',
                                              Word2Vec.load(files.get(model_path, model_path)).wv)
        model_path = 'models/word2vec/google/GoogleNews-vectors-negative300.bin'
        self.w2v_models['google'] = W2VModel('google',
                                             KeyedVectors.load_word2vec_format(files.get(model_path, model_path), binary=True))
        stopword_path = 'models/stopwords.json'
        with open(files.get(stopword_path, stopword_path)) as fh:
            self.stopwords = json.load(fh)

        # Load model state for inference
        # Only the normally trained models (no pre- or co-training) are included.
        self.USE_CUDA = False
        google_rnn_parameters = torch.load(
            'rnn/saved_models/google/rnn_normal',
            map_location=(lambda storage, loc: storage))

        blendle_rnn_parameters = torch.load(
            'rnn/saved_models/blendle/rnn_normal',
            map_location=(lambda storage, loc: storage))

        # Initiate model in eval mode with trained parameters
        self.rnn_model = {}
        self.rnn_model['blendle_normal'] = RNNModel(self.USE_CUDA, blendle_rnn_parameters)
        self.rnn_model['google_normal'] = RNNModel(self.USE_CUDA, google_rnn_parameters)

        # load other models
        self.svd_path = 'bin/summarization/bigram_svd_{}.model'.format(self.dataset_name)
        self.svd = sk_joblib.load(files.get(self.svd_path, self.svd_path))
        self.bigram_path = 'bin/summarization/bigrams_{}.pkl'.format(self.dataset_name)
        self.final_bigrams = pickle.load(open(self.bigram_path, 'rb'))

    def __call__(self, data):
        summaries = {}
        rankings = {}
        splitted_body = self.get_enrichment(data, 'sentence_splitter')
        sentences = list(self.get_tokenized_sentences(splitted_body))
        tokenized_body = self.get_enrichment(data, 'tokenizer')
        tokenized = list(self.get_tokenized_sentences(tokenized_body))
        stemmed = self.get_enrichment(data, 'stemmer')
        postagged = self.get_enrichment(data, 'postagger')
        stopword_list = self.stopwords
        w2v_models = self.w2v_models

        # Sentence representations:
        w2v_indices_google, w2v_means_google = repr.w2v_sentence_means(tokenized, w2v_models['google'])
        w2v_indices_blendle, w2v_means_blendle = repr.w2v_sentence_means(tokenized, w2v_models['blendle'])
        tfidf_indices, tfidf_repr, idf_weight_dict = repr.tfidf_repr(sentences, stopword_list, ngram_range=(1,1)) # Option to pass more than just unigrams
        w2v_tfidf_indices_blendle, w2v_tfidf_sums_blendle = repr.w2v_sentence_sums_tfidf(tokenized, w2v_models['blendle'], idf_weight_dict)
        w2v_tfidf_indices_google, w2v_tfidf_sums_google = repr.w2v_sentence_sums_tfidf(tokenized, w2v_models['google'], idf_weight_dict)
        bigram_indices, bigram_repr     = repr.bigram_repr(stemmed=stemmed, stopwords=stopword_list, svd=self.svd, bigram_list=self.final_bigrams)
        blendle_sif_indices, blendle_sif_embeddings = repr.sif_embeddings(tokenized, w2v_models['blendle'])
        google_sif_indices, google_sif_embeddings = repr.sif_embeddings(tokenized, w2v_models['google'])


        # Rankings:
        # All rankings used in the thesis (except semi-supervised RNN models) are included below.
        rankings['lb_2010'] = modified_greedy(sentences=sentences,
                                              tokenized=tokenized,
                                              model=w2v_models['blendle'],
                                              stopwords=stopword_list,
                                              original_indices=tfidf_indices,
                                              sent_representations=tfidf_repr,
                                              objective_function=objectives.lin_bilmes_2010,
                                              min_sentence_length=self.min_sentence_length)

        rankings['mmr_w2v_blendle'] = modified_greedy(sentences=sentences,
                                                      tokenized=tokenized,
                                                      model=w2v_models['blendle'],
                                                      stopwords=stopword_list,
                                                      original_indices=w2v_tfidf_indices_blendle,
                                                      sent_representations=w2v_tfidf_sums_blendle,
                                                      objective_function=objectives.lin_bilmes_2010,
                                                      min_sentence_length=self.min_sentence_length)

        rankings['mmr_w2v_google'] = modified_greedy(sentences=sentences,
                                                     tokenized=tokenized,
                                                     model=w2v_models['google'],
                                                     stopwords=stopword_list,
                                                     original_indices=w2v_tfidf_indices_google,
                                                     sent_representations=w2v_tfidf_sums_google,
                                                     objective_function=objectives.lin_bilmes_2010,
                                                     min_sentence_length=self.min_sentence_length)

        rankings['lb_2011'] = modified_greedy(sentences=sentences,
                                              tokenized=tokenized,
                                              model=w2v_models['blendle'],
                                              stopwords=stopword_list,
                                              original_indices=tfidf_indices,
                                              sent_representations=tfidf_repr,
                                              objective_function=objectives.lin_bilmes_2011,
                                              min_sentence_length=self.min_sentence_length)

        rankings['kageback_blendle'] = modified_greedy(sentences=sentences,
                                                       tokenized=tokenized,
                                                       model=w2v_models['blendle'],
                                                       stopwords=stopword_list,
                                                       original_indices=w2v_tfidf_indices_blendle,
                                                       sent_representations=w2v_tfidf_sums_blendle,
                                                       objective_function=objectives.lin_bilmes_2011,
                                                       min_sentence_length=self.min_sentence_length)


        rankings['docemb_blendle'] = modified_greedy(sentences=sentences,
                                                     tokenized=tokenized,
                                                     model=w2v_models['blendle'],
                                                     stopwords=stopword_list,
                                                     original_indices=w2v_tfidf_indices_blendle,
                                                     sent_representations=w2v_tfidf_sums_blendle,
                                                     objective_function=objectives.doc_emb,
                                                     min_sentence_length=self.min_sentence_length)

        rankings['embdist_sentence_blendle'] = modified_greedy(sentences=sentences,
                                                               tokenized=tokenized,
                                                               model=w2v_models['blendle'],
                                                               stopwords=stopword_list,
                                                               original_indices=w2v_tfidf_indices_blendle,
                                                               sent_representations=w2v_tfidf_sums_blendle,
                                                               objective_function=objectives.emb_dist_sentence,
                                                               min_sentence_length=self.min_sentence_length)

        rankings['embdist_word_blendle'] = modified_greedy(sentences=sentences,
                                                           tokenized=tokenized,
                                                           model=w2v_models['blendle'],
                                                           stopwords=stopword_list,
                                                           original_indices=w2v_tfidf_indices_blendle,
                                                           sent_representations=w2v_tfidf_sums_blendle,
                                                           objective_function=objectives.emb_dist_word,
                                                           min_sentence_length=self.min_sentence_length)

        rankings['kageback_google'] = modified_greedy(sentences=sentences,
                                                      tokenized=tokenized,
                                                      model=w2v_models['google'],
                                                      stopwords=stopword_list,
                                                      original_indices=w2v_tfidf_indices_google,
                                                      sent_representations=w2v_tfidf_sums_google,
                                                      objective_function=objectives.lin_bilmes_2011,
                                                      min_sentence_length=self.min_sentence_length)

        rankings['docemb_google'] = modified_greedy(sentences=sentences,
                                                    tokenized=tokenized,
                                                    model=w2v_models['google'],
                                                    stopwords=stopword_list,
                                                    original_indices=w2v_tfidf_indices_google,
                                                    sent_representations=w2v_tfidf_sums_google,
                                                    objective_function=objectives.doc_emb,
                                                    min_sentence_length=self.min_sentence_length)

        rankings['embdist_sentence_google'] = modified_greedy(sentences=sentences,
                                                              tokenized=tokenized,
                                                              model=w2v_models['google'],
                                                              stopwords=stopword_list,
                                                              original_indices=w2v_tfidf_indices_google,
                                                              sent_representations=w2v_tfidf_sums_google,
                                                              objective_function=objectives.emb_dist_sentence,
                                                              min_sentence_length=self.min_sentence_length)

        rankings['embdist_word_google'] = modified_greedy(sentences=sentences,
                                                          tokenized=tokenized,
                                                          model=w2v_models['google'],
                                                          stopwords=stopword_list,
                                                          original_indices=w2v_tfidf_indices_google,
                                                          sent_representations=w2v_tfidf_sums_google,
                                                          objective_function=objectives.emb_dist_word,
                                                          min_sentence_length=self.min_sentence_length)

        rankings['msv'] = msv(sentences=sentences,
                              original_indices=bigram_indices,
                              sent_representations=bigram_repr)

        rankings['textrank_complete'] = textrank(sentences=sentences,
                                                 original_indices=list(range(len(sentences))),
                                                 stemmed=stemmed,
                                                 postagged=postagged,
                                                 stopwords=stopword_list,
                                                 sent_representations=None,
                                                 tagfilter=self.tagfilter,
                                                 vectorized=False)

        rankings['textrank_w2v_google'] = textrank(sentences=sentences,
                                                   original_indices=w2v_indices_google,
                                                   stemmed=stemmed,
                                                   postagged=postagged,
                                                   stopwords=stopword_list,
                                                   sent_representations=w2v_means_google,
                                                   tagfilter=self.tagfilter,
                                                   vectorized=True)

        rankings['textrank_w2v_blendle'] = textrank(sentences=sentences,
                                                    original_indices=w2v_indices_blendle,
                                                    stemmed=stemmed,
                                                    postagged=postagged,
                                                    stopwords=stopword_list,
                                                    sent_representations=w2v_means_blendle,
                                                    tagfilter=self.tagfilter,
                                                    vectorized=True)

        rankings['textrank_tfidf_w2v_google'] = textrank(sentences=sentences,
                                                         original_indices=w2v_tfidf_indices_google,
                                                         stemmed=stemmed,
                                                         postagged=postagged,
                                                         stopwords=stopword_list,
                                                         sent_representations=w2v_tfidf_sums_google,
                                                         tagfilter=self.tagfilter,
                                                         vectorized=True)

        rankings['textrank_tfidf_w2v_blendle'] = textrank(sentences=sentences,
                                                          original_indices=w2v_tfidf_indices_blendle,
                                                          stemmed=stemmed,
                                                          postagged=postagged,
                                                          stopwords=stopword_list,
                                                          sent_representations=w2v_tfidf_sums_blendle,
                                                          tagfilter=self.tagfilter,
                                                          vectorized=True)

        rankings['textrank_sif_blendle'] = textrank(sentences=sentences,
                                                    original_indices=blendle_sif_indices,
                                                    stemmed=stemmed,
                                                    postagged=postagged,
                                                    stopwords=stopword_list,
                                                    sent_representations=blendle_sif_embeddings,
                                                    tagfilter=self.tagfilter,
                                                    vectorized=True)

        rankings['textrank_sif_google'] = textrank(sentences=sentences,
                                                   original_indices=google_sif_indices,
                                                   stemmed=stemmed,
                                                   postagged=postagged,
                                                   stopwords=stopword_list,
                                                   sent_representations=google_sif_embeddings,
                                                   tagfilter=self.tagfilter,
                                                   vectorized=True)

        rankings['rnn_ranker_blendle'] = rnn_ranker(w2v_sentence_means=w2v_means_blendle,
                                                                      original_indices=w2v_indices_blendle,
                                                                      sentences=sentences,
                                                                      output_size=2,
                                                                      rnn_model=self.rnn_model['blendle_normal'],
                                                                      USE_CUDA=self.USE_CUDA)

        rankings['rnn_ranker_google'] = rnn_ranker(w2v_sentence_means=w2v_means_google,
                                                                      original_indices=w2v_indices_google,
                                                                      sentences=sentences,
                                                                      output_size=2,
                                                                      rnn_model=self.rnn_model['google_normal'],
                                                                      USE_CUDA=self.USE_CUDA)

        rankings['lead'] = [(i, s) for i, s in enumerate(sentences)]

        # Now construct summaries from rankings.
        for summ_method, ranking in rankings.items():
            summary_length = 0
            summary_sentences = []
            for index, sentence in ranking:
                # Tokens are whitespace-delimited words.
                sentence_length = len(sentence.split())
                if sentence_length <= self.min_sentence_length:
                    continue
                summary_length += sentence_length
                summary_sentences.append((index, sentence))
                if summary_length >= self.max_length:
                    break

            # Result: chronologically ordered list of most important sentences
            summaries[summ_method] = [(index, sentence) for index, sentence in sorted(summary_sentences)]


        return self.add_enrichment(data, self.name, summaries)
