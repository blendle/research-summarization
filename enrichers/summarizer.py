import json
import pickle
from gensim.models.word2vec import Word2Vec, KeyedVectors
from sklearn.externals import joblib
import representations as repr
from sent_selectors.modified_greedy import modified_greedy
from sent_selectors.msv import msv
from sent_selectors.textrank import textrank
from sent_selectors import greedy_objectives as objectives
from enrichers.base import Enricher


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
        # summary length restriction
        self.max_length = 100
        # initiate stopword lists & w2v models
        with open('models/stopwords.json') as fh:
            self.stopword_list = json.load(fh)
        model_path = 'models/word2vec/blendle/word2vec_blendle'
        self.blendle_model = Word2Vec.load(model_path).wv
        model_path = 'models/word2vec/google/GoogleNews-vectors-negative300.bin'
        self.google_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

        # load other models
        dataset_name = kwargs.get('dataset_name')
        svd_path = 'models/bigram_svd_{}.model'.format(dataset_name)
        bigram_path = 'models/bigrams_{}.pkl'.format(dataset_name)
        self.svd = joblib.load(svd_path)
        self.final_bigrams = pickle.load(open(bigram_path, 'rb'))

    def __call__(self, data):
        summaries = {}
        rankings = {}
        splitted_body = self.get_enrichment(data, 'sentence_splitter')
        sentences = list(self.get_tokenized_sentences(splitted_body))
        tokenized_body = self.get_enrichment(data, 'tokenizer')
        tokenized = list(self.get_tokenized_sentences(tokenized_body))
        stemmed = self.get_enrichment(data, 'stemmer')
        postagged = self.get_enrichment(data, 'postagger')
        google_model = self.google_model
        blendle_model = self.blendle_model

        # Sentence representations:
        # Normal tf-idf (+ idf weights for reweighting)
        tfidf_indices, tfidf_repr, idf_weight_dict = repr.tfidf_repr(sentences, self.stopword_list, ngram_range=(1,1))
        # Blendle tf-idf reweighted additive sentence embeddings
        w2v_tfidf_indices_blendle, w2v_tfidf_sums_blendle = repr.w2v_sentence_sums_tfidf(tokenized, blendle_model, idf_weight_dict)
        # Google tf-idf reweighted additive sentence embeddings
        w2v_tfidf_indices_google, w2v_tfidf_sums_google = repr.w2v_sentence_sums_tfidf(tokenized, google_model, idf_weight_dict)
        # Bigram representations for Yogatama et al. (2015)
        bigram_indices, bigram_repr     = repr.bigram_repr(stemmed=stemmed, stopwords=self.stopword_list, svd=self.svd, bigram_list=self.final_bigrams)


        # Compute one or more sentence rankings based on representations

        # rankings['lb_2010'] = modified_greedy(sentences=sentences,
        #                                   tokenized=tokenized,
        #                                   model=blendle_model,
        #                                   stopwords=self.stopword_list,
        #                                   original_indices=tfidf_indices,
        #                                   sent_representations=tfidf_repr,
        #                                   objective_function=objectives.lin_bilmes_2010)
        #
        # rankings['mmr_w2v_blendle'] = modified_greedy(sentences=sentences,
        #                                       tokenized=tokenized,
        #                                       model=blendle_model,
        #                                       stopwords=self.stopword_list,
        #                                       original_indices=w2v_tfidf_indices_blendle,
        #                                       sent_representations=w2v_tfidf_sums_blendle,
        #                                       objective_function=objectives.lin_bilmes_2010)

        # rankings['mmr_w2v_google'] = modified_greedy(sentences=sentences,
        #                                               tokenized=tokenized,
        #                                               model=google_model,
        #                                               stopwords=self.stopword_list,
        #                                               original_indices=w2v_tfidf_indices_google,
        #                                               sent_representations=w2v_tfidf_sums_google,
        #                                               objective_function=objectives.lin_bilmes_2010)

        rankings['lb_2011'] = modified_greedy(sentences=sentences,
                                              tokenized=tokenized,
                                              model=blendle_model,
                                              stopwords=self.stopword_list,
                                              original_indices=tfidf_indices,
                                              sent_representations=tfidf_repr,
                                              objective_function=objectives.lin_bilmes_2011)

        # rankings['kageback_blendle'] = modified_greedy(sentences=sentences,
        #                                       tokenized=tokenized,
        #                                       model=blendle_model,
        #                                       stopwords=self.stopword_list,
        #                                       original_indices=w2v_tfidf_indices_blendle,
        #                                       sent_representations=w2v_tfidf_sums_blendle,
        #                                       objective_function=objectives.lin_bilmes_2011)
        #
        #
        # rankings['docemb_blendle'] = modified_greedy(sentences=sentences,
        #                                      tokenized=tokenized,
        #                                      model=blendle_model,
        #                                      stopwords=self.stopword_list,
        #                                      original_indices=w2v_tfidf_indices_blendle,
        #                                      sent_representations=w2v_tfidf_sums_blendle,
        #                                      objective_function=objectives.doc_emb)
        #
        # rankings['embdist_sentence_blendle'] = modified_greedy(sentences=sentences,
        #                                       tokenized=tokenized,
        #                                       model=blendle_model,
        #                                       stopwords=self.stopword_list,
        #                                       original_indices=w2v_tfidf_indices_blendle,
        #                                       sent_representations=w2v_tfidf_sums_blendle,
        #                                       objective_function=objectives.emb_dist_sentence)
        #
        # rankings['embdist_word_blendle'] = modified_greedy(sentences=sentences,
        #                                                tokenized=tokenized,
        #                                                model=blendle_model,
        #                                                stopwords=self.stopword_list,
        #                                                original_indices=w2v_tfidf_indices_blendle,
        #                                                sent_representations=w2v_tfidf_sums_blendle,
        #                                                objective_function=objectives.emb_dist_word)
        #
        # rankings['kageback_google'] = modified_greedy(sentences=sentences,
        #                                        tokenized=tokenized,
        #                                        model=google_model,
        #                                        stopwords=self.stopword_list,
        #                                        original_indices=w2v_tfidf_indices_google,
        #                                        sent_representations=w2v_tfidf_sums_google,
        #                                        objective_function=objectives.lin_bilmes_2011)

        # rankings['docemb_google'] = modified_greedy(sentences=sentences,
        #                                      tokenized=tokenized,
        #                                      model=google_model,
        #                                      stopwords=self.stopword_list,
        #                                      original_indices=w2v_tfidf_indices_google,
        #                                      sent_representations=w2v_tfidf_sums_google,
        #                                      objective_function=objectives.doc_emb)

        # rankings['embdist_sentence_google'] = modified_greedy(sentences=sentences,
        #                                                tokenized=tokenized,
        #                                                model=google_model,
        #                                                stopwords=self.stopword_list,
        #                                                original_indices=w2v_tfidf_indices_google,
        #                                                sent_representations=w2v_tfidf_sums_google,
        #                                                objective_function=objectives.emb_dist_sentence)
        #
        # rankings['embdist_word_google'] = modified_greedy(sentences=sentences,
        #                                            tokenized=tokenized,
        #                                            model=google_model,
        #                                            stopwords=self.stopword_list,
        #                                            original_indices=w2v_tfidf_indices_google,
        #                                            sent_representations=w2v_tfidf_sums_google,
        #                                            objective_function=objectives.emb_dist_word)
        #
        # rankings['msv'] = msv(sentences=sentences,
        #                       original_indices=bigram_indices,
        #                       sent_representations=bigram_repr)

        # rankings['textrank_complete'] = textrank(sentences=sentences,
        #                                 original_indices=list(range(len(sentences))),
        #                                 stemmed=stemmed,
        #                                 postagged=postagged,
        #                                 stopwords=self.stopword_list,
        #                                 sent_representations=None,
        #                                 tagfilter=self.tagfilter,
        #                                 vectorized=False)
        #
        # rankings['textrank_tfidf'] = textrank(sentences=sentences,
        #                                           original_indices=tfidf_indices,
        #                                           stemmed=stemmed,
        #                                           postagged=postagged,
        #                                           stopwords=self.stopword_list,
        #                                           sent_representations=tfidf_repr,
        #                                           tagfilter=self.tagfilter,
        #                                           vectorized=True)

        # rankings['textrank_tfidf_w2v_google'] = textrank(sentences=sentences,
        #                                          original_indices=w2v_tfidf_indices_google,
        #                                          stemmed=stemmed,
        #                                          postagged=postagged,
        #                                          stopwords=self.stopword_list,
        #                                          sent_representations=w2v_tfidf_sums_google,
        #                                          tagfilter=self.tagfilter,
        #                                          vectorized=True)

        # rankings['textrank_tfidf_w2v_blendle'] = textrank(sentences=sentences,
        #                                                  original_indices=w2v_tfidf_indices_blendle,
        #                                                  stemmed=stemmed,
        #                                                  postagged=postagged,
        #                                                  stopwords=self.stopword_list,
        #                                                  sent_representations=w2v_tfidf_sums_blendle,
        #                                                  tagfilter=self.tagfilter,
        #                                                  vectorized=True)

        # Construct a summary of max 100 words from the ranking
        for summ_method, ranking in rankings.items():
            summary_length = 0
            summary_sentences = []
            for index, sentence in ranking:
                # Tokens for DUC/TAC are whitespace-delimited words.
                sentence_length = len(sentence.split())
                if summary_length + sentence_length > self.max_length:
                    continue
                summary_length += sentence_length
                summary_sentences.append((index, sentence))

            # Result: chronologically ordered list of most important sentences
            summaries[summ_method] = [sentence for index, sentence in sorted(summary_sentences)]

        return self.add_enrichment(data, self.name, summaries)
