import numpy as np
from sklearn.decomposition import TruncatedSVD
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_repr(sentences, stopwords, ngram_range):
    """
    Args:
        stemmed: List of sentences, every sentence as a string.
        stopwords: List of stopwords.

    Returns: TF*IDF representation of every sentence (np-array (#sentences * #tfs))

    """
    tfidf = TfidfVectorizer(stop_words=stopwords, ngram_range=ngram_range)
    tfidf_matrix = tfidf.fit_transform(sentences).toarray()

    idf_weight_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    # Produce of dummy original indices (as in this case, all sentences are represented)
    original_indices = list(range(len(sentences)))

    return original_indices, tfidf_matrix, idf_weight_dict


def bigram_repr(stemmed, stopwords, svd, bigram_list):
    multi_hot_bigram_vectors = []
    for sentence in stemmed:
        # Exclude stopwords and punctuation and use stemmed words
        split_sentence = [stem for word, stem in sentence]
        # Turn sentence into list of bigrams
        bigrams = list(nltk.bigrams(split_sentence))
        # Make TF vector of bigram list, based on all bigrams in the data set
        multi_hot = np.array([v if k in bigrams else 0 for k, v in bigram_list])
        multi_hot_bigram_vectors.append(multi_hot)

    # Filter out non-hot vectors, and save original indices
    repr_list_with_orig_indices = [(i, s) for i, s in enumerate(multi_hot_bigram_vectors) if
                                   np.sum(s) != 0]

    original_indices, sentence_representations = zip(*repr_list_with_orig_indices)

    sentence_array = np.array(sentence_representations)
    sentence_repr = svd.transform(sentence_array)

    return original_indices, sentence_repr


def w2v_sentence_sums(tokenized, model, postagged, tagfilter):
    # define minimum number of words in the sentence that are also in the word model.
    min_word_vectors = 1

    # lowercase & split tokenized sentences
    preprocessed = [sentence.lower().split(' ')
                    for sentence in tokenized]

    # POS-tag filtering, and punctuation removal
    preprocessed = [[word.translate(str.maketrans('', '', string.punctuation))
                     for word_index, word in enumerate(sentence)
                     # if postagged[sentence_index][word_index][1] in tagfilter
                     ]
                    for sentence_index, sentence in enumerate(preprocessed)]

    vectorized = [[model[word] for word in sentence
                   if word in model.vocab
                   ]
                  for sentence in preprocessed]

    sentence_sums_with_indices = [(index, np.sum(s, axis=0))
                                      for index, s in enumerate(vectorized)
                                      if len(s) >= min_word_vectors]

    # With this, we can obtain original sentences by doing sentences[original_indices[index_of_vector]]
    original_indices, sentence_sums = zip(*sentence_sums_with_indices)

    return original_indices, np.array(sentence_sums)


def w2v_sentence_sums_tfidf(tokenized, model, idf_weight_dict):
    # define minimum number of words in the sentence that are also in the word model.
    min_word_vectors = 1

    # lowercase & split tokenized sentences
    preprocessed = [sentence.lower().split(' ')
                    for sentence in tokenized]

    # POS-tag filtering, and punctuation removal
    preprocessed = [[word.translate(str.maketrans('', '', string.punctuation))
                     for word in sentence
                     ]
                    for sentence in preprocessed]

    # Remove OOV and non-TFIDF words
    vectorized = [[model[word]*idf_weight_dict[word] for word in sentence
                   if word in model.vocab
                   and word in idf_weight_dict
                   ]
                  for sentence in preprocessed]

    sentence_sums_with_indices = [(index, np.sum(s, axis=0))
                                      for index, s in enumerate(vectorized)
                                      if len(s) >= min_word_vectors]

    # With this, we can obtain original sentences by doing sentences[original_indices[index_of_vector]]
    original_indices, sentence_sums = zip(*sentence_sums_with_indices)

    return original_indices, np.array(sentence_sums)


def w2v_sentence_sums_pca(tokenized, model, postagged, tagfilter):
    original_indices, sentence_sums = w2v_sentence_sums(tokenized,
                                                                model,
                                                                postagged,
                                                                tagfilter)

    # Remove first principle component from sentence vectors
    sentence_sums = remove_pc(sentence_sums, npc=1)

    return original_indices, sentence_sums


def compute_pc(X, npc=1):
    """
    Compute the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=42)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    pc = compute_pc(X, npc)
    if npc == 1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX