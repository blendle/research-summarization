import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer


def textrank_tagger(tokenized, w2v_model):
    """
    TextRank based on cosine similarity between TF-IDF-reweighted w2v sentence sums.
    """
    idf_weights = idf_weight_dict(tokenized)

    original_indices, sent_representations = w2v_sentence_sums_tfidf(tokenized, w2v_model, idf_weights)

    distance_matrix = pairwise_distances(sent_representations, metric='cosine')
    similarity_matrix = np.subtract(1, distance_matrix)

    # Use PageRank algorithm on similarity matrix
    nx_graph = nx.from_numpy_matrix(similarity_matrix)

    # Convergence of graph (tolerance from TextRank paper)
    scores = nx.pagerank_scipy(nx_graph, max_iter=100, tol=1e-04)

    # For now, the number of summary-worthy sentences is set to ~33% of the sentences.
    cutoff = len(tokenized) // 3
    sorted_sentences = sorted([(scores[i], original_indices[i])
                               for i, s in enumerate(tokenized)],
                              reverse=True)
    summary_indices = [index for score, index in sorted_sentences[:cutoff]]
    labels = [1 if i in summary_indices else 0 for i, _ in enumerate(tokenized)]

    return labels


def w2v_sentence_sums_tfidf(tokenized, model, idf_weight_dict):
    # Remove OOV and non-TFIDF words
    vectorized = [[model[word]*idf_weight_dict[word] for word in sentence
                   if word in model.vocab and word in idf_weight_dict]
                  for sentence in tokenized]

    sentence_sums_with_indices = [(index, np.sum(s, axis=0))
                                  for index, s in enumerate(vectorized)
                                  if len(s) > 0]

    # With this, we can obtain original sentences by doing sentences[original_indices[index_of_vector]]
    original_indices, sentence_sums = zip(*sentence_sums_with_indices)

    return original_indices, np.array(sentence_sums)


def idf_weight_dict(tokenized):
    """
    Args:
        stemmed: List of sentences, every sentence as a string.
        stopwords: List of stopwords.

    Returns: TF*IDF representation of every sentence (np-array (#sentences * #tfs))

    """
    sentences = [' '.join(sentence) for sentence in tokenized]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(sentences)
    idf_weight_dict = dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    return idf_weight_dict