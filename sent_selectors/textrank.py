from scipy import sparse
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances

def textrank(sentences,
             original_indices,
             stemmed,
             postagged,
             stopwords,
             sent_representations,
             tagfilter,
             vectorized=False):
    """Ranks sentences according to Google's TextRank algorithm.

    Reference: https://digital.library.unt.edu/ark:/67531/metadc30962/.

    Args:
        tokenized_sentences: List of sentences from the tokenizer.

    Returns:
        A list of (sentences, index)-tuples , reverse sorted on their score
    """
    # Pick the right sentences from sentence list (to match representation matrix)
    sentences = [sentences[i] for i in original_indices]

    if vectorized == False:
        # if vectorized == False, use original textrank similarity for similarity matrix
        preprocessed_sentences = [[stem for word_index, (word, stem) in enumerate(sentence)
                                   if postagged[sentence_index][word_index][1] in tagfilter
                                   and not word in stopwords
                                   ]
                                  for sentence_index, sentence in enumerate(stemmed)]
        similarity_matrix = sparse.csr_matrix([[similarity_textrank(s1, s2)
                                                if s1 and s2 else 0.
                                                for s1 in preprocessed_sentences]
                                               for s2 in preprocessed_sentences])
        # Use PageRank algorithm on similarity matrix
        nx_graph = nx.from_scipy_sparse_matrix(similarity_matrix)

    else:
        # if vectorized == True, use vectorized sentence representations for similarity matrix
        distance_matrix = pairwise_distances(sent_representations, metric='cosine')
        similarity_matrix = np.subtract(1, distance_matrix)
        # Clip negative values, as these are not meaningful for TextRank (is
        # only needed PC-removed sentence vectors, as normal averages are apparently
        # in positive space)
        similarity_matrix = similarity_matrix.clip(0)
        # Use PageRank algorithm on similarity matrix
        nx_graph = nx.from_numpy_matrix(similarity_matrix)

    # Convergence of graph (tolerance from TextRank paper)
    scores = nx.pagerank_scipy(nx_graph, max_iter=100, tol=1e-04)

    # Return list of sentences & indices,
    # reverse sorted on their score,
    # and replace the filtered indices with the original ones.
    sorted_sentences = sorted(((scores[i], s, original_indices[i])
                               for i, s in enumerate(sentences)),
                              reverse=True)

    return [(i, s) for (score, s, i) in sorted_sentences]

def similarity_textrank(s1, s2):
    """Original TextRank sentence similarity.

    Args:
        s1: Already preprocessed sentence 1 as a list of words.
        s2: Already preprocessed sentence 2 as a list of words.
    Returns:
        Similarity score.
    """
    # Edge case: when one of the sentences contain no words, return 0.
    if ((len(s1) == 0) or (len(s2) == 0)):
        return 0.

    # Edge case: when both sentences contain 1 word, return 0.
    if ((len(s1) == 1) and (len(s2) == 1)):
        return 0.

    # Numerator: number of words occurring in both sentences
    numerator = len([word for word in s1 if word in s2])
    # Denominator: logs of number of words per sentence
    denominator = np.log(len(s1)) + np.log(len(s2))

    return numerator / denominator
