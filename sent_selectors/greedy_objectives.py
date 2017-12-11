import numpy as np
from typing import Dict, Tuple, List
from scipy import spatial

def lin_bilmes_2010(similarity_matrix: np.ndarray,
                    sent_representations,
                    name_index_tuples,
                    sentence_words,
                    word_distance_matrix,
                    document_vector,
                    clustered_indices,
                    all_indices: Tuple[int],
                    summary_indices: List[int]):
    """
    Adapted MMR function as described in Lin & Bilmes (2010),
    used as objective function in mmr_greedy().

    Args:
        similarity_matrix: Numpy array of similarities between all represented sentences.
        candidate_indices: Indices of candidate sentences.
        summary_indices: Indices of (hypothetical) summary.

    Returns:
        Score indicating the quality of the (hypothetical) summary.
    """
    # Lambda-value is taken from original paper.
    l_weight = 4

    cut_indices = [i for i in all_indices if i not in summary_indices]
    graph_cut_score = np.sum(similarity_matrix[np.ix_(cut_indices, summary_indices)])
    # Use nansum in order to skip nan values (= reflexive similarities) in similarity matrix
    redundance_penalty = np.nansum(similarity_matrix[np.ix_(summary_indices, summary_indices)])

    score = graph_cut_score - l_weight * redundance_penalty

    return score

def lin_bilmes_2011(similarity_matrix: np.ndarray,
                    sent_representations,
                    name_index_tuples,
                    sentence_words,
                    word_distance_matrix,
                    document_vector,
                    clustered_indices,
                    all_indices: Tuple[int],
                    summary_indices: List[int]):
    """Submodular mixture (coverage & diversity) from Lin & Bilmes (2011)."""
    # Trade-off values alpha & lambda are taken from Figure 1 in Lin & Bilmes (2011).
    alpha = 5/len(sent_representations)
    l_weight = 6

    summary_indices_per_cluster = [[i for i in summary_indices if i in cluster]
                                   for cluster in clustered_indices]

    # Use nansum in order to skip nan values (= reflexive similarities) in similarity matrix
    overall_summ_similarity = np.nansum(similarity_matrix[summary_indices, :], axis=0)
    total_similarity = np.nansum(similarity_matrix, axis=0)
    similarity_minima = np.minimum(overall_summ_similarity, alpha * total_similarity)
    coverage_score = np.sum(similarity_minima)

    diversity_score = 0
    for indices in summary_indices_per_cluster:
        # Check whether the list is non-empty
        if indices:
            # Use nanmean in order to skip nan values (= reflexive similarities)
            mean_similarities = np.nanmean(similarity_matrix[indices, :], axis=1)
            cluster_score = np.sqrt(np.sum(mean_similarities))
            diversity_score += cluster_score

    score = coverage_score + l_weight * diversity_score

    return score

def doc_emb(similarity_matrix,
            sent_representations,
            name_index_tuples,
            sentence_words,
            word_distance_matrix,
            document_vector,
            clustered_indices,
            all_indices: Tuple[int],
            summary_indices):
    """
    Objective function from Kobayashi et al. (2015);
    similarity based on average embedding of summary and document.
    """
    # Retrieve sentence representations of summary sentences & sum them
    summary_sentence_reprs = [sent_representations[i] for i in summary_indices]
    summary_vector = np.sum(summary_sentence_reprs, axis=0)

    # Compute score: similarity of summary vector and document vector
    similarity = 1 - spatial.distance.cosine(summary_vector, document_vector)

    return similarity


def emb_dist_word(similarity_matrix,
                  sent_representations,
                  name_index_tuples,
                  sentence_words,
                  word_distance_matrix,
                  document_vector,
                  clustered_indices,
                  all_indices: Tuple[int],
                  summary_indices):
    """
    Objective function from Kobayashi et al. (2015);
    similarity based on embedding distributions of summary and document, on word level.
    """
    # Make (deduplicated) bag of words of all summary words
    summary_words = list(set([word for i, sentence in enumerate(sentence_words) for word in sentence
                     if i in summary_indices]))

    # Retrieve min distances from every article word to a summary word
    word_indices = [index for index, word in name_index_tuples if word in summary_words]
    relevant_distances = word_distance_matrix[word_indices]
    min_distances = np.nanmin(relevant_distances, axis=0)
    # Apply scaling function (which is identity function in our case)
    scaled = min_distances
    score = -np.nansum(scaled)

    return score


def emb_dist_sentence(similarity_matrix,
                      sent_representations,
                      name_index_tuples,
                      sentence_words,
                      word_distance_matrix,
                      document_vector,
                      clustered_indices,
                      all_indices: Tuple[int],
                      summary_indices):
    """
    Objective function from Kobayashi et al. (2015);
    similarity based on embedding distributions of summary and document, on sentence level.
    """
    distance_matrix = 1 - similarity_matrix
    relevant_distances = distance_matrix[summary_indices, :]

    min_distances = np.nanmin(relevant_distances, axis=0)
    # Apply scaling function (which is identity function in our case)
    scaled = min_distances
    score = -np.nansum(scaled)

    return score
