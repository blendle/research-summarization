import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
import string

def modified_greedy(sentences,
                    tokenized,
                    model,
                    stopwords,
                    original_indices,
                    sent_representations,
                    objective_function,
                    min_sentence_length):
    """Implementation of the MMR summarizer as described in Lin & Bilmes (2010)."""

    # Initialize stuff
    # Ground set indices: all indices, stays constant throughout the function
    all_indices = tuple(range(len(original_indices)))
    # Candidate indices: all candidates (gets smaller every iteration)
    candidate_indices = list(range(len(original_indices)))
    # Summary indices: indices of represented sentences added to summary
    summary_indices = []
    # Scaling factor (r) is taken from original paper: r = 0.3
    scaling_factor = .3

    # Tf-idf clustering, as described in Lin & Bilmes (2011)
    n_clusters = len(original_indices) // 5
    k_means = KMeans(n_clusters=n_clusters, random_state=42)
    clustering = k_means.fit_predict(sent_representations)
    clustered_indices = [np.array(all_indices)[np.where(clustering == i)].tolist()
                         for i in range(n_clusters)]

    # Make document vector (since w2v sentences are now sums, it is this easy):
    document_vector = np.sum(sent_representations, axis=0)

    # Pick the right sentences from sentence list (to match representation matrix)
    sentences = [sentences[i] for i in original_indices]
    tokenized = [tokenized[i] for i in original_indices]


    # Construct bag of words from representable sentences
    preprocessed = (sentence.lower().split(' ')
                    for i, sentence in enumerate(tokenized))
    # POS-tag filtering, and punctuation removal
    preprocessed = [[word.translate(str.maketrans('', '', string.punctuation))
                     for word in sentence] for sentence in preprocessed]
    # Remove OOV words
    sentence_words = [[word for word in sentence if word in model.model.vocab]
                      for sentence in preprocessed]
    # Deduplicate & flatten
    bag_of_words = list(set([word for sentence in sentence_words for word in sentence]))
    # Look up in-vocabulary word vectors
    vectorized = [(word, model.model[word]) for word in bag_of_words]
    # Construct word similarity matrix for all words in article object
    names, vectors = zip(*vectorized)
    # word_distance_matrix = pairwise_distances(vectors, metric='euclidean')
    word_distance_matrix = pairwise_distances(vectors, metric='cosine')
    # Pandas workaround
    name_index_tuples = list(zip(list(range(len(names))), names))
    # Fill diagonal with nan, to make sure it's never the minimum
    np.fill_diagonal(word_distance_matrix, np.nan)

    # Compute sentence similarity matrix based on sentence representations
    distance_matrix = pairwise_distances(sent_representations, metric='cosine')
    similarity_matrix = np.subtract(1, distance_matrix)
    np.fill_diagonal(similarity_matrix, np.nan)

    # Compute sentence lengths
    sentence_lengths = [len(s.split()) for s in sentences]
    length_scaler = np.power(sentence_lengths, scaling_factor).tolist()

    # Remove sentences that do not have similarity with other sentences from candidate set
    similarity_sum_per_sentence = np.nansum(similarity_matrix, axis=0)
    irrelevant_indices = np.where(similarity_sum_per_sentence == 0)[0].tolist()
    candidate_indices = [index for index in candidate_indices
                         if index not in irrelevant_indices]

    # Already save the best singleton summary, for comparison to iterative result later
    singleton_scores = [objective_function(similarity_matrix,
                                           sent_representations,
                                           name_index_tuples,
                                           sentence_words,
                                           word_distance_matrix,
                                           document_vector,
                                           clustered_indices,
                                           all_indices,
                                           [i])
                        if sentence_lengths[i] <= 100
                        else np.nan for i in candidate_indices]
    best_singleton_score = np.nanmax(singleton_scores)
    # Note that the singleton index is directly translated to a sentence representation index
    best_singleton_index = candidate_indices[np.nanargmax(singleton_scores)]

    # Greedily add sentences to summary
    summary_length = 0
    for iteration in range(len(sentence_lengths)):
        print("Iteration {}".format(iteration))

        # Edge case: value of objective function when summary is empty.
        if iteration == 0:
            current_score = 0.
        else:
            current_score = objective_function(similarity_matrix,
                                               sent_representations,
                                               name_index_tuples,
                                               sentence_words,
                                               word_distance_matrix,
                                               document_vector,
                                               clustered_indices,
                                               all_indices,
                                               summary_indices)

        # Compute all relevant new scores
        new_scores = [objective_function(similarity_matrix,
                                         sent_representations,
                                         name_index_tuples,
                                         sentence_words,
                                         word_distance_matrix,
                                         document_vector,
                                         clustered_indices,
                                         all_indices,
                                         summary_indices+[i])
                      if sentence_lengths[i] > min_sentence_length
                      else np.nan
                      for i in candidate_indices]

        # If there are no candidates left, break the loop
        if all(np.isnan(score) for score in new_scores):
            break

        # Remove non-candidate elements from length scaler to fit arrays
        current_length_scaler = [v for i, v in enumerate(length_scaler) if i in candidate_indices]
        added_values = np.divide(np.subtract(new_scores, current_score), current_length_scaler)
        best_index = np.nanargmax(added_values)

        # Pass best index if the sentence does not increase MMR-score (+ empty summary edge case)
        if not new_scores[best_index] - current_score >= 0 and summary_indices:
            candidate_indices.pop(best_index)
        else:
            summary_indices.append(candidate_indices[best_index])
            summary_length += sentence_lengths[candidate_indices[best_index]]
            candidate_indices.pop(best_index)
            if summary_length >= 100:
                break

    # Last step: compare singleton score with summary score, and pick best as summary
    final_summary_score = objective_function(similarity_matrix,
                                             sent_representations,
                                             name_index_tuples,
                                             sentence_words,
                                             word_distance_matrix,
                                             document_vector,
                                             clustered_indices,
                                             all_indices,
                                             summary_indices)
    if best_singleton_score >= final_summary_score:
        ranked_sentences = [sentences[i] for i in [best_singleton_index]]
        ranking = list(zip([best_singleton_index], ranked_sentences))
    else:
        ranked_sentences = [sentences[i] for i in summary_indices]
        ranking = list(zip(summary_indices, ranked_sentences))

    # Replace filtered indices with original ones
    ranking = [(original_indices[i], s) for i, s in ranking]

    return ranking
