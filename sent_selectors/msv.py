import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize

def msv(sentences,
        original_indices,
        sent_representations):
    """Reproduction of Yogatama et al. (2015)."""
    ranking = []
    indices = []
    bases = []

    # Compute cluster centroid (and convert to 2d-array for cdist)
    cluster_centroid = np.mean(sent_representations, axis=0)[None, :]

    # Pick the right sentences from sentence list (to match representation matrix)
    reprd_sentences = [sentences[i] for i in original_indices]

    # Add first sentence: farthest from cluster centroid
    distances = cdist(sent_representations, cluster_centroid, metric='cosine')
    index = np.argmax(distances)
    sentence = reprd_sentences[index]
    indices.append(index)
    ranking.append((index, sentence))
    base_vector = normalize(sent_representations[index][:, np.newaxis], axis=0).ravel()
    bases.append(base_vector)

    # Add other sentences: greedy furthest from subspace
    for i in range(len(reprd_sentences)-1):
        if i == 50:
            break
        print("Starting iteration {}".format(i))
        distances = np.array([distance_from_subspace(s, bases)
                              for s in sent_representations])

        distances[indices] = np.nan

        # Find index of furthest sentence
        index = np.nanargmax(distances)
        sentence = reprd_sentences[index]
        indices.append(index)
        ranking.append((index, sentence))
        base_vector = normalize(sent_representations[index][:, np.newaxis], axis=0).ravel()
        bases.append(base_vector)

    # Return list of indices & sentences,
    # and replace the filtered indices with the original ones.
    ranking = [(original_indices[i], s) for i, s  in ranking]

    return ranking

def distance_from_subspace(sentence_repr, bases):
    """Computation of distance from subspace as described in Yogatama et al. (2015)."""
    summed_projections = np.sum([projection_onto_base(sentence_repr, base)
                                                      for base in bases], axis=0)
    distance = np.linalg.norm(sentence_repr - summed_projections)

    return distance

def projection_onto_base(sentence_repr, base):
    return np.dot(sentence_repr, base) * base
