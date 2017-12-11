import os
import string
import numpy as np
import torch
from rnn.textrank import textrank_tagger

class DataIterator:
    def __init__(self, dirname, w2v_model, dataset_name='train'):
        self.w2v_model = w2v_model
        self.idx = 0
        self.dirname = dirname
        self.dataset_name = dataset_name
        # Construct a list of all summary filepaths
        self.file_list = []
        pname = os.path.join(self.dirname, self.dataset_name)
        for root, dirs, files in os.walk(pname):
            for name in files:
                if (".summary" in name) and not (name[0] == "."):
                    self.file_list.append(os.path.join(root, name))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            filename = self.file_list[self.idx]
        except IndexError:
            raise StopIteration()
        if self.idx % 5000 == 0:
            if self.idx == 0:
                print('Start preprocessing of {} data'.format(self.dataset_name), flush=True)
            else:
                print('Preprocessed {} samples'.format(self.idx), flush=True)
        self.idx += 1
        try:
            return file_to_data(filename, self.w2v_model)
        except ValueError:
            return self.__next__()
        except KeyError:
            return self.__next__()

def file_to_data(filename, model):
    """Transforms a single summary file into a data object for pipeline."""
    # open file
    with open(filename, 'r', encoding='utf-8') as fh:
        summary_file = fh.read()

    # Create keys for summary dict
    data_keys = ["url", "body", "summary", "entities"]

    # Split summary file into target values
    data_values = summary_file.split('\n\n')

    # Create dictionary from keys and summary file
    data = dict(zip(data_keys, data_values))

    # Transform 'sentences' into a list of tuples
    data['body'] = [labeled_sentence.split('\t\t\t')
                    for labeled_sentence in data['body'].split('\n')]

    # Transform 'entities' into a dict of all entities
    data['entities'] = dict([entity.split(":", 1) for entity in data['entities'].split('\n')])

    # Replace entity ids with entities in every sentence, and split sentence
    for labeled_sentence in data['body']:
        labeled_sentence[0] = insert_entities(labeled_sentence[0], data['entities'], filename)

    sentences, labels = zip(*data['body'])

    # Convert labels from type string to int
    labels = list(map(int, labels))

    indices, embeddings = w2v_sentence_means(sentences, model)

    # Remove labels of sentences that do not have an embedding
    labels = [label for i, label in enumerate(labels) if i in indices]

    # Change label 2 (might be extracted) to 0 (should not be extracted)
    labels = [0 if label == 2 else label for label in labels]

    # Compute which sentences are tagged by TextRank
    textrank_labels = textrank_tagger(sentences, model)

    # Turn both embeddings and labels into torch variables
    embeddings = torch.from_numpy(embeddings)
    labels = torch.LongTensor(labels)
    textrank_labels = torch.LongTensor(textrank_labels)

    return embeddings, labels, textrank_labels


def insert_entities(sentence, entities, filename):
    """Returns paragraph content with entities instead of entity ids."""
    split_p = [entities[word].lower().split(' ') if word[0:7] == '@entity' else [word]
               for word in sentence.split(' ')]

    flat_p = [word for split_elements in split_p
              for word in split_elements
              if not word in string.punctuation]

    # Return list of tokens per sentence
    return flat_p


def w2v_sentence_means(split_sentences, model):
    vectorized = [[model[word] for word in sentence
                   if word in model.vocab]
                  for sentence in split_sentences]

    sentence_means_with_indices = [(index, np.mean(s, axis=0))
                                  for index, s in enumerate(vectorized)
                                  if len(s) > 0]

    # With this, we can obtain original sentences by doing sentences[original_indices[index_of_vector]]
    original_indices, sentence_means = zip(*sentence_means_with_indices)

    return original_indices, np.array(sentence_means)
