from enrichers.base import Enricher
from nltk.data import load as nltk_load
import json
import regex
from itertools import chain


class SentenceSplitter(Enricher):
    name = 'sentence_splitter'
    persistent = False
    requires = ('cleaner',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        files = kwargs.get('files', {})
        self.dataset_name = kwargs.get('dataset_name', '')
        # Load sentence splitter, and update with abbreviation list
        model_path = 'models/sent_tokenizer.pickle'
        self.sent_splitter = nltk_load('file:' + files.get(model_path, model_path), format='pickle')
        abbrev_path = 'models/abbreviations.json'
        with open(files.get(abbrev_path, abbrev_path)) as fh:
            abbreviations = json.load(fh)
        self.sent_splitter._params.abbrev_types.update(abbreviations)

    def __call__(self, data):
        sent_splitter = self.sent_splitter

        splitted = []
        for e in self.get_elements(data):
            if 'content' in e and e['content']:
                # DUC data is already sentence splitted
                if self.dataset_name == 'duc':
                    sentences = [e['content']]
                else:
                    # Split element into list of sentences (without separating word tokens)
                    sentences = sent_splitter.tokenize(e['content'])
                    # Additionally split on double newlines (incl. its surrounding whitespace)
                    sentences = [regex.split('\s*(?:\n\n)\s*', x) for x in sentences]
                    # Flatten the resulting list
                    sentences = list(chain(*sentences))
                splitted.append({'content': sentences, 'type': e['type']})

        return self.add_enrichment(data, self.name, splitted)
