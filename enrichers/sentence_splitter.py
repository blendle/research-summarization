from enrichers.base import Enricher
from nltk.data import load as nltk_load
import regex
from itertools import chain


class SentenceSplitter(Enricher):
    name = 'sentence_splitter'
    persistent = False
    requires = ('cleaner',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sent_splitter = nltk_load('file:models/sent_tokenizer.pkl', format='pickle')

    def __call__(self, data):
        splitted = []
        for e in self.get_elements(data):
            if 'content' in e and e['content']:
                # Split element into list of sentences (without separating word tokens)
                sentences = self.sent_splitter.tokenize(e['content'])
                # Additionally split on double newlines (incl. its surrounding whitespace)
                sentences = [regex.split('\s*(?:\n\n)\s*', x) for x in sentences]
                # Flatten the resulting list
                sentences = list(chain(*sentences))
                splitted.append({'content': sentences, 'type': e['type']})

        return self.add_enrichment(data, self.name, splitted)
