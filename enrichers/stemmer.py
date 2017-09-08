from nltk.stem.snowball import SnowballStemmer
from enrichers.base import Enricher


class Stemmer(Enricher):
    name = 'stemmer'
    persistent = False
    requires = ('cleaner', 'sentence_splitter', 'tokenizer')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stemmer = SnowballStemmer('english')

    def __call__(self, data):
        tokenized_body = self.get_enrichment(data, 'tokenizer')

        stemmed_sentences = []
        for sentence in self.get_tokenized_sentences(tokenized_body):
            # the tokenizer outputs space-separated token strings
            # per sentence, so have to split them here again
            tokens = sentence.split(' ')
            stems = [self.stemmer.stem(token) for token in tokens]
            stemmed_sentences.append(list(zip(tokens, stems)))

        return self.add_enrichment(data, self.name, stemmed_sentences)
