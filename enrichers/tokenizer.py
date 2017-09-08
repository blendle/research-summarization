from pattern.en import tokenize
from enrichers.base import Enricher


class Tokenizer(Enricher):
    name = 'tokenizer'
    persistent = False
    requires = ('sentence_splitter',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, data):
        splitted_body = self.get_enrichment(data, 'sentence_splitter')

        tokenized = []
        for paragraph in splitted_body:
            if 'content' in paragraph and paragraph['content']:
                # Tokenize the splitted sentences and
                # join potential sentence splits detected by pattern
                tokenized_sentences = [' '.join(tokenize(s))
                                       for s in paragraph['content']]
                tokenized.append({'content': tokenized_sentences, 'type': paragraph['type']})

        return self.add_enrichment(data, self.name, tokenized)