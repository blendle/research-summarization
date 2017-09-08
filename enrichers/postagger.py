from sklearn.externals import joblib
from enrichers.base import Enricher


class PoSTagger(Enricher):
    name = 'postagger'
    persistent = False
    requires = ('tokenizer',)
    model_file = 'models/crf_pos.pkl'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = joblib.load(self.model_file)

    def __call__(self, data):
        tokenized_body = self.get_enrichment(data, 'tokenizer')

        tagged_sentences = []
        for sentence in self.get_tokenized_sentences(tokenized_body):
            tokens = sentence.split(' ')
            features = [self.word2features(tokens, i) for i in range(len(tokens))]
            postags = self.model.predict([features])[0]
            tagged_sentences.append(list(zip(tokens, postags)))

        return self.add_enrichment(data, self.name, tagged_sentences)

    @classmethod
    def word2features(cls, sent, i):
        word = sent[i]
        features = {
            'bias': 1.0,
            'word.identity': word.lower(),
            'word.suf3': word[-3:],
            'word.suf2': word[-2:],
            'word.isupper': word.isupper(),
            'word.startcap': word.istitle(),
            'word.isdigit': word.isdigit()
        }
        if i > 0:
            prev_word = sent[i - 1]
            features.update({
                'prev_word.identity': prev_word.lower(),
                'prev_word.suf3': prev_word[-3:],
                'prev_word.suf2': prev_word[-2:],
                'prev_word.startcap': prev_word.istitle(),
                'prev_word.isupper': prev_word.isupper(),
                'prev_word.isdigit': prev_word.isdigit()
            })
        else:
            features['SentStart'] = True

        if i < len(sent) - 1:
            next_word = sent[i + 1]
            features.update({
                'next_word.identity': next_word.lower(),
                'next_word.startcap': next_word.istitle(),
                'next_word.isupper': next_word.isupper()
            })
        else:
            features['SentEnd'] = True

        return features
