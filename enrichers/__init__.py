from .base import Enricher  # flake8: NOQA
from .postagger import PoSTagger  # flake8: NOQA
from .sentence_splitter import SentenceSplitter  # flake8: NOQA
from .tokenizer import Tokenizer  # flake8: NOQA
from .cleaner import Cleaner  # flake8: NOQA
from .stemmer import Stemmer  # flake8: NOQA
from .summarizer import Summarizer  # flake8: NOQA


def get_enricher(enricher_name):
    for enricher in inheritors(Enricher):
        if enricher.name == enricher_name:
            return enricher


def enrichers():
    return [e.name for e in inheritors(Enricher)]


def inheritors(cls):
    subclasses = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses