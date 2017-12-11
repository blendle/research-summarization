import os
from bs4 import BeautifulSoup
import xmltodict
from collections import OrderedDict

def data_generator(dirname):
    """Transforms all summaries in directory into data objects for pipeline.

    Simply iterates over the directory and calls file_to_data.
    """
    for root, dirs, files in os.walk(dirname):
        for name in files:
            # exclude hidden files
            if not name[0] == ".":
                filename = os.path.join(root, name)
                yield file_to_data(filename)


def file_to_data(filename):
    """Transforms a single summary file into a data object for pipeline."""
    # open file
    with open(filename, 'r') as fh:
        # transform to real xml
        soup = BeautifulSoup(fh.read(), features='xml')

    # remove P-tags from xml (to make LA Times readable)
    cleaned_soup = str(soup).replace('<P>', '')
    cleaned_soup = cleaned_soup.replace('</P>', '')

    # parse xml and put into ordered dict, remove outside DOC-tag
    article = xmltodict.parse(cleaned_soup)['DOC']

    # retrieve document id
    doc_id = article['DOCNO']['s']['#text']

    # retrieve all tagged sentences from the document
    sentences = []

    text = list(find(['TEXT'], article))
    if type(text[0]) is list:
        text = [item for sublist in text for item in sublist]
    for x in text:
        for y in find(['s'], x):
            if type(y) is list:
                for e in y:
                    sentences.append(e.get('#text', ''))
            else:
                sentences.append(y.get('#text', ''))

    # put sentences in list of dicts
    body = [{'type': 'p', 'content': s} for s in sentences]

    # generate yield baseline (first 100 tokens)
    lead_baseline = []
    max_length = 100
    summary_length = 0
    #     lead_baseline = list(lead_sentence_feeder(article))[:3]
    for sentence in lead_sentence_feeder(article):
        sentence_length = len(sentence.split())
        summary_length += sentence_length
        if summary_length > max_length + 15:
            break
        lead_baseline.append(sentence)
        if summary_length >= max_length:
            break

    # Create dictionary of doc_id and body
    data = {'doc_id': doc_id, 'body': body, 'summaries': [], 'lead': lead_baseline}

    return data


def find(keys, dictionary):
    if not dictionary is None:
        for k, v in dictionary.items():
            if k in keys:
                yield v
            elif type(v) is OrderedDict:
                for result in find(keys, v):
                    yield result
            elif type(v) is list:
                for d in v:
                    for result in find(keys, d):
                        yield result


def lead_sentence_feeder(article):
    text = list(find(['TEXT', 'LEADPARA', 'LP'], article))
    if type(text[0]) is list:
        text = [item for sublist in text for item in sublist]

    # yield whitespace-delimited tokens from all sentences
    for x in text:
        for y in find(['s'], x):
            if type(y) is list:
                for element in y:
                    yield element.get('#text', '')
            else:
                yield y.get('#text', '')

def add_summaries(articles, dirname):
    """Adds summaries from summary directory to the correct article object."""
    all_summaries = []

    for root, dirs, files in os.walk(dirname, topdown=True):
        for name in files:
            # only include "perdocs" files
            if name == "perdocs":
                filename = os.path.join(root, name)
                with open(filename, 'r') as fh:
                    # split perdocs-file into separate summary tag strings
                    summaries = [x + "</SUM>" for x in fh.read().split('</SUM>')
                                 if not (x == "</SUM>" or x.isspace())]
                    # transform summaries into actual xml
                    summaries = [str(BeautifulSoup(summary, 'xml')) for summary in summaries]
                    # parse every summary into dict
                    summaries = [xmltodict.parse(summary)['SUM'] for summary in summaries]
                    # add the parsed summaries the full list of summaries
                    all_summaries += summaries

    # add summaries to corresponding article objects
    for summary in all_summaries:
        for article in articles:
            article_summaries = article['summaries']
            # ..but only the first two, as the rest is duplicate from other clusters
            if article['doc_id'] == summary['@DOCREF'].strip() and not len(article_summaries) > 1:
                article['summaries'].append(summary['#text'])

    return articles