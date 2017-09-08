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
            # Exclude hidden files and only cover non-update files
            if not name[0] == "." and root[-1] == 'A':
                filename = os.path.join(root, name)
                # Get cluster id (ending with '-A') from root name
                cluster_id = root[-8:-3] + '-A'
                yield file_to_data(filename, cluster_id)


def file_to_data(filename, cluster_id):
    """Transforms a single summary file into a data object for pipeline."""
    # open file
    with open(filename, 'r') as fh:
        # transform to real xml
        soup = BeautifulSoup(fh.read(), features='xml')

    # remove P-tags from xml (to make LA Times readable)
    # cleaned_soup = str(soup)
    cleaned_soup = str(soup).replace('<P>', '')
    cleaned_soup = cleaned_soup.replace('</P>', '')

    # parse xml and put into ordered dict, remove outside DOC-tag
    article = xmltodict.parse(cleaned_soup)['DOC']

    # retrieve document
    doc_id = article['@id']

    # retrieve all relevant text from the document
    text = [article.get('HEADLINE', ''), article.get('TEXT', '')]

    # put sentences in list of dicts
    body = [{'type': 'p', 'content': p} for p in text]

    # Create dictionary of doc_id and body
    data = {'doc_id': doc_id, 'cluster_id': cluster_id, 'body': body, 'summaries': []}

    return data

def merge_clusters(articles):
    """Merges articles of every cluster into one article object."""
    clusters = []
    cluster_ids = set([article['cluster_id'] for article in articles])
    for id in cluster_ids:
        body = []
        for article in articles:
            if article['cluster_id'] == id:
                body += article['body']
        data = {'cluster_id': id, 'body': body, 'summaries': []}
        clusters.append(data)

    return clusters


def add_summaries(articles, dirname):
    """Adds summaries from summary directory to the correct article object."""
    all_summaries = {}
    for root, dirs, files in os.walk(dirname, topdown=True):
        for name in files:
            filename = os.path.join(root, name)
            # Cluster id is in the first 7 characters of the filename
            cluster_id = name[:7]
            with open(filename, 'r') as fh:
                summary = fh.read()
            # Add summary to right list (or initiate cluster if not existing yet)
            if cluster_id in all_summaries:
                all_summaries[cluster_id] += [summary]
            else:
                all_summaries[cluster_id] = [summary]

    # Add list of summaries (4 per cluster) to corresponding article objects
    for article in articles:
        article['summaries'] = all_summaries[article['cluster_id']]
