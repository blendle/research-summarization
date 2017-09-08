import os
import codecs


def data_generator(dirname):
    """Transforms all summaries in directory into data objects for pipeline.

    Simply iterates over the directory and calls file_to_data.
    """
    for root, dirs, files in os.walk(dirname):
        for name in files:
            # Exclude hidden files
            if not name[0] == ".":
                topic_id = name[0:-9]
                filename = os.path.join(root, name)
                yield file_to_data(filename, topic_id)


def file_to_data(filename, topic_id):
    """Transforms a single summary file into a data object for pipeline."""
    # open file
    review_sentences = []
    with codecs.open(filename, "r", encoding='utf-8', errors='ignore') as fh:
        review_sentences = fh.read().splitlines()

    # put sentences in list of dicts
    body = [{'type': 'p', 'content': p} for p in review_sentences]

    # Create dictionary of doc_id and body
    data = {'topic_id': topic_id, 'body': body, 'summaries': []}

    return data


def add_summaries(articles, dirname):
    """Adds summaries from summary directory to the correct article object."""
    all_summaries = {}
    for root, dirs, files in os.walk(dirname, topdown=True):
        for name in files:
            filename = os.path.join(root, name)
            # Cluster id is in the first 7 characters of the filename
            with open(filename, 'r') as fh:
                summary = fh.read()
            # Add summary to right list (or initiate topic if not existing yet)
            topic_id = name[:-7]
            if topic_id in all_summaries:
                all_summaries[topic_id] += [summary]
            else:
                all_summaries[topic_id] = [summary]

    # Add list of summaries to corresponding article objects
    for article in articles:
        article['summaries'] = all_summaries[article['topic_id']]