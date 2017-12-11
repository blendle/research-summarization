import os
os.chdir("/home/Lucas/blendle/blendle-etl")
# Import Torch here to avoid static TLS issues while debugging in PyCharm
import torch

import data_extractors.duc02processor as ducproc
import data_extractors.tac08processor as tacproc
import data_extractors.opinosisprocessor as opiproc
from rouge import evaluate
import json
import numpy as np
import pprint as pp

from enrichers.pipeline import Pipeline
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')


def save_summaries(articles, dataset_name):
    for summ_method in articles[0]['enrichments']['summarizer'].keys():
        indices_summaries = [a['enrichments']['summarizer'][summ_method] for a in articles]
        article_lengths = [len(a['enrichments']['stemmer']) for a in articles]
        indices = list(zip(article_lengths,
                           [[x[0] for x in summary] for summary in
                            indices_summaries]))
        summaries = [[x[1] for x in summary] for summary in
                     indices_summaries]

        indices_summaries = {'indices': indices, 'summaries': summaries}

        with open('/home/Lucas/Downloads/summ_results/{}/{}.json'.format(dataset_name, summ_method),
                  'w') as fh:
            json.dump(indices_summaries, fh)

def save_ref_summaries(articles, dataset_name):
        reference_summaries = [article['summaries'] for article in articles]
        with open('/home/Lucas/Downloads/summ_results/{}/references.json'.format(dataset_name),
                  'w') as fh:
            json.dump(reference_summaries, fh)

def eval_summaries(articles, dataset_name):
    print("Evaluating result with ROUGE...")

    evals = {}
    # evaluate() expects system summaries as a list of sentences, where each sentence is a string.
    for key in articles[0]['enrichments']['summarizer'].keys():
        evals[key] = []
        for article in articles:
            system_summary = [s for i, s in article['enrichments']['summarizer'][key]]
            evals[key].append(evaluate(system_summary=system_summary,
                                       reference_summaries=article['summaries'],
                                       stemming=False,
                                       stopwords=False,
                                       ngram=2))

    for eval_name in evals.keys():
        with open('/home/Lucas/Downloads/rouge_json_files/{d}/{e}.json'.format(d=dataset_name,
                                                                               e=eval_name),
                  'w') as fh:
            json.dump(evals[eval_name], fh)

    measures = ('ROUGE-1-F', 'ROUGE-1-R', 'ROUGE-2-F', 'ROUGE-2-R')

    rouge_results = {
    key: {measure: "{0:.2f}".format(np.mean([x[measure] for x in evals[key]]) * 100)
          for measure in measures}
    for key in evals.keys()}

    pp.pprint(rouge_results)

def eval_ref_test(articles, dataset_name):
    print("Evaluating result with ROUGE...")

    counter = 0
    measures = ('ROUGE-1-F', 'ROUGE-1-R', 'ROUGE-2-F', 'ROUGE-2-R')
    results = []
    # evaluate() expects system summaries as a list of sentences, where each sentence is a string.

    for article in articles:
        if len(article['summaries']) < 2:
            print(article['doc_id'])
            counter += 1
            continue
        results.append(evaluate(system_summary=[article['summaries'][0]],
                                reference_summaries=[article['summaries'][1]],
                                stemming=False,
                                stopwords=False,
                                ngram=2))

    rouge_results = {measure: "{:.2f} ({:.2f})".format(np.mean([x[measure] for x in results]) * 100,
                                                         np.std([x[measure] for x in results]) * 100)
                     for measure in measures}

    pp.pprint(rouge_results)


# FOR DUC 2002 DATASET
# dataset_name = 'duc'
# data_dirname = "/home/Lucas/blendle/data/duc/2002/docs"
# sum_dirname = "/home/Lucas/blendle/data/duc/2002/extracts_abstracts"
# articles = list(ducproc.data_generator(data_dirname))
# ducproc.add_summaries(articles, sum_dirname)

# FOR TAC 2008 DATASET
# dataset_name = 'tac'
# data_dirname = '/home/Lucas/blendle/data/tac/2008/docs'
# sum_dirname = '/home/Lucas/blendle/data/tac/2008/summaries'
# articles = list(tacproc.data_generator(data_dirname))
# # Articles are now actually clusters:
# articles = tacproc.merge_clusters(articles)
# tacproc.add_summaries(articles, sum_dirname)

# FOR OPINOSIS DATASET
dataset_name = 'opinosis'
data_dirname = '/home/Lucas/blendle/data/opinosis/topics'
sum_dirname = '/home/Lucas/blendle/data/opinosis/summaries'
articles = list(opiproc.data_generator(data_dirname))
opiproc.add_summaries(articles, sum_dirname)

pipeline = Pipeline(('cleaner',
                     'sentence_splitter',
                     'tokenizer',
                     'stemmer',
                     'postagger',
                     'summarizer'
                     ),
                    config={},
                    adapter='item',
                    dataset_name=dataset_name
                    )
# To retrieve article lengths, make stemmer persistent
pipeline.enrichers['stemmer'].persistent = True

for index, article in enumerate(articles):
    pipeline(article)
    print("Finished article #", index+1)


eval_ref_test(articles=articles, dataset_name=dataset_name)
save_ref_summaries(articles=articles, dataset_name=dataset_name)
save_summaries(articles=articles, dataset_name=dataset_name)
eval_summaries(articles=articles, dataset_name=dataset_name)
