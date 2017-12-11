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


def save_summaries(articles, dataset_name, save_path):
    for summ_method in articles[0]['enrichments']['summarizer'].keys():
        indices_summaries = [a['enrichments']['summarizer'][summ_method] for a in articles]
        article_lengths = [len(a['enrichments']['stemmer']) for a in articles]
        indices = list(zip(article_lengths,
                           [[x[0] for x in summary] for summary in
                            indices_summaries]))
        summaries = [[x[1] for x in summary] for summary in
                     indices_summaries]

        indices_summaries = {'indices': indices, 'summaries': summaries}

        with open(os.path.join(save_path, '{}/{}.json'.format(dataset_name, summ_method)),
                  'w') as fh:
            json.dump(indices_summaries, fh)

def eval_summaries(articles, dataset_name, save_path):
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
        with open(os.path.join(save_path, '{d}/{e}.json'.format(d=dataset_name, e=eval_name)),
                  'w') as fh:
            json.dump(evals[eval_name], fh)

    measures = ('ROUGE-1-F', 'ROUGE-1-R', 'ROUGE-2-F', 'ROUGE-2-R')

    rouge_results = {
    key: {measure: "{0:.2f}".format(np.mean([x[measure] for x in evals[key]]) * 100)
          for measure in measures}
    for key in evals.keys()}

    pp.pprint(rouge_results)

# General settings:
# Path for saving summaries
summary_save_path = # TODO: set path
eval_save_path = # TODO: set path

# FOR DUC 2002 DATASET
# dataset_name = 'duc'
# data_dirname = # TODO: set path
# sum_dirname = # TODO: set path
# articles = list(ducproc.data_generator(data_dirname))
# ducproc.add_summaries(articles, sum_dirname)

# FOR TAC 2008 DATASET
# dataset_name = 'tac'
# data_dirname = # TODO: set path
# sum_dirname = # TODO: set path
# articles = list(tacproc.data_generator(data_dirname))
# # Articles are now actually clusters:
# articles = tacproc.merge_clusters(articles)
# tacproc.add_summaries(articles, sum_dirname)

# FOR OPINOSIS DATASET
dataset_name = 'opinosis'
data_dirname = 'datasets/Opinosis/topics'
sum_dirname = 'datasets/Opinosis/summaries'
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

save_summaries(articles=articles, dataset_name=dataset_name, save_path=summary_save_path)
eval_summaries(articles=articles, dataset_name=dataset_name, save_path=eval_save_path)
