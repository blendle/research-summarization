import data_extractors.duc02processor as ducproc
import data_extractors.opinosisprocessor as opiproc
import data_extractors.tac08processor as tacproc
from rouge import evaluate
import numpy as np
import warnings
# Ignore the All-NaN warning, as it's handled correctly
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
from enrichers.pipeline import Pipeline

pipeline = Pipeline(('cleaner',
                     'sentence_splitter',
                     'tokenizer',
                     'stemmer',
                     'postagger',
                     'summarizer'))

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
data_dirname = '/Users/Lucas/blendle/benchmark_sets/opinosis/topics'
sum_dirname = '/Users/Lucas/blendle/benchmark_sets/opinosis/summaries'
articles = list(opiproc.data_generator(data_dirname))[0:2]
opiproc.add_summaries(articles, sum_dirname)


for index, article in enumerate(articles):
    pipeline(article)
    print("Finished article #", index+1)

print("Evaluating result with ROUGE...")

evals = {}
# evaluate() expects system summaries as a list of sentences, where each sentence is a string.
for key in articles[0]['enrichments']['summarizer'].keys():
    evals[key]= [evaluate(system_summary=article['enrichments']['summarizer'][key],
                          reference_summaries=article['summaries'],
                          stemming=False,
                          stopwords=False,
                          ngram=2)
                 for article in articles]


measures = ('ROUGE-1-F', 'ROUGE-1-R', 'ROUGE-2-F', 'ROUGE-2-R')
rouge_results = {key: {measure: "{0:.2f}".format(np.mean([x[measure] for x in evals[key]]) * 100)
                       for measure in measures}
                 for key in evals.keys()}

print(rouge_results)
