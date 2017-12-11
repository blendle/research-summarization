# This file contains the code to extract bigrams and train the SVD models
# used as sentence representation by Yogatama et al. (2015).
# This code has already been run, and the resulting models are saved models/.

import data_extractors.duc02processor as ducproc
import data_extractors.opinosisprocessor as opiproc
import data_extractors.tac08processor as tacproc
import nltk
from enrichers.pipeline import Pipeline
from collections import Counter
import string
import json
import pickle
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib

stopword_path = 'models/stopwords.json'
with open(stopword_path) as fh:
    stopwords = json.load(fh)

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

# FOR OPINOSIS DATASET
dataset_name = 'opinosis'
data_dirname = 'datasets/Opinosis/topics'
sum_dirname = 'datasets/Opinosis/summaries'
articles = list(opiproc.data_generator(data_dirname))
opiproc.add_summaries(articles, sum_dirname)

pipeline = Pipeline(('cleaner',
                     'sentence_splitter',
                     'tokenizer',
                     'stemmer'))
# Keep the results of stemmer for later use
pipeline.enrichers['stemmer'].persistent = True

# Clean, tokenize & stem all articles
for article in articles:
    pipeline(article)

bigrams_df_article = []
bigrams_df_sentence = []
for article in articles:
    article_bigrams = []
    for sentence in article['enrichments']['stemmer']:
        split_sentence = [stem for word, stem in sentence
                          if word not in string.punctuation
                          ]
        sentence_bigrams = nltk.bigrams(split_sentence)
        # Filter out every bigram containing two stopwords
        sentence_bigrams = [(word1, word2) for word1, word2 in sentence_bigrams
                            if not (word1 in stopwords and word2 in stopwords)]

        # Deduplicate on sentence level
        sentence_bigrams = list(set(sentence_bigrams))

        # Add unique sentence bigrams to list of bigrams in article
        article_bigrams += sentence_bigrams

    # Update bigram counts on sentence level (before deduplicating on article level)
    bigrams_df_sentence += article_bigrams

    # Deduplicate on article level
    article_bigrams = list(set(article_bigrams))

    # Update bigram counts on article level
    bigrams_df_article += article_bigrams

# Construct document frequency dicts on sentence and article level
df_article = Counter(bigrams_df_article)
df_sentence = Counter(bigrams_df_sentence)

# Filter out bigrams that occur in less than 3 sentences
# As done by Gillick et al. (2008) (as Yogatama et al. are ambiguous, and reference this paper):
final_bigrams = list({k:v for k, v in df_article.items() if not v < 3}.items())

print("Total number of unique bigrams: ", len(set(df_article)))
print("Most common bigrams: ", df_article.most_common(10))
print("Number of trimmed unique bigrams: ", len(final_bigrams))

sentence_repr_list = []
for article in articles:
    for sentence in article['enrichments']['stemmer']:
        split_sentence = [stem for word, stem in sentence
                          if word not in string.punctuation
                          ]
        bigrams = list(nltk.bigrams(split_sentence))
        sentence_repr = [v if k in bigrams else 0 for k, v in final_bigrams]
        sentence_repr_list.append(sentence_repr)

print("Sentence array shape before trimming: ", np.array(sentence_repr_list).shape)
# Remove all sentences that do not contain bigrams from the final bigram dict
sentence_repr_list = [s for s in sentence_repr_list if np.sum(s) != 0]
sentence_array = np.array(sentence_repr_list)
print("Sentence array shape after trimming: ", sentence_array.shape)

# Perform SVD on the full sentence matrix
svd = TruncatedSVD(n_components=600, n_iter=7, random_state=42)
svd.fit(sentence_array)
# Save model & bigram dict
joblib.dump(svd, 'models/bigram_svd_{}.model'.format(dataset_name), compress=True)
with open('models/bigrams_{}.pkl'.format(dataset_name), 'wb') as fp:
    pickle.dump(final_bigrams, fp)

print("Saved svd model & bigram dict.")
