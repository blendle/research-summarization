# Blogpost code

This code accompanies a blog post from Blendle Research.
It can be used to exactly reproduce all experimental results.

# How to run

Set the summarization function(s) in summarizer.py, and then run main.py to output results.

Some files are not included:

* The google word2vec model is not included in this repo, but can be downloaded [here](https://code.google.com/archive/p/word2vec/); it is expected to be in models/word2vec/google/
* The datasets are not included as access can only be granted by NIST.

## Requirements

* python >= 3.5
* [pythonrouge](https://github.com/tagucci/pythonrouge)
* regex
* scipy
* networkx
* gensim
* xmltodict
* numpy
* pattern
* nltk
* beautifulsoup4
* scikit_learn
* typing