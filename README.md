# Blogpost code

This code accompanies a blog post on embedding-based extractive summarization from Blendle Research.
It can be used to exactly reproduce all experimental results.

# How to run

Set the summarization function(s) in summarizer.py, and then run main.py to output results.

Some files are not included:

* The Google word2vec model is not included in this repo, but can be downloaded [here](https://code.google.com/archive/p/word2vec/); it is expected to be in models/word2vec/google/, and is necessary to run main.py out-of-the-box.
* The [DUC-2002](http://www-nlpir.nist.gov/projects/duc/data.html) and [TAC-2008](https://tac.nist.gov/data/index.html) dataset are not included as access can only be granted by NIST (click on the links for more information on obtaining access).
* The [Opinosis dataset](http://kavita-ganesan.com/opinosis-opinion-dataset) is included, and main.py is configured to run on this dataset by default.

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