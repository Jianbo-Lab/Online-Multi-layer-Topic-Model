This software runs the two layer topic model presented in the paper "Online learning for multi-layer conjugate exponential families" by Jianbo Chen (2016) in a truly online setting. English Wikipedia articles are randomly downloaded and input into the algorithm. At each step, the perplexity of the randomly downloaded but unanalyzed articles as an indicator of the predictive power of our model.

dictnostops.txt contains all the vocabularies selected to analyze. This is downloaded from David Blei's lab: https://github.com/blei-lab/onlineldavb 



wikirandom.py randomly downloads wikipedia articles from the website. This requires the installation of wikipedia package into python, which can be downloaded at https://pypi.python.org/pypi/wikipedia/



online_wiki_functions.py has all the functions needed and represent our model as a class. Some of the functions in this python script is revised from David Blei's open-source software for online LDA, available at https://github.com/blei-lab/

onlineldavb

online_wiki.py is the main program. 
An example to use it is: "python online_wiki.py 10". 

The argument 10 means that we analyze 10 mini-batches of articles, which are 10*32 articles randomly downloaded from wikipedia.

