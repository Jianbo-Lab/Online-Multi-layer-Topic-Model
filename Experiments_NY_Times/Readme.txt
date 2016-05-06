The Python code implements the stochastic natural gradient method for the two-hidden-layer topic model presented in the paper "Online learning for multi-layer conjugate exponential families" by Jianbo Chen (2016).The model is fit to a New York Times data set with 300,000 documents in an online setting. 

Files provided:

transform_data.py transforms the original dataset docword.nytimes.txt.gz into required form. Concretely, it maps the whole dictionary to our selected vocab. It also splits the large txt file into small pieces.And it constructs the test dataset by combining the last few files after conversion. The original dataset can be downloaded from https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/.

dataset is a folder containing all split txt files storing word counts and a vocabulary.
 
vocab.nytimes.txt contains all the vocabularies in the data set respectively.

dictnostops.txt contains all the vocabularies selected to analyze. This is downloaded from David Blei's lab: https://github.com/blei-lab/onlineldavb 
 
twolayerfunctions.py contains all the functions and builds the two-hidden-layer model as a class.

nytimes_analysis.py uses the functions in twolayerfunctions.py to fit two-hidden-layer model to the NY-times data set. Perplexity on a held-out set is used as an evaluation metric and output to a txt file.

results for small K2, results for learning rates are the perplexity output by our experiments

figures.py uses these results to plot figures which is analyzed in the Experiments section of the paper.  