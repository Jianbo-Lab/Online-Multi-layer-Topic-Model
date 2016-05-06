import cPickle, string, numpy, getopt, sys, random, time, re, pprint

import numpy as np

import twolayerfunctions
    
def f(K1,K2,tau,kappa): 
    rho = (tau,kappa)
    K = (K1,K2)
    """
    This function fits New York Times articles to the two hidden layer topic model in an online style.
    """
    
    # The number of documents to analyze in each iteration
    batchsize = 32
    # The estimated total number of documents  
    D = 3e5
     
    vocab = file('dataset/vocab.nytimes.txt').readlines()
    # The total number of iterations
    M = 4650
    vocab = file('dataset/vocab.nytimes.txt').readlines()
 
    eta0 = 1 / np.float(K1)
    eta1 = 1 / np.float(K2)
    eta2 = 1 / np.float(K2)
     
         
    return twolayerfunctions.fit_model(K,rho,M,vocab,D,eta0, eta1, eta2) 

    # Initialize the algorithm.
    # Create a grid of K1,K2,rho. 
    # Create a list of tuples to store parameters.
     