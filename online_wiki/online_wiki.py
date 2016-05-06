import cPickle, string, getopt, sys, random, time, re, pprint
import numpy as np
import online_wiki_functions 
import wikirandom

def main():
    """
    This function fits Wikipedia articles to the two-hidden-layer model in an online style.
    """

    # The number of documents to analyze in each iteration
    batchsize = 32
    # The estimated total number of documents  
    D = 5.13e6
    # The number of topics
    K1 = 30
    K2 = 3
    eta0 = 1 / np.float(K1)
    eta1 = 1 / np.float(K2)
    eta2 = 1 / np.float(K2) 

    # The total number of iterations
    if (len(sys.argv) < 2):
        M = 100
    else:
        M = int(sys.argv[1])

 
    vocab = file('./dictnostops.txt').readlines()
    W = len(vocab)

    # Initialize the algorithm.
    model = online_wiki_functions.Online_two_hidden_layers(vocab, K1, K2, D, eta0, eta1, eta2, 256, 0.6) 
    for iteration in range(0, M):
        # Download wikipedia articles randomly.
        (docset, articlenames) = \
            wikirandom.get_random_wikipedia_articles(batchsize)
        # Compute the held-out perplexity and fit them to the deep LDA model.
        bound = model.update_lambda_docs(docset) 
        print '%d: held-out perplexity estimate = %f' % \
            (iteration, bound) 

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        #if (iteration % 10 == 0):
        #    numpy.savetxt('lambda-%d.dat' % iteration, olda._lambda)
        #    numpy.savetxt('gamma-%d.dat' % iteration, gamma)
    return bound

if __name__ == '__main__':
    main()
