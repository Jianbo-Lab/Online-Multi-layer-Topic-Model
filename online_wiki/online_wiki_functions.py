import sys, re, time, string
import numpy as np
from scipy.special import gammaln, psi
 
import numpy as np
from scipy.misc import logsumexp
np.random.seed(100000001)
meanchangethresh = 0.001
def log_beta(v):
    """
    For a vector v, this function computes the log of the multivariate beta function
    evaluated at v.
    """
    return(sum(gammaln(v))-gammaln(sum(v)))
def dirichlet_expectation(alpha):
    if (len(alpha.shape) == 1):
        return(psi(alpha) - psi(np.sum(alpha)))
    return(psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis])

def parse_doc_list(docs, vocab):
    """
    Parse a document into a list of word ids and a list of counts,
    or parse a set of documents into two lists of lists of word ids
    and counts. 
   Wordids are what vocabulary tokens present in
    each document. wordids[i][j] gives the jth unique token present in
    document i. 

    Wordcts, says how many times each vocabulary token is
    present. wordcts[i][j] is the number of times that the token given
    by wordids[i][j] appears in document i.
    """
    if (type(docs).__name__ == 'str'):
        temp = list()
        temp.append(docs)
        docs = temp

    D = len(docs)
    
    wordids = list()
    wordcts = list()
    for d in range(0, D):
        docs[d] = docs[d].lower()
        docs[d] = re.sub(r'-', ' ', docs[d])
        docs[d] = re.sub(r'[^a-z ]', '', docs[d])
        docs[d] = re.sub(r' +', ' ', docs[d])
        words = string.split(docs[d])
        ddict = dict()
        for word in words:
            if (word in vocab):
                wordtoken = vocab[word]
                if (not wordtoken in ddict):
                    ddict[wordtoken] = 0
                ddict[wordtoken] += 1
        wordids.append(ddict.keys())
        wordcts.append(ddict.values())

    return((wordids, wordcts))

class Online_two_hidden_layers:
    """
    Implements online VB for LDA as described in (Hoffman et al. 2010).
    """

    def __init__(self, vocab, K1, K2, D, eta0,eta1,eta2, tau0, kappa):
        """
        Arguments:
        K1,K2: The size of the first and second hidden layers.
        vocab: A set of words to recognize. Ignore the words not in this set.
        D: Total number of documents in the population, or an estimate 
        of the number in the truely online setting.
        eta0,eta1,eta2: Hyperparameters for the weight matrices in the first, second and third layers respectively.  
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """
        self._vocab = dict()
        for word in vocab:
            word = word.lower()
            word = re.sub(r'[^a-z]', '', word)
            self._vocab[word] = len(self._vocab)

        self._K1 = K1
        self._K2 = K2
        self._W = len(self._vocab)
        self._D = D
        self._eta0 = eta0
        self._eta1 = eta1
        self._eta2 = eta2
        self._const = K1 * (self._W * gammaln(eta0) - gammaln(self._W * eta0)) + K2 * (K1 *gammaln(eta1) - gammaln(K1 * eta1))
        self._tau0 = tau0 + 1
        self._kappa = kappa
        self._updatect = 0
        # Initialize the variational distribution q(W|lambda).
        self._lambda0 = np.random.gamma(100., 1./100., (self._K1, self._W))
        self._ElogW0 = dirichlet_expectation(self._lambda0)
        self._lambda1 = np.random.gamma(100., 1./100., (self._K2, self._K1))        
        self._ElogW1 = dirichlet_expectation(self._lambda1)
        
    def do_e_step(self, wordids, wordcts):
        """
        Arguments: wordids and wordcts are two lists. Each element in 
        lists are from a separate incoming documents. 
        This function updates the local variational parameters phi and 
        gamma, with fixed lambda.
        """
        batchD = len(wordids)
        # Initialize the variational distribution q(theta|gamma) for
        # the mini-batch.
        #gamma = np.random.gamma(100., 1./100., (batchD, self._K2))
        #ElogW2 = dirichlet_expectation(gamma)
        #expElogW2 = np.exp(ElogW2)
        
        ## Initialize the variational distribution q(z|phi) for the mini-batch.
        
        meanchange = 0
        def update_gammaphi(d):
            ids = wordids[d]
            cts = wordcts[d]
            Wd = len(ids)
            ElogW0d = self._ElogW0[:,ids]
            phi1 = np.tile(1/self._K1,(len(ids),self._K1))
            phi2 = np.tile(1/self._K2,(len(ids),self._K2))
            gamma = np.random.gamma(100., 1./100.,self._K2)
            ElogW2 = dirichlet_expectation(gamma)
            for it in range(100):
                lastgamma = gamma
                # Wd * K1 matrix
                unnormalized_log_phi1 = np.dot(phi2,self._ElogW1) + np.transpose(ElogW0d)
                # Wd vector
                phi1normalizer = map(logsumexp,unnormalized_log_phi1)
                # Wd * K1 matrix
                phi1 = np.exp(unnormalized_log_phi1 - np.transpose(np.tile(phi1normalizer,(self._K1,1))))
                # Wd * K2 matrix
                unnormalized_log_phi2 = np.dot(phi1,np.transpose(self._ElogW1)) + np.tile(ElogW2,(Wd,1)) 
                # Wd vector
                phi2normalizer = map(logsumexp,unnormalized_log_phi2)
                # Wd * K2 matrix
                phi2 = np.exp(unnormalized_log_phi2 - np.transpose(np.tile(phi2normalizer,(self._K2,1))))
                # K2 vector
                gamma = np.tile(self._eta2,self._K2) + np.dot(np.transpose(phi2),cts)
                meanchange = np.mean(abs(gamma - lastgamma))
                if (meanchange < meanchangethresh):
                    break
            # Compute phi1 times cts 
            # Psi1: K1 * Wd
            # Psi2: K2 * K1
            Psi1 = np.multiply(np.transpose(phi1),cts)
            Psi2 = np.transpose(np.dot(Psi1,phi2))
            return((ids,cts,Psi1,Psi2,gamma,phi1,phi2))
        
        return(map(update_gammaphi,range(0,batchD)))
    
    def do_e_step_docs(self, docs):
        # This is to handle the case where someone just hands us a single
        # document, not in a list.
        if (type(docs).__name__ == 'str'):
            temp = list()
            temp.append(docs)
            docs = temp

        (wordids, wordcts) = parse_doc_list(docs, self._vocab)

        return self.do_e_step(wordids, wordcts)
    
    def update_lambda_docs(self, docs):
        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._rhot = rhot
        gamma_phi = self.do_e_step_docs(docs)
        bound = self.perplexity_approx(gamma_phi)
        
        for d in range(len(gamma_phi)):
            # self._lambda0[:,gamma_phi[d][0]]: K1 * Wd.
            self._lambda0[:,gamma_phi[d][0]] = self._lambda0[:,gamma_phi[d][0]] * \
            (1-rhot) + rhot * (np.tile(self._eta0,(self._K1,len(gamma_phi[d][0]))) + \
                               self._D * gamma_phi[d][2] / np.float(len(docs)))
            # self._lambda1: K2 * K1
            self._lambda1 = self._lambda1 * (1-rhot) + rhot * (np.tile(self._eta1,(self._K2,self._K1)) \
                       + self._D * gamma_phi[d][3] / np.float(len(docs)))
        
        self._ElogW0 = dirichlet_expectation(self._lambda0)       
        self._ElogW1 = dirichlet_expectation(self._lambda1)
        
        return(bound)
                                 
    def perplexity_approx(self,gamma_phi):
        score0 = sum(log_beta(self._lambda0)) + sum(log_beta(self._lambda1))  
        score0 += np.sum(np.multiply(np.tile(self._eta0,(self._K1,self._W)) - self._lambda0,self._ElogW0))
        score0 += np.sum(np.multiply(np.tile(self._eta1,(self._K2,self._K1)) - self._lambda1,self._ElogW1))
        ratio = self._D / np.float(len(gamma_phi))
        # This function approximates the perplexity before updating the 
        # global parameters for a particular document.
        def perplexityd(d):
            (ids,cts,Psi1,Psi2,gamma,phi1,phi2) = gamma_phi[d]
            score = np.sum(np.multiply(self._ElogW0[:,ids],Psi1))
            score += np.sum(np.multiply(Psi2,self._ElogW1))
            score += np.dot(dirichlet_expectation(gamma),np.dot(np.transpose(phi2),cts) + \
                             np.tile(self._eta2,(self._K2)) - gamma)
            score += (log_beta(gamma) - log_beta(np.tile(self._eta2,(self._K2))))
            score = score - np.sum(np.dot(np.transpose(np.multiply(phi1,np.log(phi1))),cts))
            score = score - np.sum(np.dot(np.transpose(np.multiply(phi2,np.log(phi2))),cts))
            return(score)

        totalscore = ratio * sum(map(perplexityd,range(len(gamma_phi)))) - self._const + score0
        numtotalwords = sum([sum(gamma_phi[d][1]) for d in range(len(gamma_phi))])                           
        return(np.exp(- totalscore / np.float(numtotalwords * ratio)))                           
                                 
    def approx_bound(self,obs_wordids,obs_wordcts,unobs_wordids,unobs_wordcts):
        """ 
        This function form the variational model by using the lambda learned from 
        the training dataset and using the local variational parameters learned given 
        the ten percent of the words in each document.
        
        The arguments: 
        obs_wordids, obs_wordcts are two lists. obs_wordids[d][j] 
        gives the jth unique word in the selected ten percent of dth document.
        obs_wordcts[d][j] gives its corresponding count.
        
        unobs_wordids,unobs_wordcts are two lists. unobs_wordids[d][j] 
        gives the jth unique word in the rest 90 percent of dth document.
        unobs_wordcts[d][j] gives its corresponding count.
        
        The output is the held-out perplexity given by the approximation 
        using the variational bound.
        """
        # Computing the local parameters.
        local_par = compute_local_par_for_held_out(obs_wordids, obs_wordcts)
        # Calculating the perplexity.
        # Note: We should write a py document to form a test data set.