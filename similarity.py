"""
This code contains classes for calculating word similarity.

The classes take a vocabulary list and export similarity() 
to get word similarity and get_similarity_matrix() to get 
the whole matrix for all words in the stored vocab.
"""

__author__ = "Richard Futrell"
__copyright__ = "Copyright 2012, Richard Futrell"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__maintainer__ = "Richard Futrell"
__email__ = "See the author's website"


from numpy import *
import scipy.spatial.distance as spd
import scipy.sparse as sps
import sklearn.preprocessing as sklpp
from nltk.corpus import wordnet as wn
import multiprocessing
from functools import partial
import copy_reg
import types

def reduce_method(m):
    return (getattr, (m.__self__, m.__func__.__name__))

copy_reg.pickle(types.MethodType, reduce_method)

class DistribSim:
    """ DistribSim: A class for distributional similarity.

    Allows a ContextList to be used to compute distributional
    similarity of a given vocab of words.

    Similarity is computed as cosine similarity of context
    vectors, which may be weighted by a user-specified weight
    function (from the weighting module).
    """
    def __init__(self, ctxList, vocab=None, weight=None):
        self.ctxList = ctxList
        self.weight = weight
        if not self.weight: self.weight = lambda x:x
        self.vocab = vocab
        if not self.vocab: self.vocab = ctxList.get_targets()
        self.matrix = self.get_similarity_matrix(vocab=vocab,weight=self.weight)

    def get_context_matrix(self, vocab=None, ctxList=None):
        """ Return the context matrix for a given vocab.

        Args:
        vocab: An ordered list of words. If a word is not
        in the ContextList, then its context vector is zeros.
        ctxList: Optionally, a ContextList object which 
        exports get_sparse_matrix(). If nothing is specified,
        the class's internal ContextList is used.

        Returns: A CSR matrix with rows as words and cols as
        contexts.
        """

        if not ctxList: ctxList = self.ctxList
        if not vocab: vocab = self.vocab
        return ctxList.get_sparse_matrix(targets=vocab) # sparse row matrix

    def similarity(self, w1, w2):
        return self.matrix[self.vocab.index(w1),self.vocab.index(w2)]
    
    def get_similarity_matrix(self, vocab=None, ctxList = None, weight = None):
        """ Return a similarity matrix.
       
        Returns the similarity matrix for words; this is
        a diagonal matrix where 1 represents maximal
        similarity and 0 represents minimal similarity.
        The main diagonal is all 1, since a word has similarity 
        1 with itself.

        Args:
        vocab: Optionally, a list of vocabulary words which 
        will make up the rows/cols of the similarity matrix.
        ctxList: Optionally, a custom ContextList object.
        weight: Optionally, a weighting function to be applied
        to the counts in the ContextList.

        Returns: A CSR similarity matrix.
        """

        if (not vocab and not ctxList and not weight): 
            return self.matrix
        if not ctxList: ctxList = self.ctxList
        if not vocab: vocab = self.vocab
        if not weight: weight = self.weight

        def remove_negatives(x):
            x = sps.coo_matrix(x)
            toRemove = []
            for i,d in enumerate(x.data):
                if d <= 0:
                    toRemove.append(i)
            x.data = delete(x.data,toRemove)
            x.row = delete(x.row, toRemove)
            x.col = delete(x.col, toRemove)
            x = sps.csr_matrix(x)
            return x

        m = weight(ctxList.get_sparse_matrix(targets=vocab))
        sklpp.normalize(m,copy=False)
        m = dot(m,m.transpose()) # pairwise cosine similarities
        #m = remove_negatives(m) # might be slow

        zeroRows = (m.sum(1)==0).astype(int)
        toAdd = sps.spdiags(zeroRows.transpose(), 0,
                            zeroRows.shape[0],
                            zeroRows.shape[0],format='csr')
        m = m + toAdd # deal with OOV sims

        return m

class WordnetSim:
    def __init__(self, vocab=None, method=None, multithread=False):
        self.vocab = vocab
        if not self.vocab: 
            self.vocab = []

        self.method = method
        if not self.method:
            self.method = wn.path_similarity

        self.multithread = multithread
        #self.matrix = self.get_similarity_matrix(vocab,self.method)

    def similarity(self, w, w2=None, method=None):
        """ Word similarity.

        Returns the Wordnet word similarity of two words 
        according to the optionally specified function 
        (by default pathlen). Words are passed as a (w1,w2)
        tuple.
        """
        
        if not method: method = self.method
        if not w2:
            w1 = w[0]
            w2 = w[1]
        else:
            w1 = w

        if not wn.synsets(w1) or not wn.synsets(w2):
            return 0
        if w1==w2: return 0
        sims = [method(s1,s2) for s1 in wn.synsets(w1) 
                for s2 in wn.synsets(w2)]
        toReturn = max(sims)
        if not toReturn: return 0
        else: return toReturn
  
    def get_similarity_matrix(self, vocab=None, method=None):
        """ Return a similarity matrix.
       
        Returns the similarity matrix for words; this is
        a diagonal matrix where 1 represents maximal
        similarity and 0 represents minimal similarity.
        The main diagonal is all 1, since a word has similarity 
        1 with itself.

        Args:
        vocab: Optionally, a list of vocabulary words which 
        will make up the rows/cols of the similarity matrix.
        method: Optionally, a function for computing wordnet
        similarity. By default, uses path_len.

        Returns: A CSR similarity matrix.
        """

        if not vocab: vocab = self.vocab
        if not method: method = self.method

        vocabLen = len(vocab)
        similarity = partial(self.similarity,method=method)
        pairs = [(vocab[i],vocab[j],method)
                 for i in xrange(vocabLen)
                 for j in xrange(i+1,vocabLen)]

        if self.multithread:
            cpus = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(cpus)
            m = array(pool.map(similarity,pairs))
            #pool.close() #???
        else:
            m = array(map(similarity,pairs))

        del pairs
            
        m = sps.csr_matrix(spd.squareform(m))

        zeroRows = (m.sum(1)==0).astype(int)
        toAdd = sps.spdiags(zeroRows.transpose(),0,zeroRows.shape[0],zeroRows.shape[0],format='csr')
        m = m + toAdd # deal with OOV sims

        return m
