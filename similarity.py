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
    # Usage example:
    # simSmooth = SimilaritySmoothing(sim=DistribSim(lm.getVocab(), ctxList, pos(pmi)))
    # lm.generate(1,simSmooth)
    # Or more concisely:
    # lm.generate(1,SimilaritySmoothing(sim=DistribSim(lm.getVocab(),ctxList)))
    def __init__(self, vocab, ctxList, weight=None):
        self.ctxList = ctxList
        self.weight = weight
        if self.weight == None:
            self.weight = self.nullWeight
        self.vocab = vocab
        self.matrix = self.getSimilarityMatrix(vocab=vocab)

    def nullWeight(self, x):
        return x

    def getContextMatrix (self, vocab=[], ctxList=None):
        if ctxList == None: ctxList = self.ctxList
        if vocab==[]: vocab = self.vocab
        return ctxList.getCSRMatrix(targets=vocab) # sparse row matrix
    
    def getSimilarityMatrix(self, vocab=None, ctxList = None, weight = None):
        if (vocab==None and ctxList==None and weight==None): 
            return self.matrix
        if ctxList == None: ctxList = self.ctxList
        if vocab == None: vocab = self.vocab
        if weight == None: weight = self.weight

        m = weight(ctxList.getCSRMatrix(targets=vocab))
        sklpp.normalize(m,copy=False)
        m = dot(m,m.transpose()) # pairwise cosine similarities

        zeroRows = (m.sum(1)==0).astype(int)
        toAdd = spdiags(zeroRows.transpose(),0,zeroRows.shape[0],zeroRows.shape[0],format='csr')
        m = m + toAdd # deal with OOV sims

        return m

class WordnetSim:

    def __init__(self, vocab, method=None, multithread=False):
        self.vocab = vocab
        self.method = method
        if self.method == None:
            self.method = wn.path_similarity
        self.multithread = multithread
        self.matrix = self.getSimilarityMatrix(vocab,self.method)

    def similarity(self, w, method=None):
        if method==None: method = self.method
        w1 = w[0]
        w2 = w[1]
        if not wn.synsets(w1) or not wn.synsets(w2):
            return 0
        if w1==w2: return 0
        sims = [method(s1,s2) for s1 in wn.synsets(w1) 
                for s2 in wn.synsets(w2)]
        toReturn = max(sims)
        if toReturn==None: return 0
        else: return toReturn
  
    def getSimilarityMatrix(self, vocab=[],method=None):
        if vocab==[]: vocab = self.vocab
        if method==None: method = self.method

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
            
        m = spd.squareform(m)
        fill_diagonal(m,1)
        return m
