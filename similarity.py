from numpy import *
import scipy.spatial.distance as spd
import scipy.sparse as sps
import sklearn.preprocessing as sklpp
from nltk.corpus import wordnet as wn


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
        m = array([ctxList.getVector(word) for word in vocab])
        return sps.csr_matrix(m) # sparse row matrix
    
    def getSimilarityMatrix(self, vocab=None, ctxList = None, weight = None):
        if (vocab==None and ctxList==None and weight==None): 
            return self.matrix
        if ctxList == None: ctxList = self.ctxList
        if vocab == None: vocab = self.vocab
        if weight == None: weight = self.weight

        m = sps.csr_matrix(weight(self.getContextMatrix(vocab, ctxList)))
        sklpp.normalize(m,copy=False)
        m = sps.dia_matrix(dot(m,m.transpose())) # pairwise cosine similarities
        m.data[m.offsets.tolist().index(0),:] = 1 # deal with OOV sims
        return array(m.todense())

class WordnetSim:

    def __init__(self, vocab, method=None):
        self.vocab = vocab
        self.method = method
        if self.method == None:
            self.method = wn.path_similarity
        self.matrix = self.getSimilarityMatrix(vocab,self.method)

    def similarity(self, w1, w2, method):
        if method==None: method = self.method
        if not wn.synsets(w1) or not wn.synsets(w2):
            return 0
        if w1==w2: return 0
        sims = []
        for s1 in wn.synsets(w1):
            for s2 in wn.synsets(w2):
                sim = method(s1,s2)
                if sim: sims.append(sim)
        if sims: return min(sims)
        else: return 0
  
    def getSimilarityMatrix(self, vocab=[],method=None):
        if vocab==[]: vocab = self.vocab
        if method==None: method = self.method

        vocabLen = len(vocab)
        m = array(
            [self.similarity(vocab[i],vocab[j],method) 
             for i in xrange(vocabLen) 
             for j in xrange(i+1,vocabLen)])
        
        m = spd.squareform(m)
        m = self.normalize(m)
        fill_diagonal(m,1)
        return m

    def normalize(self, m):
        norms = array([self.norm(v) for v in m])
        return m / norms[:,newaxis]

    def norm(self, v):
        v = array(v)
        n = max(v)
        if n == 0:
            return 1.
        else: return float(n)
