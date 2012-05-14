from numpy import *

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

    def normalize(self, m):
        smallNum = 0.0000000001 # hack to deal with div by 0
        norms = maximum(sqrt(dot(m,m.transpose())).diagonal(),smallNum)
        return m / norms[:,newaxis] # convert to diagonals

    def getContextMatrix (self, vocab=[], ctxList=None):
        if ctxList == None: ctxList = self.ctxList
        if vocab==[]: vocab = self.vocab
        m = array([ctxList.getVector(word) for word in vocab])
        return m
    
    def getSimilarityMatrix(self, vocab=None, ctxList = None, weight = None):
        if (vocab==None and ctxList==None and weight==None): 
            return self.matrix
        if ctxList == None: ctxList = self.ctxList
        if vocab == None: vocab = self.vocab
        if weight == None: weight = self.weight

        m = weight(self.getContextMatrix(vocab, ctxList))
        m = self.normalize(m)
        m = dot(m,m.transpose())
        for i, row in enumerate(m):
            if sum(row) == 0:
                row[i] = 1
        return m

class WordnetSim:
    from nltk.corpus import wordnet as wn
    from numpy import array

    def __init__(self, vocab, method=None):
        if self.weight == None:
            self.weight = self.nullWeight
        self.vocab = vocab
        if self.method == None:
            self.method = wn.path_similarity

    def similarity(self, w1, w2, method=None):
        if method==None: method = self.method
        sims = []
        for s1 in wn.synsets(w1):
            for s2 in wn.synsets(w2):
                sims.append(method(s1,s2))
        return min(sims)

    def getSimilarityMatrix(self, vocab=[],method=None):
        if vocab==[]: vocab = self.vocab
        if method==None: method = self.method

        m = array([
                [self.similarity(w1,w2,method) for w1 in vocab] 
                for w2 in vocab])
        
        m = self.normalize(m)
        for i, row in enumerate(m):
            if sum(row) == 0:
                row[i] = 1
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
