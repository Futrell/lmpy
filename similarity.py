from numpy import *

class DistribSim:
    # Usage example:
    # sim = DistribSim(lm.getVocab(), ctxList, pos(pmi))
    # simSmooth = SimilaritySmoothing(mat=sim.getSimilarityMatrix())
    # lm.generate(1,simSmooth)
    # Or more concisely:
    # lm.generate(1,SimilaritySmoothing(sim=DistribSim(lm.getVocab(),ctxList)))
    def __init__(self, vocab, ctxList, weight=None):
        self.ctxList = ctxList
        self.weight = weight
        if self.weight == None:
            self.weight = self.nullWeight
        self.vocab = vocab

    def nullWeight(self, x):
        return x

    def normalize(self, m):
        norms = array([self.norm(v) for v in m])
        return m / norms[:,newaxis]

    def norm(self, v):
        v = array(v)
        n = sqrt(dot(v,v.conj()))
        if n == 0:
            return 1
        else: return n
    
    def getContextMatrix (self, vocab=[], ctxList=None):
        if ctxList == None: ctxList = self.ctxList
        if vocab==[]: vocab = self.vocab
        m = array([ctxList.getVector(word) for word in vocab])
        return m
    
    def getSimilarityMatrix(self, vocab=[], ctxList = None, weight = None):
        if ctxList == None: ctxList = self.ctxList
        if vocab == []: vocab = self.vocab
        if weight == None: weight = self.weight

        m = weight(self.getContextMatrix(vocab, ctxList))
        m = self.normalize(m)
        m = dot(m,m.transpose())
        for i, row in enumerate(m):
            if sum(row) == 0:
                row[i] = 1
        return m
