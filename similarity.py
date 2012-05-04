class DistribSim:
    from numpy import *

    def __init__(self, ctxList, weight=None):
        self.ctxList = ctxList
        self.weight = weight
        if self.weight == None:
            self.weight = self.nullWeight

    def nullWeight(self, x):
        return x

    def normalize(self, m):
        norms = array([self.norm(v) for v in m])
        return m / norms[:,newaxis]

    def norm(self, v):
        v = array(v)
        return sqrt(dot(v,v.conj()))
    
    def getContextMatrix (self, vocab=[], ctxList=None):
        if ctxList == None: ctxList = self.ctxList
        m = array([ctxList.getVector(word) for word in vocab])
        return m
    
    def getSimilarityMatrix(self, vocab=[], ctxList = None, weight = None):
        if vocab==[]:
            vocab = self.ctxList.getVocab()
        if ctxList == None:
            ctxList = self.ctxList
        if weight == None:
            weight = self.weight

        m = self.normalize(m)
        m = weight(self.getContextMatrix(vocab, ctxList))
        m = dot(m,m.transpose())
        for i, row in enumerate(m):
            if sum(row) == 0:
                row[i] = 1
        return m
