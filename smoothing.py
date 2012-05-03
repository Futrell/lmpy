from numpy import *
from collections import Counter

class Smoothing:
    def __init__(self, counts=dict(), **kwargs):
        self.counts = counts
        self.params = dict()
        for key in kwargs:
            self.params[key] = kwargs[key]
        self.setDefaultParams()
        self.vocab = []
        self.updateVocab()

    def setDefaultParams(self, **kwargs):
        pass

    def updateCounts(self, counts): 
        self.counts = counts
        self.updateVocab()

    def updateVocab(self):
        if () in self.counts:
            self.vocab = self.counts[()].keys()

    def prob(self, word, context):
        if type(context) is not tuple: 
            context = tuple(context)
        
        if context in self.counts:
            num = self.numerator(self.counts[context], word)
            denom = self.normalizer(self.counts[context])
        else:
            num = self.numerator(Counter(),word)
            denom = self.normalizer(Counter())

        if denom == 0:
            return log2(0)
        return log2(num) - log2(denom)

    def numerator(self, counts, word):
        return counts[word]

    def normalizer(self, counts):
        return sum([self.numerator(counts, x) for x in self.vocab])

class AdditiveSmoothing(Smoothing):
    """Implements add-k smoothing, taking keyword parameter k."""

    def setDefaultParams(self):
        if 'k' not in self.params:
            self.k = 1
        else: self.k = self.params['k']

    def numerator(self, counts, word):
        return counts[word] + self.k


class SimilaritySmoothing(Smoothing):
    """Implements similarity-based smoothing based on the formula
    in Erk, Pado & Pado (2010). Requires either a similarity
    matrix, with values between 0 and 1, or a "sim" object,
    which can be anything that exports a similarity matrix.
    1 should represent maximal similarity. """

    def setDefaultParams(self):
        if 'mat' in self.params:
            self.mat = self.params['mat']
            del self.params['mat']
            if 'sim' in self.params:
                del self.params['sim']

        elif 'sim' in self.params:
            sim = self.params['sim']
            self.mat = self.makeMatrix(sim)
            del self.params['sim']

        else:
            print "I need a similarity matrix or a similarity object!"
            self.mat = array([[]])
            #ABORT

        #self.normalizeMatrix()
    
    def normalizeMatrix(self):
        """ Row-normalize the internal similarity matrix so the
        highest value is 1. """
        rowMax = self.mat.max(axis=1)
        self.mat /= rowMax[:,numpy.newaxis]

    def makeMatrix(self, sim):
        return sim.getSimilarityMatrix(self.vocab)

    def prob(self, word, context):
        if type(context) is not tuple: 
            context = tuple(context)

        localCounts = []
        for w in self.vocab:
            if context in self.counts:
                localCounts.append(self.counts[context][w])
            else:
                localCounts.append(0)
        adjustedCounts = dot(localCounts,self.mat)
        adjustedCounts = dict(zip(self.vocab,adjustedCounts))

        num = self.numerator(adjustedCounts, word)
        denom = self.normalizer(adjustedCounts)
        return log2(num) - log2(denom)
        
