from numpy import *
from collections import Counter
import scipy.sparse as sps
import sklearn.preprocessing as sklpp

class Smoothing:
    def __init__(self, counts=dict(), **kwargs):
        self.counts = counts
        self.params = dict()
        for key in kwargs:
            self.params[key] = kwargs[key]
        self.setDefaultParams()
        self.vocab = []
        self.OOV = '!OOV!'
        self.updateVocab()

    def setDefaultParams(self):
        pass

    def updateCounts(self, counts): 
        self.counts = counts
        self.updateVocab()

    def updateVocab(self):
        if () in self.counts:
            self.vocab = self.counts[()].keys()
        if self.OOV not in self.vocab:
            self.vocab.append(self.OOV)

    def prob(self, word, context, pdist=None):
        if type(context) is not tuple: 
            context = tuple(context)
        if pdist==None:
            pdist = sps.csr_matrix(self.probdist(context))

        if word not in self.vocab:
            return self.prob(self.OOV, context, pdist)

        return pdist[self.vocab.index(word)]

    def probdist(self, context):
        if type(context) is not tuple:
            context = tuple(context)

        return self.normalize(self.localCounts(context))

    def localCounts(self, context):
        if type(context) is not tuple:
            context = tuple(context)

        if context in self.counts:
            localCounts = self.counts[context]
        else:
            localCounts = Counter()

        # can this be sped up using sparse matrices?
        distribution = array([localCounts[w] for w in self.vocab]) 
        return distribution

    def normalize(self, distribution):
        normalizer = sum(distribution)
        if normalizer == 0:
            return log2(distribution)
        else:
            return log2(distribution) - log2(normalizer)

class MLE(Smoothing):
    def placeholder(self):
        pass

class AdditiveSmoothing(Smoothing):
    """Implements add-k smoothing, taking keyword parameter k."""

    def setDefaultParams(self):
        if 'k' not in self.params:
            self.params['k'] = 1

    def probdist(self, context):
        if type(context) is not tuple:
            context = tuple(context)

        return self.normalize(self.localCounts(context)+self.params['k'])


class SimilaritySmoothing(Smoothing):
    """Implements similarity-based smoothing based on the formula
    in Erk, Pado & Pado (2010). Requires either a similarity
    matrix, with values between 0 and 1, or a "sim" object,
    which can be anything that exports a similarity matrix.
    1 should represent maximal similarity. """

    def setDefaultParams(self):
        if 'mat' not in self.params:
            if 'sim' not in self.params:
                print "I need a similarity matrix or a similarity object!"
            else:
                self.params['mat'] = self.params['sim'].getSimilarityMatrix()
        if 'matvocab' in self.params:
            self.params['mat'] = self.fixVocab(self.params['mat'], self.params['matvocab'])

    def fixVocab(self, mat, vocab):
        print 'Not implemented yet.'
        return mat

    def localCounts(self, context):
        if type(context) is not tuple:
            context = tuple(context)

        if context in self.counts:
            localCounts = self.counts[context]
        else:
            localCounts = Counter()

        distribution = array([localCounts[w] for w in self.vocab])
        return dot(distribution,self.params['mat'])


class AdaptiveSimilaritySmoothing(SimilaritySmoothing):
    def setDefaultParams(self):
        self.params['mat'] = self.makeSimilarityMatrix(self.counts)

    def makeSimilarityMatrix(self, counts):
        counts = self.counts2array(counts).transpose() #matrix with words as rows
        sklpp.normalize(counts,copy=False) #l2 norm
        return dot(counts,counts.transpose())

    def counts2array(self, counts):
        return array([
            [self.count(counts[ctx],w) 
             for w in self.vocab]
            for ctx in counts 
            if len(ctx)==self.order
            ])
