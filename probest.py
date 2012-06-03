from numpy import *
from collections import Counter
import scipy.sparse as sps
from scipy.stats import rv_discrete
import random
import sklearn.preprocessing as sklpp

class MLE:
    """ Maximum Likelihood Estimator

    A class for returning MLE probability estimates
    based on stored occurrence counts.
    This class is mostly useful for subclassing.

    Contains a dict of counts and methods prob for
    returning the probability of a word given context,
    and probdist for returning a probability distribution
    of words given a context.
    """

    def __init__(self, counts=dict(), **kwargs):
        self.counts = counts
        self.params = dict()
        for key in kwargs:
            self.params[key] = kwargs[key]
        self._set_default_params()
        self.vocab = []
        self.OOV = '!OOV!'
        self.update_vocab()
        random.seed()

    def _set_default_params(self):
        pass

    def update_counts(self, counts): 
        """ Update counts.

        Replace internal counts dict with the specified dict,
        and update the internal vocabulary accordingly.
        """
        self.counts = counts
        self.update_vocab()

    def update_vocab(self, vocab=[]):
        """ Synchronize the internal vocabulary with the 
        current counts dict. """
        if vocab==[]:
            if () in self.counts:
                self.vocab = self.counts[()].keys()
        else: 
            self.vocab = vocab

        if self.OOV not in self.vocab:
            self.vocab.append(self.OOV)

    def prob(self, word, context, pdist=None):
        if type(context) is not tuple: 
            context = tuple(context)
        if pdist==None:
            pdist = self.probdist(context)

        if word not in self.vocab:
            return self.prob(self.OOV, context, pdist)

        return pdist[self.vocab.index(word)]

    def probdist(self, context):
        if type(context) is not tuple:
            context = tuple(context)

        return self.normalize(self._local_counts(context))

    def _local_counts(self, context):
        if type(context) is not tuple:
            context = tuple(context)

        if context in self.counts:
            localCounts = self.counts[context]
        else:
            localCounts = Counter()

        # can this be sped up using sparse matrices?
        c = array([localCounts[w] for w in self.vocab]) 
        return c

    def normalize(self, distribution):
        #print distribution
        normalizer = distribution.sum()
        if normalizer == 0:
            return log2(distribution)
        else:
            return log2(distribution) - log2(normalizer)

    def generate_word(self, context, vocab=None, sentinel='</S>'):
        """Generate a single word. 

        Sample from the probability distribution of words following
        context. If there are no possible continuations,
        end the string by appending the sentinel word.
        """
        
        if vocab==None: vocab=self.vocab
        distribution = 2**self.probdist(context)
        if sum(distribution)==0: return sentinel
        words = [tuple(range(len(vocab))),
                 tuple(distribution)]
        distribution = rv_discrete(name='words',values=words)
        word = vocab[distribution.rvs()]
        return word


class AdditiveSmoothing(MLE):
    """Implements add-k smoothing, taking keyword parameter k."""

    def _set_default_params(self):
        if 'k' not in self.params:
            self.params['k'] = 1

    def probdist(self, context):
        if type(context) is not tuple:
            context = tuple(context)

        return self.normalize(self._local_counts(context)+self.params['k'])


class SimilaritySmoothing(MLE):
    """Implements similarity-based smoothing based on the formula
    in Erk, Pado & Pado (2010). Requires either a similarity
    matrix, with values between 0 and 1, or a "sim" object,
    which can be anything that exports a similarity matrix.
    1 should represent maximal similarity. """

    def _set_default_params(self):
        if 'mat' not in self.params:
            if 'sim' not in self.params:
                print "I need a similarity matrix or a similarity object!"
            else:
                self.params['mat'] = self.params['sim'].get_similarity_matrix()
        if 'matvocab' in self.params:
            self.params['mat'] = self._fix_vocab(self.params['mat'], self.params['matvocab'])

    def _fix_vocab(self, mat, vocab):
        print 'Matvocab correction not implemented yet.'
        return mat

    def probdist(self, context):
        if type(context) is not tuple:
            context = tuple(context)

        distribution = sps.csr_matrix(self._local_counts(context))
        distribution = dot(distribution,self.params['mat'])
        return self.normalize(array(distribution.todense())[0,:])

class BackoffSmoothing(MLE):

    def _local_counts(self, context):
        print 'Warning! Backoff smoothing works so far for generation, but not for probability estimation.'

        if type(context) is not tuple:
            context = tuple(context)

        if context in self.counts:
            localCounts = self.counts[context]
        else:
            return _local_counts(context[1:])

        # can this be sped up using sparse matrices?
        c = array([localCounts[w] for w in self.vocab]) 
        return c

class BackoffSimilaritySmoothing(SimilaritySmoothing,BackoffSmoothing):
    def placeholder(self): pass

class AdaptiveSimilaritySmoothing(SimilaritySmoothing):
    def _set_default_params(self):
        self.params['mat'] = self._make_similarity_matrix(self.counts)

    def _make_similarity_matrix(self, counts):
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
