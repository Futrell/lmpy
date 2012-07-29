"""
This code contains classes for the collection of contextual
vectors for words.

A corpus is read and target words are found in it; the 
contexts of those words are parsed into features using
one of the ContextParser classes and stored in ContextList.

The __main__ method reads a corpus from standard input and,
with specified parameters, builds a ContextList for that
corpus. 

Classes:
ContextList: Contains the contexts for a set of target
words. Read in a corpus with process_corpus and get 
contextual vectors with get_vector or get_matrix.
"""

__author__ = "Richard Futrell"
__copyright__ = "Copyright 2012, Richard Futrell"
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__maintainer__ = "Richard Futrell"
__email__ = "See the author's website"

import sys, fileinput, gzip
import re
from collections import Counter, defaultdict, OrderedDict
from defaultordereddict import DefaultOrderedDict
import contextparsers as ctxparse

DEFAULT_CONTEXT_PARSER = ctxparse.PositionalParser()

class ContextList(object):
    """ ContextList class.
    
    A ContextList object is a list of target words and the
    counts of the contexts that those target words appear in.
    Its basic parameters are the target words and the vocabulary
    of possible contexts. Context counts for the target words
    are derived from a corpus using the process_corpus method.
    If vocabulary is not specified, all contexts will be counted.
    """

    def __init__(self, targets=None, vocab=None, corpus=None):
        """Initialize a context list with list of limiting targets,
        limiting vocab, and an optional corpus to process."""

        if not targets: targets = []
        if not vocab: vocab = []

        self.contextCount = defaultdict(Counter)
        self._set_targets(targets)
        self._set_vocab(vocab)
        self.contextType = None

        if corpus: 
            self.process_corpus(corpus)

    def __contains__(self, item):
        return item in self.contextCount

    def _set_targets(self, targets):
        self.targets = OrderedDict() # an ordered set (dict -> True)
        self.limitingTargets = frozenset([t.strip() for t in targets])

    def _set_vocab(self, vocab):
        self.vocab = OrderedDict() # an ordered set
        self.limitingVocab = frozenset([v.strip() for v in vocab])

    def _add_to_targets(self, target):
        self.targets[target] = True

    def _add_to_vocab(self, vocabItem):
        self.vocab[vocabItem] = True

    def contexts_to_vector(self, contexts, vocab=None):
        if not vocab:
            vocab = self.vocab
        contextCounter = Counter(c)
        return [contextCounter[c] for c in vocab]

    def add_context(self, target, contextItem, count=1):
        """Add a context item for a target. 

        Only adds context if either vocab is not fixed, or if
        limitingVocab contains the relevant word.
        """

        if not self.limitingVocab or contextItem.word in self.limitingVocab:
            self.contextCount[target][contextItem] += count
            if contextItem not in self.vocab:
                self._add_to_vocab(contextItem)

    def process_corpus(self, corpusFile=None, parser=DEFAULT_CONTEXT_PARSER):
        if not corpusFile: 
            corpusFile = sys.stdin

        if not self.contextType:
            self.contextType = parser.contextType
        elif parser.contextType != self.contextType:
            print('Warning! Old contexts used %s, new ones use %s!' 
                % (self.contextType, parser.contextType))
            

        while True:
            try:
                line = corpusFile.readline()
            except:
                line = ''
            if line == '':
                break

            count = parser.get_count(line) # 1 unless data is ngrams
            line = parser.preprocess(line) # get rid of XML and junk
            words = parser.tokenize(line)
            for targetPos, target in enumerate(words):
                if not self.limitingTargets or target in self.limitingTargets: 
                    if not target in self.targets:
                        self._add_to_targets(target)
                    for contextItem in parser.parse(targetPos, words):
                        self.add_context(target, contextItem, count)

    def print_all(self):
        for target in self.contextCount:
            print target + " = "
            for context in self.contextCount[target]:
                print context + ":" + str(self.contextCount[target][context]) + ","

    def get_targets(self):
        return self.targets.keys()

    def get_vocab(self):
        return self.vocab.keys()

    def print_to_files(self, filename, targets=None, vocab=None):
        if not targets: 
            targets = self.targets
        if not vocab: 
            vocab = self.vocab
        
        rowsFile = open("%s_rows" % filename, 'w')
        colsFile = open("%s_cols" % filename, 'w')
        matFile = open("%s_mat" % filename, 'w')

        for target in targets:
            rowsFile.write("%s," % target)
            for context in vocab:
                colsFile.write("%s," % context)
                matFile.write(str(self.contextCount[target][context]))
                matFile.write(" ")

        rowsFile.close()
        colsFile.close()
        matFile.close()
                
    def get_vector(self, target, vocab=None):
        if not vocab: 
            vocab = self.vocab        
        return [self.contextCount[target][c] for c in vocab]

    def get_matrix(self, targets=None, vocab=None):
        if not vocab: 
            vocab = self.vocab
        if not targets: 
            targets = self.targets

        return [self.get_vector(t, vocab) for t in targets]

    def get_sparse_matrix(self, targets=None, vocab=None):
        """ Get sparse matrix:

        Return a CSR matrix representation of the contexts. If numpy
        and scipy can't be imported, return the parameters for a 
        COO matrix. 
        """

        if not vocab: 
            vocab = self.vocab
        if not targets: 
            targets = self.targets

        targetIndices = {v:i for i,v in enumerate(targets)}
        vocabIndices = {v:i for i,v in enumerate(vocab)}

        data = []
        rows = []
        cols = []
        for t in targets:
            for c in self.contextCount[t]:
                if c in vocabIndices:
                    data.append(self.contextCount[t][c])
                    rows.append(targetIndices[t])
                    cols.append(vocabIndices[c])

        cooMatrixParameters = (data, (rows, cols))
        try:
            from numpy import array
            from scipy.sparse import csr_matrix, coo_matrix
            m = coo_matrix(cooMatrixParameters,
                shape=(len(targets),len(vocab)),dtype="float")
            return csr_matrix(m)
        except ImportError:
            print "Could not import scipy; returning parameters for a coo_matrix"
            return cooMatrixParameters
    
        
    def read_from_files(self, filename):
        rowsFile = open(filename+'_rows','r')
        targets = rowsFile.readline().strip(",").split(",")
        rowsFile.close()
        print "Loaded rows"

        colsFile = open(filename+'_cols','r')
        vocab = colsFile.readline().strip(",").split(",")
        colsFile.close()
        print "Loaded columns"

        self.contextCount = defaultdict(Counter) #TODO: make ordered dict
        matFile = gzip.open(filename+'_mat.gz','r')
        for lineNum,line in enumerate(matFile):
            vector = [int(v) for v in line.strip().split(" ")]
            for i,v in enumerate(vector):
                if v>0:
                    self.contextCount[targets[lineNum]][vocab[i]] = v
                  
        matFile.close()
        print "Loaded matrix"
        
        self._set_targets(targets)
        self._set_vocab(vocab)
        print "Set internal parameters."


def _lines_as_list(filename):
    if filename:
        infile = open(args.t,'r')
        lines = [x.strip() for x in infile.readlines()]
        infile.close()
    else:
        lines = []
    return lines

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert contexts to vectors.')
    parser.add_argument('-t, --targets', metavar='t', type=str, dest='t',
                        help='a file containing target words', default='')
    parser.add_argument('-v, --vocab', metavar='v', type=str, dest='v',
                        help='a file containing limiting vocabulary for contexts',
                        default='')
    parser.add_argument('-r, --relations', metavar='r', type=str, dest='r',
                        help='a file containing limiting relations for dependency contexts',
                        default='')
    parser.add_argument('-c, --corpus-type', metavar='c', type=str, dest='c',
                        help='corpus type: dep for dependency-parsed corpus, ngram for ngrams',
                        default='')
    parser.add_argument('-f, --corpus-file', metavar='f', type=str, dest='corpus',
                        help="corpus file: in case you'd rather not read from stdin",
                        default='')
    parser.add_argument('-o, --outfile', metavar='o', type=str, dest='outfile',
                        help="output file: output will be printed to this filename",
                        default='out')
    parser.add_argument('-l, --lemma-delimiter', metavar='l', type=str, dest='lemmaDelimiter',
                        help="delimiter if dependency corpus is pre-lemmatized",
                        default='')

    args = parser.parse_args()
    if not args.corpus:
        corpus = sys.stdin
    else:
        corpus = open(args.corpus,'r')

    t = _lines_as_list(args.t)
    v = _lines_as_list(args.v)
    r = _lines_as_list(args.r)
    
    if args.c == 'dep' or args.c == 'deps':
        p = DepsParser(limitingRels = r, 
                       preLemmatized = args.lemmaDelimiter)
    elif args.c == 'ngram':
        p = PositionalNGramParser()
    else:
        p = PositionalParser()


    c = ContextList(targets=t, vocab=v)
    c.process_corpus(corpus, p)
    corpus.close()

    c.print_to_files(args.outfile)

