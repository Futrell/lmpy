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
1. ContextList: Contains the contexts for a set of target
words. Read in a corpus with process_corpus and get 
contextual vectors with get_vector or get_matrix.

2. ContextParser: A class for converting a line of a corpus
containing a target word into a representation of the 
contexts for the target word in that line. Subclasses:

2a1. BagParser: Processes raw text into a bag-of-words
representation.

2a2. PositionalParser: Processes raw text into a bag-of-words
with context words marked for their position relative to
the target word.

2a3. NGramParser: Processes n-grams with counts into a 
bag of words with appropriate counts. To include positional
information, use PositionalNGramParser.

2b. DepsParser: Processes a dependency-parsed corpus into
features including grammatical relation.
"""

__author__ = "Richard Futrell"
__copyright__ = "Copyright 2012, Richard Futrell"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__maintainer__ = "Richard Futrell"
__email__ = "See the author's website"

import sys
import fileinput
import re
import gzip
from collections import OrderedDict
from collections import defaultdict
from collections import Counter

class ContextParser(object):
    """ A context parser.

    Contains methods for parsing a line into context features.
    A feature is a tuple whose first element is the wordform
    (or n-gram) and whose following elements have to do with
    its relations and/or position.
    """
    def __init__(self, lemmatize=lambda x:x, boundaries=False, 
                 **kwargs):
        try:
            import nltk.tokenize.treebank as tb
            self.tokenize = tb.TreebankWordTokenizer().tokenize
        except ImportError:
            print "Could not import NLTK tokenizer. Tokenizing on space instead"
            self.tokenize = lambda x : x.split(" ")
        self.lemmatize = lemmatize
        self.boundaries = boundaries
        if 'preLemmatized' in kwargs:
            def get_lemma(x):
                x = x.split(kwargs['preLemmatized'])
                return x[-1]
            def get_lemmata(xs):
                return [get_lemma(x) for x in xs]
            self.lemmatize = get_lemmata


        self._set_parameters(**kwargs)
    
    def _set_parameters(self, **kwargs): pass
    
    def parse(self, target, line):
        for word in line:
            yield (word,)

    def preprocess(self, line): 
        if self.boundaries:
            line.insert(0,'<S>')
            line.append('</S>')
        return line.lower()

    def get_count(self, line):
        return 1

class DepsParser(ContextParser):
    def __init__(self, **kwargs):
        self.tokenize = lambda x : x.split(" ")
        self._set_parameters(**kwargs)
        self._initialize_regexes()

    def _set_parameters(self, **kwargs):
        if 'limitingRels' in kwargs:
            self.limitingRels = frozenset(kwargs['limitingRels'])
        else: self.limitingRels = None
        
    def _initialize_regexes(self):
        self.XMLMatcher = re.compile("</?D>")
        self.wordNumMatcher = re.compile("-[0-9]+")
        self.parenthesisMatcher = re.compile("[\()]")
        self.commaMatcher = re.compile(", ")
    
    def preprocess(self, line):
        line = re.sub(self.XMLMatcher,"",line)
        line = re.sub(self.parenthesisMatcher," ",line)
        line = re.sub(self.commaMatcher," ",line)
        line = re.sub("VB[ZNDPG]-","VB-",line)
        line = re.sub("NNS-","NN-",line)
        line = re.sub(self.wordNumMatcher,"",line)
        # Resulting line:
        #    xcomp ready|ready#JJ to|to#TO
        return line.lower().strip()

    def parse(self, target, line):
        rel = line[0]
        if not self.limitingRels or rel in self.limitingRels:
            line = line[1:]
            line = self.lemmatize(line)
            for pos, word in enumerate(line):
                if not pos+1==target:
                    yield (word, rel, pos+1)


class BagParser(ContextParser):

    def _set_parameters(self, **kwargs):
        if 'windowSize' in kwargs:
            self.windowSize = kwargs['windowSize']
        else: self.windowSize = 3

        if 'prefix' in kwargs:
            self.prefix = kwargs['prefix']
        else: self.prefix = True

        if 'suffix' in kwargs:
            self.suffix = kwargs['suffix']
        else: self.suffix = True

    def parse(self, target, line):
        if self.prefix:
            pre = max(0,target-self.windowSize)
        else: 
            pre = target

        if self.suffix:
            post = min(len(line),target+self.windowSize+1)
        else:
            post = target

        line = self.lemmatize(line)
        return self._parse_line(target, line, range(pre, post))

    def _parse_line(self, target, line, indices):
        for i in indices:
            if not i==target:
                yield (line[i],)


class PositionalParser(BagParser):

    def _parse_line(self, target, line, indices):
        for i in indices:
            if not i==target:
                pos = i-target
                yield (line[i], pos)


class NGramParser(BagParser):
    def get_count(self, line):
        return int(line.split("\t")[-1])
    
    def preprocess(self, line):
        line = line.split("\t")[0]
        return line.lower()

    def parse(self, target, line):
        if (self.suffix and target == 0) or (self.prefix 
                                             and target == len(line)-1):
            line = self.lemmatize(line)
            return self._parse_line(target, line, range(0, len(line)))
        else: return []


class PositionalNGramParser(NGramParser, PositionalParser):
    pass
        
    
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
        """Initialize a context list with list of targets and optional
        limiting vocab. """

        if not targets: targets = []
        if not vocab: vocab = []

        self.contextCount = defaultdict(Counter)
        self._set_targets(targets)
        self._set_vocab(vocab)

        if corpus: 
            self.process_corpus(corpus)

    def _set_targets(self, targets):
        self.targets = targets
        self.limitingTargets = frozenset(targets)

    def _set_vocab(self, vocab, fixed=True, stripRelations=True):
        if fixed:
            if stripRelations:
                self.limitingVocab = frozenset([self._strip_relation(v.strip()) 
                                                for v in vocab])
            else:
                self.limitingVocab = frozenset([v.strip() for v in vocab])
        else: 
            self.limitingVocab = None
        self.vocab = OrderedDict([(v, None) for v in vocab])

    def add_context(self, target, contextItem, count=1):
        """Add a context item for a target. 

        Only adds context if either vocab is not fixed, or if
        limitingVocab contains the relevant word.
        """

        # if the ctx item has already been observed for this target, increment it
        if contextItem in self.contextCount[target]:
            self.contextCount[target][contextItem] += count
        else:
            # add to counts if it's in the vocab words, or if vocab isn't fixed
            if not self.limitingVocab or contextItem[0] in self.limitingVocab:
                self.contextCount[target][contextItem] = count
                self.vocab[contextItem] = None # add to vocab

    def process_corpus(self, corpusFile = None, parser = PositionalParser()):
        if not corpusFile: corpusFile = sys.stdin
        while True:
            try:
                line = corpusFile.readline()
            except:
                line = ''
            if line == '':
                break

            count = parser.get_count(line)
            line = parser.preprocess(line)
            words = parser.tokenize(line)
            for targetPos, target in enumerate(words):
                if not self.limitingTargets or target in self.limitingTargets: 
                    if not target in self.contextCount:
                        self.targets.append(target)
                    for contextItem in parser.parse(targetPos, words):
                        self.add_context(target,contextItem, count)

    def print_all(self):
        for target in self.contextCount:
            print target + " = "
            for context in self.contextCount[target]:
                print context + ":" + str(self.contextCount[target][context]) + ","

    def get_targets(self):
        return self.targets

    def get_vocab(self):
        return self.vocab.keys()

    def print_vector(self, target, vocab=None):
        if not vocab: vocab = self.vocab
        for context in vocab:
            print str(self.contextCount[target][context]),

    def print_matrix(self, targets=None, vocab=None):
        if not targets: targets = self.targets
        if not vocab: vocab = self.vocab
        
        for target in targets:
            self.print_vector(target, vocab)
            print

    def print_to_files(self, filename, targets=None, vocab=None):
        if not targets: targets=self.targets
        if not vocab: vocab=self.vocab
        
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
        if not vocab: vocab = self.vocab        
        return [self.contextCount[target][c] for c in vocab]

    def get_matrix(self, targets=None, vocab=None):
        if not vocab: vocab = self.vocab
        if not targets: targets = self.targets

        return [self.get_vector(t, vocab) for t in targets]

    def get_sparse_matrix(self, targets=None, vocab=None):
        if not vocab: vocab=self.vocab
        if not targets: targets=self.targets

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

        try:
            from numpy import array
            from scipy.sparse import csr_matrix, coo_matrix
            m = coo_matrix((data,(rows,cols)),shape=(len(targets),len(vocab)),dtype="float")
            return csr_matrix(m)
        except ImportError:
            print "Could not import scipy; returning parameters for a coo_matrix"
            return (data,(rows,cols))
    
        
    def read_from_files(self, filename):
        rowsFile = open(filename+'_rows','r')
        targets = rowsFile.readline().strip(",").split(",")
        rowsFile.close()
        print "Loaded rows"

        colsFile = open(filename+'_cols','r')
        vocab = colsFile.readline().strip(",").split(",")
        colsFile.close()
        print "Loaded columns"

        self.contextCount = defaultdict(defaultdict(Counter)) #TODO: make ordered dict
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

    args = parser.parse_args()
    if not args.corpus:
        corpus = sys.stdin
    else:
        corpus = open(args.corpus,'r')
    
    if args.c == 'dep' or args.c == 'deps':
        p = DepsParser(limitingRels = args.r, 
                       preLemmatized = lemmaDelimiter)
    elif args.c == 'ngram':
        p = PositionalNGramParser()
    else:
        p = PositionalParser()
    c = ContextList(targets=args.t, vocab=args.v)
    c.process_corpus(corpus, p)
    corpus.close()
    
    c.print_to_files(args.outfile)

