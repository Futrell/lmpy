import sys
import fileinput
import getopt
import re
import cPickle as pickle
import csv
import gzip

###########################################################
class ContextItem(object):
    """ A context item.

    A context item has a word, a rel(ation), and a pos(ition).
    For instance "nsubj X doctor" would have word doctor, rel 
    nsubj, and pos 1.
    """
    def __init__(self, word=None, rel=None, pos=None):
        self.word = word
        self.rel = rel
        self.pos = pos

class ContextParser(object):
    """ A context parser.

    Contains methods for parsing a line into contexts.
    """
    def __init__(self):
        try:
            import nltk.tokenize.treebank as tb
            self.tokenize = tb.TreebankWordTokenizer.tokenize
        except ImportError:
            print "Could not import NLTK tokenizer. Tokenizing on space instead."
            self.tokenize = lambda x: x.split(" ")
    
    def parser(self, **kwargs):
        return lambda target, line: self.parse(target, line, **kwargs)

    def parse(self, target, line, **kwargs):
        return line

    def preprocess(self, line): return line 

class DepsParser(ContextParser):
    
    def __init__(self):
        self.tokenize = lambda x: x.split(" ")
        self.XMLMatcher = re.compile("</?D>")
        self.wordNumMatcher = re.compile("-[0-9]+")
        self.parenthesisMatcher = re.compile("[\()]")
        self.commaMatcher = re.compile(", ")
    
    def preprocess(self, line):
        line = re.sub(self.XMLMatcher,"",line)
        line = re.sub(self.parenthesisMatcher,"",line)
        line = re.sub(self,commaMatcher,"",line)
        line = re.sub("\^","#",line)
        line = re.sub("VB[ZNDPG]-","VB-",line)
        line = re.sub("NNS-","NN-",line)
        line = re.sub(self.wordNumMatcher,"",line)
        # Resulting line:
        #    xcomp ready|ready#JJ to|to#TO
        return line.strip()

    def parse(self, target, line, limitingRels=None, 
              preLemmatized=True, lemmatize=lambda x:x):
        rel = line[0]
        if rel in limitingRels:
            for pos,word in enumerate(line[1:]):
                yield ContextItem(word,rel,pos+1)


class BagParser(ContextParser):
    def parse(target, line, windowSize=3, prefix=True, suffix=True,
              lemmatize=lambda x:x):
        if not suffix:
            line = self._remove_after_target(target, line)
        if not prefix:
            line = self._remove_before_target(target, line)
            target = 0

        pre = target-windowSize
        post = target+windowSize
        line = lemmatize(line)
        return self._parse_line(target, line, pre, post)
        
    def _parse_line(self, target, line, pre, post):
        for i in xrange(pre, post+1):
            if not i==target:
                yield ContextItem(line[i])

    def _remove_after_target(self, targetpos, line):
        return line[targetpos:]

    def _remove_before_target(self, targetpos, line):
        return line[:targetpos]


class PositionalParser(BagParser):
    def _parse_line(self, target, line, pre, post):
        for i in xrange(pre, post+1):
            pos = i-target
            if pos:
                yield ContextItem(line[i], pos=pos)
        
    
class ContextItem(object):
    """ A context item.

    A context item has a word, a rel(ation), and a pos(ition).
    For instance "nsubj X doctor" would have word doctor, rel 
    nsubj, and pos 1.
    """
    def __init__(self, ):
        self.word = word
        self.rel = rel
        self.pos = pos


class ContextList(object):
    """ ContextList class.
    
    A ContextList object is a list of target words and the
    counts of the contexts that those target words appear in.
    Its basic parameters are the target words and the vocabulary
    of possible contexts. Context counts for the target words
    are derived from a corpus using the process_corpus method.
    If vocabulary is not specified, all contexts will be counted.
    """

    def __init__(self, targets=None, vocab=None):
        """Initialize a context list with list of targets and optional
        limiting vocab. """

        if not targets: targets = []
        if not vocab: vocab = []

        self.contextCount = defaultdict(defaultdict(Counter)) #TODO: make ordered dict
        self._set_targets(targets)
        self._set_vocab(vocab)

    def _set_targets(self, targets):
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
        self.vocab = set(vocab) #TODO: Make ordered set.

    def add_context(self, target, contextItem):
        """Add a context item for a target. 

        Only adds context if either vocab is not fixed, or if
        limitingVocab contains the relevant word.
        """

        # if the ctx item has already been observed for this target, increment it
        if contextItem in self.contextCount[target]:
            self.contextCount[target][contextItem] += 1
        else:
            # add to counts if it's in the vocab words, or if vocab isn't fixed
            if not self.limitingVocab or contextItem.word in self.limitingVocab
                self.contextCount[target][contextItem] = 1
                self.vocab.add(contextItem)

    def process_corpus(self, corpusFile = None, parser = ContextParser()):
        if not corpusFile: corpusFile = sys.stdin
        while True:
            try:
                line = corpusFile.readline()
            except:
                line = ''
            if line == '':
                break

            line = parser.preprocess(line)
            words = parser.tokenize(line)
            for targetPos, target in enumerate(words):
                if not self.limitingTargets or target in self.limitingTargets: 
                    for contextItem in parser(targetPos, contextItem):
                        self.add_context(target,contextItem)


    def print_all(self):
        for target in self.contextCount:
            print target + " = "
            for context in self.contextCount[target]:
                print context + ":" + str(self.contextCount[target][context]) + ","

    def get_targets(self):
        return self.contextCount.keys()

    def get_vocab(self):
        return self.vocab

    def print_matrix(self, targets=None, vocab=None):
        if not targets: targets = self.contextCount.keys()
        if not vocab: vocab = self.vocab
        
        for target in targets:
            for context in vocab:
                print str(self.contextCount[target][context]),
            print


    def print_to_files(self, filename, targets=None, vocab=None):
        if not targets: targets=self.contextCount.keys()
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
        if not targets: targets = self.contextCount.keys()

        return [self.get_vector(t, vocab) for t in targets]

    def get_sparse_matrix(self, targets=None, vocab=None):
        if not vocab: vocab=self.vocab
        if not targets: targets=self.contextCount.keys()

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
