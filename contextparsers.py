"""
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
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__maintainer__ = "Richard Futrell"
__email__ = "See the author's website"

import contexttypes as ctxtypes

DEFAULT_LEMMATIZE = lambda x: x

class ContextParser(object):
    """ A context parser.

    Contains methods for parsing a line into context features.
    A feature is a tuple whose first element is the wordform
    (or n-gram) and whose following elements have to do with
    its relations and/or position.
    """
    contextType = ctxtypes.Context

    def __init__(self, lemmatize=DEFAULT_LEMMATIZE, boundaries=False, 
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
            yield self.contextType(word)

    def preprocess(self, line): 
        if self.boundaries:
            line.insert(0,'<S>')
            line.append('</S>')
        return line.lower()

    def get_count(self, line):
        return 1

class DepsParser(ContextParser):
    contextType = ctxType.DependencyContext

    def __init__(self, lemmatize=DEFAULT_LEMMATIZE, **kwargs):
        self.tokenize = lambda x : x.split(" ")
        self.lemmatize = lemmatize
        if 'preLemmatized' in kwargs:
            preLemmatized = kwargs['preLemmatized']
            if preLemmatized:
                def get_lemma(x):
                    x = x.split(preLemmatized)
                    return x[-1]
                def get_lemmata(xs):
                    return [get_lemma(x) for x in xs]
                self.lemmatize = get_lemmata
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
                    yield self.contextType(word, rel, pos+1)


class BagParser(ContextParser):
    contextType = ctxtypes.Context

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
                yield self.contextType(line[i])


class PositionalParser(BagParser):
    contextType = ctxtypes.PositionalContext

    def _parse_line(self, target, line, indices):
        for i in indices:
            if not i==target:
                pos = i-target
                yield self.contextType(line[i], pos)

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

class PositionalNGramParser(NGramParser, PositionalParser):
    pass