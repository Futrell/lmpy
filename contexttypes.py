__author__ = "Richard Futrell"
__copyright__ = "Copyright 2012, Richard Futrell"
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__maintainer__ = "Richard Futrell"
__email__ = "See the author's website"

import contextparsers as ctxparse

class Context(object):
    defaultParser = ctxparse.ContextParser

    def __init__(self, word):
        self.word = word

    def __repr__(self):
        return 'Context(%s)' % str(self.__dict__)

class DependencyContext(object):
    defaultParser = ctxparse.DependencyParser

    def __init__(self, word, relation, position):
        self.word = word
        self.relation = relation
        self.position = position

class PositionalContext(object):
    defaultParser = ctxparse.PositionalParser
    
    def __init__(self, word, position):
        self.word = word
        self.position = position