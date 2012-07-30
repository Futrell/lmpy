__author__ = "Richard Futrell"
__copyright__ = "Copyright 2012, Richard Futrell"
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__maintainer__ = "Richard Futrell"
__email__ = "See the author's website"

import numpy as np
import sklearn.preprocessing as sklpp
import contexttypes as ctxtypes

DEFAULT_CONTEXT_TYPE = ctxtypes.Context

class Typicality(object):
	def __init__(self, bgCtx, fgCorpusFilename=None, weight=None):
		self.bgCtx = bgCtx
		
		if fgCorpusFilename:
			infile = open(fgCorpusFilename, 'r')
			self.fgCorpus = [line.strip() for line in infile]
			infile.close()
		else:
			self.fgCorpus = None

		self.weight = weight

	def typicality(self, word, sentence, bgVector=None):
		ctxtype = self.bgCtx.contextType
		parser = ctxtypes.defaultParser

		if not bgVector:
			bgVector = np.array(self.bgCtx.get_vector(word))

		contexts = [c for c in parser(sentence.index(word), sentence)]
		fgVector = np.array(self.bgCtx.contexts_to_vector(contexts))
		return _cosine_similarity(fgVector, bgVector)

	def typicalities(self, word, sentences, bgVector=None):
		if not bgVector:
			bgVector = np.array(self.bgCtx.get_vector(word))

		return [(self.typicality(word, s, bgVector=bgVector), s) 
				for s in sentences]

	def most_typical_sentences(self, word, corpus=None, n=5):
		if not corpus:
			corpus = self.fgCorpus
		return _Xst_typical_sentences(word, corpus, n, reverse=True)

	def least_typical_sentences(self, word, corpus=None, n=5):
		if not corpus:
			corpus = self.fgCorpus
		return _Xst_typical_sentences(word, corpus, n, reverse=False)

	def _Xst_typical_sentences(self, word, corpus, n, reverse):
		typicalities = sorted(typicalities(word, self.fgCorpus), reverse=reverse)
		return typicalities[-n:]

	def _cosine_similarity(one, two):
		sklpp.normalize(one, copy=False)
		sklpp.normalize(two, copy=False)
		return np.dot(one, two)