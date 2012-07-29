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
	def __init__(self, bgCtx=None, weight=None):
		self.bgCtx = bgCtx

	def typicality(self, word, sentence, bgVector=None):
		ctxtype = self.bgCtx.contextType
		parser = ctxtype.defaultParser

		if not bgVector:
			bgVector = np.array(self.bgCtx.get_vector(word))

		contexts = parser(sentence.index(word), sentence)
		fgVector = np.array(self.bgCtx.contexts_to_vector(sentence))
		return _cosine_similarity(fgVector, bgVector)

	def typicalities(self, word, sentences):
		bgVector = np.array(self.bgCtx.get_vector(word))

	def most_typical_sentences(self, word, sentences, n=5):
		return _Xst_typical_sentences(word, sentences, n, reverse=True)

	def least_typical_sentences(self, word, sentences, n=5):
		return _Xst_typical_sentences(word, sentences, n, reverse=False)

	def _Xst_typical_sentences(self, word, sentences, n, reverse):
		typicalities = sorted(typicalities(word, sentences), reverse=reverse)
		return typicalities[-n:]

	def _cosine_similarity(vectorOne, vectorTwo):
		sklpp.normalize(vectorOne, copy=False)
		sklpp.normalize(vectorTwo, copy=False)
		return np.dot(vectorOne, vectorTwo)