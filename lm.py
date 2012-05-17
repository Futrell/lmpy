from tok import Tokenizer
from collections import Counter
import numpy
import random
from scipy.stats import rv_discrete

class LanguageModel:
    def __init__(self, order=3, smoothing=None):
        if order < 1:
            print "Invalid order: " + str(order),
            print "Defaulting to order 3."
            self.order = 3
        self.order = order
        self.counts = dict()
        self.tk = Tokenizer()
        self.Starter = '<S>'
        self.Ender = '</S>'
        self.OOV = '!OOV!'
        if smoothing==None:
            from smoothing import MLE
            self.smoothing = MLE()

    def generateString(self, context='',smoothing=None):
        return ' '.join(self.generate(context,smoothing))

    def generateStrings(self, n, context='', smoothing=None):
        return [self.generateString(context,smoothing) for _ in xrange(n)]
        
    def generate(self, context='', smoothing=None):
        if smoothing == None:
            smoothing = self.smoothing #MLE() by default
        smoothing.updateCounts(self.counts)
        return [w for w in self.generateWords(context, smoothing)]

    def generateWords(self, context=[], smoothing=None):
        if smoothing == None:
            smoothing = self.smoothing #MLE() by default
        if type(context)==str:
            context = self.tk.tokenize(context)
        generated = [self.Starter for i in xrange(self.order-1)]
        if not context == []: 
            if type(context)==str:
                context = self.tk.tokenize(context)
            generated.append(context)
        random.seed()
        vocab = list(self.counts[()].keys())
        while True:
            context = tuple(generated[-(self.order-1):])
            word = self.generateWord(context,smoothing,vocab)
            if word==self.Ender or word==self.Starter: break
            yield word
            generated.append(word)

    def generateWord(self, context, smoothing=None, vocab=None):
        if smoothing==None: smoothing=self.smoothing
        if vocab==None: vocab = list(self.counts[()].keys())
        distribution = 2**smoothing.probdist(context)
        if sum(distribution)==0: return self.Ender
        words = [tuple(range(len(vocab))),
                 tuple(distribution)]
        distribution = rv_discrete(name='words',values=words)
        word = vocab[distribution.rvs()]
        return word
        

    def prob(self, text, smoothing=None, boundariesCount=False, verbose=False):
        prob = 0.0
        if type(text) == str:
            text = self.tk.tokenize(text)
        text = self.addDelimiters(text)
        for i in xrange(self.order-1,len(text)):
            word = text[i]
            if word == self.Ender: break
            
            context = text[(i-(self.order-1)):i]
            while self.Starter in context:
                context.remove(self.Starter)

            if verbose:
                print 'p(',word,'|',context,') =',
                print  self.p(word, context, smoothing)
            prob += self.p(word, context, smoothing)

        return prob
        
    def p(self, word, context=tuple(), smoothing=None, **kwargs):
        if smoothing == None:
            smoothing = self.smoothing
        smoothing.updateCounts(self.counts)

        if type(context) == str:
            context = self.tk.tokenize(context)
        context = context[-(self.order-1):]

        return smoothing.prob(word, context)

    def addText(self, text, order=0):
        if order == 0: order = self.order
        if type(text) == str:
            text = self.tk.tokenize(text)
        text = self.addDelimiters(text,order)
        for o in xrange(1,order+1):
            for i in xrange(len(text)-(o-1)):
                self.addGram(text[i:i+o])
        if self.smoothing is not None:
            self.smoothing.updateCounts(self.counts)

    def addDelimiters(self, text, order=0):
        if order==0: order = self.order
        if type(text) == str:
            text = self.tok.tokenize(text)
        text.insert(0,self.Starter)
        text.append(self.Ender)
        for i in xrange(order-2):
            text.insert(0,self.Starter)
            text.append(self.Ender)
        return text

    def addGram(self, text):
        if text == []:
            return
        context = tuple(text[:-1])
        word = text[-1:][0]
        if context not in self.counts:
            self.counts[context] = Counter()
        if word is not self.Starter:
            self.counts[context][word] += 1
        #print "Added gram:",context,":",word
        
    def addTextFile(self, infile, order=0):        
        if type(infile) == str:
            infile = open(infile,'r')
        for line in infile:
            self.addText(line.strip())
    
    def getVocab(self):
        vocab = self.counts[()].keys()
        if self.OOV not in vocab:
            vocab.append(self.OOV)
        return vocab
