from tok import Tokenizer
from collections import Counter
import numpy

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

    def generate(self, n=1, smoothing=None):
        if smoothing == None:
            smoothing = self.smoothing #MLE() by default
        smoothing.updateCounts(self.counts)
        return [self.generateString(smoothing) for _ in xrange(n)]

    def generateString(self, smoothing=None):
        import random
        from scipy.stats import rv_discrete
        
        if smoothing == None:
            smoothing = self.smoothing #MLE() by default
        generated = []
        random.seed()
        context = tuple(self.Starter for i in xrange(self.order-1))
        vocab = list(self.counts[()].keys())
        while True:
            distribution = 2**smoothing.probdist(context)
            print distribution
            if sum(distribution) == 0: break
            words = [tuple(range(len(vocab))),
                     tuple(distribution)]
            distribution = rv_discrete(name='words',values=words)
            word = vocab[distribution.rvs()]
            if word == self.Ender or word == self.Starter: break

            generated.append(word)
            #print word,
            context = list(context)
            context.append(word)
            context = tuple(context[1:])

        return ' '.join(generated)

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
