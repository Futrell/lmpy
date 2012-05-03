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
        self.smoothing = smoothing
        

    def generate(self, num=1, smoothing=None):
        return [self.generateString(smoothing) for i in xrange(num)]

    def generateString(self, smoothing=None):
        import random
        from scipy.stats import rv_discrete
        
        generated = ''
        random.seed()
        context = tuple(self.Starter for i in xrange(self.order-1))
        vocab = list(self.counts[()].keys())
        while True:            
            words = [tuple(range(len(vocab))),tuple(2**self.p(w,context,smoothing) for w in vocab)]
            distribution = rv_discrete(name='words',values=words)
            word = vocab[distribution.rvs()]
            if word == self.Ender or word == self.Starter: break

            generated += word + ' '
            context = list(context)
            context.append(word)
            context = tuple(context[1:])

        return generated.strip()

    def ppl(self, text, smoothing=None):
        perplexity = 0.0
        if type(text) == str:
            text = self.tk.tokenize(text)
        text = self.addDelimiters(text)
        for i in xrange(self.order-1,len(text)):
            #print text[i],
            #print text[(i-(self.order-1)):i],
            #print self.p(text[i],text[(i-(self.order-1)):i])
            word = text[i]
            if word == self.Ender: return perplexity

            context = text[(i-(self.order-1)):i]
            if self.Starter in context:
                context.remove(self.Starter)

            perplexity += self.p(word, context, smoothing)
            i += 1
        return perplexity
        
    def p(self, word, context=tuple(), smoothing=None, **kwargs):
        if smoothing == None:
            smoothing = self.smoothing
        else:
            smoothing.updateCounts(self.counts)

        if type(context) == str:
            context = self.tk.tokenize(context)
        context = context[-(self.order-1):]

        if smoothing == None:
            return self.mle(word, context)
        else:
            return smoothing.prob(word, context)

    def mle(self, word, context=tuple()):
        if type(context) == str:
            context = self.tk.tokenize(context)
        context = tuple(context)
        if context in self.counts:
            count = self.counts[context][word]
            return self.MLEnormalize(count,context)
        else:
            return numpy.log2(0)

    def MLEnormalize(self, count, context):
        allCounts = [self.counts[context][word] for word in self.counts[()]]
        denominator = sum(allCounts)
        return numpy.log2(count) - numpy.log2(denominator)

    def addText(self, text, order=0):
        if order == 0: order = self.order
        if type(text) == str:
            text = self.tk.tokenize(text)
        text = self.addDelimiters(text,order)
        for i in xrange(len(text)):
            self.addGrams(text[i:i+order])
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

    def addGrams(self, text):
        if text == []:
            return
        context = tuple(text[:-1])
        word = text[-1:][0]
        if context not in self.counts:
            self.counts[context] = Counter()
        self.counts[context][word] += 1
        self.addGrams(text[1:])
        
    def addTextFile(self, infile, order=0):
        
        if type(infile) == str:
            infile = open(infile,'r')

        
            
