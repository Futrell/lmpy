from nltk.tokenize import TreebankWordTokenizer as Tokenizer
from collections import Counter
import numpy
import random
from scipy.stats import rv_discrete

class LanguageModel:
    """A language model class.
    
    Stores counts of n-grams of specified order, and generates
    or assigns probability to strings using the specified
    estimation method, by default MLE.
    """
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
        """Generate a string.
        
        Generate a string starting with specified prefix context
        and specified smoothing method.
        """
        return ' '.join(self.generate(context,smoothing))

    def generateStrings(self, n, context='', smoothing=None):
        """Generate multiple strings. """
        return [self.generateString(context,smoothing) for _ in xrange(n)]
        
    def generate(self, context='', smoothing=None):
        """Generate a list of words.
        
        Generate a string conditioned on previous context,
        starting with context <S> and user-specified context,
        which can be a string or a list of word tokens. 
        """
        if smoothing == None:
            smoothing = self.smoothing #MLE() by default
        smoothing.updateCounts(self.counts)
        generated=[]
        if not context=='':
            if type(context)==str:
                context = self.tk.tokenize(context)
            generated.extend(context)
        generated.extend([w for w in self.generateWords(context, smoothing)])
        return generated

    def generateWords(self, context=[], smoothing=None):
        """Generate words.
   
        This method yields words conditioned on precious
        context. Words are generated then added to the
        accumulated context so far, which is the context
        for the generation of the next word. Generation
        stops when the Ender word is reached.
        """
        if smoothing == None:
            smoothing = self.smoothing #MLE() by default
        if type(context)==str:
            context = self.tk.tokenize(context)
        generated = [self.Starter for i in xrange(self.order-1)]
        if not context==[]:
            generated.extend(context)
        random.seed()
        vocab = list(self.counts[()].keys())
        while True:
            context = tuple(generated[-(self.order-1):])
            word = self.generateWord(context,smoothing,vocab)
            if word==self.Ender or word==self.Starter: break
            yield word
            generated.append(word)

    def generateWord(self, context, smoothing=None, vocab=None):
        """Generate a single word. 

        Get the probability distribution of words following
        context, with specified smoothing and vocab of 
        possible words. If there are no possible continuations,
        end the string by appending the Ender word.
        """
        if smoothing==None: smoothing=self.smoothing
        if vocab==None: vocab = list(self.counts[()].keys())
        distribution = 2**smoothing.probdist(context)
        if sum(distribution)==0: return self.Ender
        words = [tuple(range(len(vocab))),
                 tuple(distribution)]
        distribution = rv_discrete(name='words',values=words)
        word = vocab[distribution.rvs()]
        return word
        

    def prob(self, text, smoothing=None, verbose=False):
        """Determine the probability of a string.

        Tokenize a string, then get its probability according to
        the language model with specified smoothing. 

        In the case of trigrams, 
        calculates p(w0)p(w1|w0)p(w2|w0,w1)p(w3|w1,w2)...
        
        Set verbose to True to see all transitional probabilities.
        """
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
        """Probability of a word after a context.
        
        Takes a word and context, which can be either a tuple
        or a string. Context is truncated to fit the order of
        the model.
        """
        if smoothing == None:
            smoothing = self.smoothing
        smoothing.updateCounts(self.counts)

        if type(context) == str:
            context = self.tk.tokenize(context)
        context = context[-(self.order-1):]

        return smoothing.prob(word, context)

    def addText(self, text, order=0):
        """Add text to the language model.
        
        Breaks a text into 1:n-grams and stores those grams.
        Text can be either a list of tokens or a string, in
        which case it is tokenized.
        """
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
        """ Add delimiters to a text.

        Append the appropriate number of Starter and Ender
        words to the beginning and end of a list of tokens,
        or of a string.
        """
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
        """ Add a single n-gram to the model. 

        This adds the 1:n-grams from a given string 
        to the counts of the language model, converting
        each of those lists to tuples.
        """
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
        """ Add a text file to the model. """
        if type(infile) == str:
            infile = open(infile,'r')
        for line in infile:
            self.addText(line.strip())
    
    def getVocab(self):
        """ Return the possible words, plus the OOV word."""
        vocab = self.counts[()].keys()
        if self.OOV not in vocab:
            vocab.append(self.OOV)
        return vocab
