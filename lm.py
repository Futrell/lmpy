#from nltk.tokenize import TreebankWordTokenizer as Tokenizer
from tok import Tokenizer
from collections import Counter
import numpy as np

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
        self.probEst = smoothing # might be None
        if self.probEst==None:
            from probest import MLE
            self.probEst = MLE()

    def generate_string(self, context='',smoothing=None):
        """Generate a string.
        
        Generate a string starting with specified prefix context
        and specified smoothing method.
        """
        return ' '.join(self.generate(context,smoothing))

    def generate_strings(self, n, context='', smoothing=None):
        """Generate multiple strings. """
        return [self.generate_string(context,smoothing) for _ in xrange(n)]
        
    def generate(self, context=None, smoothing=None, boundaryMatters=True):
        """Generate a list of words.
        
        Generate words conditioned on previous context,
        starting with context <S> plus user-specified context,
        which can be a string or a list of word tokens. 

        Set boundaryMatters=False to generate from a random 
        beginning, rather than from <S>.
        """
        if not context: context = []
        if smoothing == None:
            smoothing = self.probEst #MLE() by default
        smoothing.update_counts(self.counts)
        if boundaryMatters:
            prefix = [self.Starter for i in xrange(self.order-1)]
        else:
            prefix = []

        if type(context)==str:
            context = self.tk.tokenize(context)
        prefix.extend(context)

        generated = context # probably []
        generated.extend([w for w in self.word_generator(prefix, smoothing)])
        return generated

    def word_generator(self, context=None, smoothing=None, vocab=[]):
        """A generator for words.
   
        This method yields words conditioned on previous
        context. Words are generated then added to the
        accumulated context so far, which is the context
        for the generation of the next word. Generation
        stops when the Ender word is reached.
        """
        if not context: context = []
        probEst = smoothing
        if probEst == None:
            probEst = self.probEst #MLE() by default
        if type(context)==str:
            context = self.tk.tokenize(context)
        generated=context

        while True:
            context = tuple(generated[-(self.order-1):])
            word = probEst.generate_word(context,sentinel=self.Ender)
            while word==self.OOV: #regenerate until no OOV is generated
                word = probEst.generate_word(context,sentinel=self.Ender)
            if word==self.Ender or word==self.Starter: break

            yield word
            generated.append(word)

    def prob(self, text, smoothing=None, verbose=False):
        """Determine the probability of a string.

        Tokenize a string, then get its probability according to
        the language model with specified smoothing. 

        Start- and end-tokens don't matter, e.g. for trigrams
        calculates p(w0)p(w1|w0)p(w2|w0,w1)p(w3|w1,w2),...
        not p(w0|<S>,<S>) etc.
        
        Set verbose=True to see all transitional probabilities.
        """
        prob = 0.0
        if type(text) == str:
            text = self.tk.tokenize(text)
        text = self.add_delimiters(text)
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
        
    def p(self, word, context=tuple(), smoothing=None):
        """Probability of a word after a context.
        
        Takes a word and context, which can be either a tuple
        or a string. Context is truncated to fit the order of
        the model.
        """
        if smoothing == None:
            smoothing = self.probEst
        smoothing.update_counts(self.counts)

        if type(context) == str:
            context = self.tk.tokenize(context)
        context = context[-(self.order-1):]

        return smoothing.prob(word, context)

    def probdist(self, context=tuple(), smoothing=None):
        """Probability of a word after a context.
        
        Takes a word and context, which can be either a tuple
        or a string. Context is truncated to fit the order of
        the model.
        """
        probEst = smoothing
        if probEst == None:
            probEst = self.probEst
        probEst.update_counts(self.counts)

        if type(context) == str:
            context = tuple(self.tk.tokenize(context))
        if type(context) == list:
            context = tuple(context)

        d = probEst.probdist(context)
        return {v : 2**d[i] 
                for i,v in enumerate(probEst.vocab)
                if not np.isinf(d[i])}



    def add_text(self, text, order=0):
        """Add text to the language model.
        
        Breaks a text into 1:n-grams and stores those grams.
        Text can be either a list of tokens or a string, in
        which case it is tokenized.
        """
        if order == 0: order = self.order
        if type(text) == str:
            text = self.tk.tokenize(text)
        text = self.add_delimiters(text,order)
        for o in xrange(1,order+1):
            for i in xrange(len(text)-(o-1)):
                self.add_gram(text[i:i+o])
        if self.probEst is not None:
            self.probEst.update_counts(self.counts)

    def add_delimiters(self, text, order=0):
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

    def add_gram(self, text):
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
        
    def add_text_file(self, infile, order=0):        
        """ Add a text file to the model. """
        if type(infile) == str:
            infile = open(infile,'r')
        for line in infile:
            self.add_text(line.strip())
    
    def get_vocab(self):
        """ Return the possible words, plus the OOV word."""
        vocab = self.counts[()].keys()
        if self.OOV not in vocab:
            vocab.append(self.OOV)
        return vocab
