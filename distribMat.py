#!/usr/bin/env python

"""
This program takes a list of target words and reads a dependency-
parsed corpus to assemble vectors representing the contexts of
each target word. The program:
1. Collects a list of contexts for each target word,
2. Converts the context list into a vector representation where
columns represent the same context items across target word vectors.
The resulting matrix is then printed so that it can be manipulated
in MATLAB or the like.

With some modification of the regular expressions used in the
processCorpus method of ContextList, this code should be usable
for corpora with other kinds of formating/markup.
"""

import sys
import fileinput
import getopt
import re
import cPickle as pickle
import csv
import gzip

###########################################################
class ContextList:

    def __init__(self, targets=[], vocab=[]):
        """Initialize a context list with list of targets and optional
        limiting vocab. """
        self.contextCount = dict()
        for target in targets:
            target = target.strip()
            self.contextCount[target] = dict()
        self.targets = frozenset(self.contextCount.keys())

        self.setVocab(vocab)
        if len(self.vocabWords) == 0:
            self.fixedVocab = False
        else:
            self.fixedVocab = True
            self.vocabWords = frozenset(self.vocabWords)
        

    def addContext(self, target, contextItem):
        """Add a context item for a target. Assume target is already 
        in the dict contextCount.
        Only add context if either vocab is not fixed, or if
        vocabWords contains the relevant word and vocab is fixed."""

        contextItem = self.cleanLine(target,contextItem)
        bareContextItem = self.stripRelation(contextItem)

        # if the ctx item has already been observed for this target, increment it
        if contextItem in self.contextCount[target]:
            self.contextCount[target][contextItem] += 1
        else:
            # add to counts if it's in the vocab words, or if vocab isn't fixed
            if not self.fixedVocab or bareContextItem in self.vocabWords:
                self.contextCount[target][contextItem] = 1
                self.vocab.append(contextItem)

            # if vocab isn't fixed, add this word to the vocab words
            if not self.fixedVocab:
                self.vocabWords.add(bareContextItem)


    def processCorpus(self, corpusFile):
        while True:
            try:
                line = sys.stdin.readline()
            except:
               line = ''
            if line == '':
                break

            relationLabel = self.getRelationLabel(line)
            #print "Processing line " + line + " which has relation " + relationLabel

            if relations == [] or relationLabel in relations:
                matches = re.findall('([\w#]+)-[0-9]+', line) # returns 2 matches
                for (i,m) in enumerate(matches):
                    if relations == [] or relations[relationLabel] == 0 or relations[relationLabel] == i+1:
                        if m in self.targets: # assume targets given in advance
                            self.addContext(m,line)

    def getRelationLabel(self, line): return None

    def stripRelation(self, contextItem): return contextItem

    def cleanLine(self, target, line): return line

    def printAll(self):
        for target in self.contextCount:
            print target + " = "
            for context in self.contextCount[target]:
                print context + ":" + str(self.contextCount[target][context]) + ","


    def getTargets(self):
        return self.contextCount.keys()


    def getVocab(self):
        return self.vocab


    def getContextCounts(self):
        return self.contextCount


    def printAsMatrix(self, vocab=[]):
        if vocab == []:
            vocab = self.vocab
        
        for target in self.getTargets():
            for context in vocab:
                if context in self.contextCount[target]:
                    print str(self.contextCount[target][context]),
                else:
                    print '0',
            print


            # Print the contents of this instance of the class.
            # If options are given, print those contents instead.
    def printToFiles(self, filename, targets=[], vocab = [], onlyMat = False):
        if targets==[]: targets=self.targets
        if vocab==[]: vocab=self.vocab
        
        rowsFile = open("%s_rows" % filename, 'w')
        colsFile = open("%s_cols" % filename, 'w')
        matFile = open("%s_mat" % filename, 'w')

        for target in targets:
            if not onlyMat:
                rowsFile.write("%s," % target)
            for context in vocab:
                if not onlyMat:
                    colsFile.write("%s," % context)
                try:
                    matFile.write(str(self.contextCount[target][context]))
                except KeyError:
                    matFile.write(str(0))
                matFile.write(" ")

        rowsFile.close()
        colsFile.close()
        matFile.close()
                
        
			
    def getVector(self, target, vocab=[]):
        if vocab == []:
            vocab = self.vocab
            
        contextVector = []
        for context in vocab:
            if target not in self.contextCount or context not in self.contextCount[target]:
                contextVector.append(0)
            else:
                contextVector.append(self.contextCount[target][context])
        return contextVector
                

    def getMatrix(self, vocab=[], targets=[]):
        if vocab == []:
            vocab = self.vocab

        if targets == []:
            targets = self.getTargets()
            
        contextMatrix = []
        for target in targets:
            contextMatrix.append(self.getVector(target,vocab))
        return contextMatrix


    def getSparseMatrix(self, vocab=[], targets=[]):
        if vocab==[]: vocab=self.vocab
        if targets==[]: targets=self.getTargets()
        vocab = {v:i for i,v in enumerate(vocab)}

        m = [(self.contextCount[t][c], 
              (targets.index(t),vocab[c]))
             for t in targets for c in self.contextCount[t]
             if c in vocab]
        try:
            from numpy import array
            from scipy.sparse import csr_matrix, coo_matrix
            m = coo_matrix(m,shape=(len(targets),len(vocab)))
            return csr_matrix(m)
        except:
            print "Could not import scipy; returning parameters for a coo_matrix"
            return m
        # COO matrix is a [(data,(i,j))] list.
        


    def removeUnfoundTargets(self, vocab=[]):
        if vocab == []:
            vocab = self.vocab
            
        for target in self.contextCount.keys():
            if len(self.contextCount[target]) == 0:
                self.contextCount.pop(target)
                self.targets = self.contextCount.keys()


    def removeUnfoundContexts(self):
        vocab = self.vocab
        for context in vocab:
            if all(context not in self.contextCount[target] for target in self.contextCount):
                self.vocab.remove(context)


    def setVocab(self, vocab):
        self.vocab = []
        self.vocabWords = set([])
        for vocabWord in vocab:
            self.vocabWords.add(vocabWord.strip())


    def readFromFiles(self, filename):
        rowsFile = open(filename+'_rows','r')
        targets = rowsFile.readline().strip(",").split(",")
        rowsFile.close()
        print "Loaded rows"

        colsFile = open(filename+'_cols','r')
        self.vocab = colsFile.readline().strip(",").split(",")
        colsFile.close()
        print "Loaded columns"

        self.contextCount = dict()
        matFile = gzip.open(filename+'_mat.gz','r')
        for lineNum,line in enumerate(matFile):
            vector = [int(v) for v in line.strip().split(" ")]
            if targets[lineNum] not in self.contextCount:
                self.contextCount[targets[lineNum]] = dict()
            nonZeroIndices = [n for n,i in enumerate(vector) if i>0]
            for i,v in enumerate(vector):
                if v>0:
                    self.contextCount[targets[lineNum]][self.vocab[i]] = v
                  
        matFile.close()
        print "Loaded matrix"
        
        self.removeUnfoundTargets()
        self.vocabWords = frozenset([self.stripRelation(x) for x in self.vocab])
        self.vocab = set(self.vocab)
        print "Set internal vocabulary"


class DependencyContextList(ContextList):
    # Process a corpus file or other iterable, with optional
    # restriction of relations to be counted. Relations is an
    # iterable containing permitted relations in the format
    # relation/number where number indicates the argument
    # position of the target word.
    def processCorpus(self, corpusFile, relations=[]):
        if not relations == []:
            relations = dict([r.split("/") for r in relations])
            for r in relations.keys():
                relations[r] = int(relations[r])

        while True:
            try:
                line = sys.stdin.readline()
            except:
               line = ''
            if line == '':
                break

            #line = line[3:-5] # get rid of XML
            line = re.sub("\^","#",line)
            line = re.sub("VB[ZNDPG]","VB",line)
            line = re.sub("NNS","NN",line)
            relationLabel = self.getRelationLabel(line)
            #print "Processing line " + line + " which has relation " + relationLabel

            if relations == [] or relationLabel in relations:
                matches = re.findall('([\w#]+)-[0-9]+', line) # returns 2 matches
                for (i,m) in enumerate(matches):
                    if relations == [] or relations[relationLabel] == 0 or relations[relationLabel] == i+1:
                        if m in self.targets: # assume targets given in advance
                            self.addContext(m,line)


    def cleanLine(self, target, line):
        line = re.sub('[\(),]',' ',line) # kill parens and commas
        line = re.sub(' \w+\|',' ',line) # kill wordforms
        line = re.sub(' ' + target + '-'," X-",line) # kill target w
        line = re.sub(r'-[0-9]+','',line) # kill numbers
        line = re.sub(' +',' ',line) # kill extraneous spaces
        return line.strip()


    def stripRelation(self, context):
        context = re.sub(r'^[^ ]+','',context)
        context = re.sub(r' X','',context)
        return context.strip()


    def getRelationLabel(self, context):
        return context.split('(')[0]

    def getRelationLabelLater(self, context):
        return context.split(' ')[0]

    def getContextVectors(self,target,relations=[]):
        if not relations == []:
            relations = frozenset([r.split("/")[0] for r in relations])
            
        contexts = dict() #dict of (word,rel) -> counts
        counts = dict()
        for context in self.contextCount[target]:
            (contextWord,contextRel) = (self.stripRelation(context),self.getRelationLabelLater(context))
            if contextWord in self.contextCount: #is that context itself a target?
                #print (contextWord,contextRel)
                if relations == [] or contextRel in relations:
                    contexts[(contextWord,contextRel)] = self.contextCount[target][context]
                    if contextWord not in counts:
                        counts[contextWord] = 0
                    counts[contextWord] += self.contextCount[target][context]
                    

        targetWords = [x[0] for x in contexts]
        ctxList = ContextList(targetWords)
        ctxList.fixedVocab = False
        for (targetWord,targetRel) in contexts:
            if targetWord in self.contextCount:
                for v in self.contextCount[targetWord].keys():
                    (vWord,vRel) = (self.stripRelation(v),self.getRelationLabelLater(v))
                    #if vRel == targetRel:
                    #ctxList.vocabWords.add(vWord)
                    ctxList.vocab.append(v)
                ctxList.contextCount[targetWord] = self.contextCount[targetWord]
        ctxList.removeUnfoundTargets()
        ctxList.targetFreqs = counts
        return ctxList





