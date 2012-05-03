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
from distribMat import ContextList
from distribMat import ContextMatrix

############################################################
# Read arguments from command line.

def readArguments():
    targetsFile = ""
    corpusFile = ""
    vocabFile = ""
    relation = ""

    try:
        opts, args = getopt.getopt(sys.argv[1:],"t:c:v:r:")
    except getopt.GetoptError, err:
        print str(err)
        usage()

    for opt, arg in opts:
        if opt in ("--target","-t"):
            targetsFile = arg
        elif opt in ("--corpus","-c"):
            corpusFile = arg
        elif opt in ("--vocab","-v"):
            vocabFile = arg
        elif opt in ("--relation","-r"):
            relation = arg
    return targetsFile, corpusFile, vocabFile, relation

############################################################
# Main.

targetsFilename,corpusFilename,vocabFilename,relationsFile = readArguments()

if vocabFilename == "":
    vocab = []
else:
    vocabFile = open(vocabFilename,'r')
    vocab = [line.strip() for line in vocabFile]
    vocabFile.close

targetsFile = open(targetsFilename,'r')
contextList = ContextList(targetsFile.readlines(),vocab)
targetsFile.close

if corpusFilename == '':
    corpusFile = sys.stdin
else:
    corpusFile = open(corpusFilename,'r')

if relationsFile == '':
    contextList.processCorpus(corpusFile)
else:
    relationsFile = open(relationsFile,'r')
    relations = set([r.strip() for r in relationsFile])
    contextList.processCorpus(corpusFile,relations)
    relationsFile.close()

contextList.removeUnfoundTargets()
contextList.printToFiles('context')
contextList.printAsMatrix()
#print contextList.getMatrix(vocab)
