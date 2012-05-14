from numpy import *

def pos(f):
    return lambda x : maximum(0,f(x))

def pmi(x):
    Pwf, Pf, Pw = getProbs(x)
    return log2(Pwf) - log2(Pw * Pf)

def ttest(x):
    Pwf, Pf, Pw = getProbs(x)
    return (Pwf - (Pf*Pw))/sqrt(Pf*Pw)

def getProbs(x):
    x = x.astype(float)
    Pwf = x / x.sum() #P(w,f)
    Pf = x / x.sum(0) #divide features by row sums
    Pw = (x.transpose() / x.sum(1)).transpose() #div words by col sums
    return Pwf, Pf, Pw
