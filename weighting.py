from numpy import *

def pos(x):
    return maximum(x,0)

def pmi(x):
    Pwf, Pf, Pw = getProbs(x)
    return log2(Pwf / (Pw * Pf))

def ttest(x):
    Pwf, Pf, Pw = getProbs(x)
    return (Pwf - (Pf*Pw))/sqrt(Pf*Pw)

def getProbs(x):
    x = x.astype(float)
    Pwf = x / x.sum() #P(w,f)
    Pf = x / x.sum(0) #divide features by row sums
    Pw = (x.transpose() / x.sum(1)).transpose() #div words by col sums
    return Pwf, Pf, Pw
