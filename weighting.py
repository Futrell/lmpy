from numpy import *
import scipy.sparse as sps

def pos(f):
    return lambda x: sps.csr_matrix(maximum(0,f(x).todense()))

def pmi(x):
    PwfObserved, PwfExpected = getProbs(x) 
    PwfObserved.data[:] = log2(PwfObserved.data) #let log2(0) = 0
    PwfExpected.data[:] = log2(PwfExpected.data)
    return PwfObserved - PwfExpected

def ttest(x):
    o, e = getProbs(x)
    denom = e
    denom.data[:] = sqrt(denom.data)
    
    return (o-e)/denom # elementwise /

def getProbs(x):
    x = x.astype(float)
    if not sps.issparse(x):
        x = sps.csr_matrix(x)
    Pwf = x / x.sum() #P(w,f)

    Pw = x.sum(1)
    Pw /= Pw.sum()

    Pf = x.sum(0)
    Pf /= Pf.sum()
    return Pwf, sps.csr_matrix(Pw * Pf)
