from numpy import *
import scipy.sparse as sps

def pos(f):
    def inner(x):
        x = f(x)
        if sps.issparse(x):
            x = x.todense()
        return sps.csr_matrix(x.clip(min=0))
    return inner

def pmi(x):
    o, e = getProbs(x) 
    o = log2(o) #let log2(0) = 0
    e = log2(e)
    result = o-e
    result[isinf(result)]=0
    return sps.csr_matrix(result)

def ttest(x):
    o, e = getProbs(x)
    return (o-e)/sqrt(e) # elementwise /

def getProbs(x):
    x = x.astype(float)
    normalizer = x.sum()
    
    Pwf = memmap('o',shape=(x.shape[0],x.shape[1]),dtype='float',mode='write')
    if sps.issparse(x):
        Pwf.data[:] = x.todense()
    else: Pwf.data[:] = x
    Pwf = Pwf/normalizer

    Pw = x.sum(1) / normalizer
    Pf = x.sum(0) / normalizer
    EPwf = memmap('e',shape=(x.shape[0],x.shape[1]),dtype='float',mode='write')
    EPwf.data[:] = Pw*Pf

    return Pwf, EPwf
