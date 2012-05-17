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
    PwfObserved, PwfExpected = getProbs(x)
    numerator = (PwfObserved - PwfExpected)
    denominator = PwfExpected 
    denominator.data[:] = sqrt(denominator.data)
    return numerator/denominator # elementwise /

def getProbs(x):
    x = x.astype(float)
    if not sps.issparse(x):
        x = sps.csr_matrix(x)
    Pwf = x / x.sum() #P(w,f)

    Pw = x.sum(1)
    Pw /= Pw.sum(0)

    Pf = x.sum(0)
    Pf /= Pf.sum(1)
    return Pwf, sps.csr_matrix(Pw * Pf)
