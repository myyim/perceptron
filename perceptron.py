import numpy as np
import pylab
import itertools
import math

# parameters for perceptron
eta = 1 # learning rate
Niter = 100 # maximum number of iterations

def nCr(n,r):
    """nCr(n,r) returns n choose r"""
    if n>=r:
        return math.factorial(n)/math.factorial(r)/math.factorial(n-r)
    else:
        return 0

def cover(n,p):
    """cover(n,p) returns the number of linearly separable pattern sets out of p sets given n inputs in general position
        based on Cover's counting theorem (Cover 1965)."""
    count = 0
    for j in range(np.min([n,p])):
        count += 2*nCr(p-1,j)
    return count

def perceptron(u,yloc,is_thre=1):
    """perceptron(u,yloc,is_thre=1) returns 1 if the the given dichotomy is realizable and 0 otherwise.
        u is a NXP matrix representing the P patterns in N-dimensional space.
        yloc is a list of positive labels from 0 to P-1.
        is_thre determines whether threshold is incorporated in the algorithm.
        """
    N = u.shape[0]
    P = u.shape[1]
    y = np.zeros(P)
    y[yloc] = 1
    theta = 0.
    w = np.zeros(N)
    sigma = np.zeros(P)
    itrial = 0
    while itrial < Niter and np.sum(np.abs(y-sigma))>0:
        for ix in range(P):
            sigma[ix] = (np.dot(w,u[:,ix])-theta) > 0
            w += eta*(y[ix]-sigma[ix])*u[:,ix]
            if is_thre:
                theta += -eta*(y[ix]-sigma[ix])
        itrial += 1
        sloc = np.argwhere(np.dot(w,u)-theta>0).T[0]
    if len(sloc)!=len(yloc) or np.sum(sloc==yloc)!=len(yloc):
        return 0
    else:
        return 1

def realizability(u):
    """realizability(u) prints out positive labels of all realizable dichotomies given patterns u."""
    P = u.shape[1]
    for K in range(1,P/2+1):
        print '----'
        print 'K = '+str(K)
        com = [list(temp) for temp in itertools.combinations(range(P),K)]
        count = 0
        for ic in range(len(com)):
            if perceptron(u,com[ic]):
                count += 1
                print com[ic]
        print str(count)+' out of '+str(len(com))+' are realizable.'

# Examples below
if 0:
    u = np.array([[1,0,1,0,1,0],[0,1,0,1,0,1],[1,0,0,1,0,0],[0,1,0,0,1,0],[0,0,1,0,0,1]])
    realizability(u)
    u = pylab.rand(5,6)
    realizability(u)
