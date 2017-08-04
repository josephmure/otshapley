import numpy as np
import itertools
import openturns as ot
import pandas as pd

from .indices import Indices
from .kriging import KrigingIndices


class ShapleyIndices(Indices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        Indices.__init__(self, input_distribution)
        # Add shapley function
        self.first_order_indice_func = None


class ShapleyKrigingIndices(KrigingIndices, ShapleyIndices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        KrigingIndices.__init__(self, input_distribution)
        ShapleyIndices.__init__(self, input_distribution)


def shapley_indices(method, m, model, Xall, Xcond, d, Nv, No, Ni=3):
    """Permutation Shapley indices.
    
    Parameters
    ----------
    method : str,
        The kind of permutations.
    m : int,
        The number of permutations (if random).
    model : callable,
        The model function.
    Xall : array,
    """
    if (method == 'exact'):
        #perms = list(itertools.permutations(range(d), d))
        #m = len(perms) # number of permutation
        perms_tool = ot.KPermutations(d, d)
        perms = perms_tool.generate()
        m = perms.getSize() # number of permutation
    elif (method == 'random'):
        perms = [np.random.permutation(d) for i in range(m)]
    else:
        raise ValueError('Unknow permutation method {0}'.format(method))
    
    # Creation of the design matrix
    X = np.zeros((Nv + m*(d-1)*No*Ni, d))
    X[:Nv, :] = Xall(Nv)

    for i_p, perm in enumerate(perms):
        idx_perm_sorted = np.argsort(perm)
        for j in range(d-1):
            Sj = perm[:j+1] # set of the 0st-(j-1)th elements in perm
            Sjc = perm[j+1:] # set of the jth-dth elements in perm
            xjcM = Xcond(No, Sjc, None, None) # sampled values of the inputs in Sjc
            for l, xjc in enumerate(xjcM):
                # sample values of inputs in Sj conditional on xjc
                xj = Xcond(Ni, Sj, Sjc, xjc)
                xx = np.c_[xj, [xjc]*Ni]
                ind_inner = Nv + i_p*(d-1)*No*Ni + j*No*Ni + l*Ni
                X[ind_inner:ind_inner+Ni, :] = xx[:, idx_perm_sorted]
    
    # Model evaluation
    y = model(X)

    # Initialize Shapley, main and total Sobol effects for all players
    Sh = np.zeros(d)
    Vsob = np.zeros(d)
    Tsob = np.zeros(d)
    nV = np.zeros(d) # number of samples used to estimate V1,...,Vd
    nT = np.zeros(d) # number of samples used to estimate T1,...,Td
    
    # Estimate Var[Y]
    Y = y[:Nv]
    y = y[Nv:]
    EY = np.mean(Y)
    VarY = np.var(Y, ddof=1)

    # Estimate Shapley, main and total Sobol effects
    cVar = np.zeros(No)

    for perm in perms:
        prevC = 0
        for j in range(d):
            if (j == (d-1)):
                Chat = VarY
                delta = Chat - prevC
                Vsob[perm[j]] = Vsob[perm[j]] + prevC # first order effect
                nV[perm[j]] = nV[perm[j]] + 1
            else:
                for l in range(No):
                    Y = y[:Ni]
                    y = y[Ni:]
                    cVar[l] = np.var(Y, ddof=1)
                Chat = np.mean(cVar)
                delta = Chat - prevC
            Sh[perm[j]] = Sh[perm[j]] + delta
            prevC = Chat
            if (j == 0):
                Tsob[perm[j]] = Tsob[perm[j]] + Chat # Total effect
                nT[perm[j]] = nT[perm[j]] + 1
    
    Sh = Sh / m / VarY
    if (method == 'exact'):
        Vsob = Vsob / (m/d) / VarY # averaging by number of permutations with j=d-1
        Vsob = 1 - Vsob 
        Tsob = Tsob / (m/d) / VarY # averaging by number of permutations with j=1 
    else:
        Vsob = Vsob / nV / VarY # averaging by number of permutations with j=d-1
        Vsob = 1 - Vsob 
        Tsob = Tsob / nT / VarY # averaging by number of permutations with j=1 
    
    col = ['X%d' % (i+1) for i in range(d)]
    effects = pd.DataFrame(np.array([Sh,Vsob,Tsob]), index = ['Shapley effects', 'First order Sobol', 'Total Sobol'], columns = col)

    return effects

def condMVN(mean, cov, dependent_ind, given_ind, X_given):
    """ Returns conditional mean and variance of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cov = np.array(cov)
    
    B = cov.take(dependent_ind, axis = 1)
    B = B[dependent_ind]
    
    C = cov.take(dependent_ind, axis = 1)
    C = C[given_ind]
    
    D = cov.take(given_ind, axis = 1)
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C),np.linalg.inv(D))
    
    condMean = mean[dependent_ind] + np.dot(CDinv,(X_given - mean[given_ind]))
    condVar = B - np.dot(CDinv,C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean, condVar

def r_condMVN(n, mean, cov, dependent_ind, given_ind, X_given):
    """ Function to simulate conditional gaussian distribution of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cond_mean,cond_var = condMVN(mean, cov, dependent_ind, given_ind, X_given)
    distribution = ot.Normal(cond_mean,cond_var)
    return distribution.getSample(n)