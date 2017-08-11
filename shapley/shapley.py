import numpy as np
import itertools
import openturns as ot
import pandas as pd

from .indices import Indices
from .kriging import KrigingIndices

def condMVN(mean, cov, dependent_ind, given_ind, X_given):
    """ Returns conditional mean and variance of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cov = np.array(cov)
    
    B = cov.take(dependent_ind, axis=1)
    B = B[dependent_ind]
    
    C = cov.take(dependent_ind, axis=1)
    C = C[given_ind]
    
    D = cov.take(given_ind, axis=1)
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C), np.linalg.inv(D))
    
    condMean = mean[dependent_ind] + np.dot(CDinv,(X_given - mean[given_ind]))
    condVar = B - np.dot(CDinv, C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean, condVar

def r_condMVN(n, mean, cov, dependent_ind, given_ind, X_given):
    """ Function to simulate conditional gaussian distribution of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cond_mean,cond_var = condMVN(mean, cov, dependent_ind, given_ind, X_given)
    distribution = ot.Normal(cond_mean,cond_var)
    return distribution.getSample(n)

class ShapleyIndices(Indices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        Indices.__init__(self, input_distribution)
        # Add shapley function
        self.indice_func = shapley_indices

    def build_mc_sample(self, model, n_perms, Nv=10**4, No=10**3, Ni=3, estimation_method='exact'):
        """
        """
        dim = self.dim
        if (estimation_method == 'exact'):
            perms = list(ot.KPermutations(dim, dim).generate())
            n_perms = len(perms)
        elif (estimation_method == 'random'):
            perms = [np.random.permutation(dim) for i in range(n_perms)]
        else:
            raise ValueError('Unknow permutation method {0}'.format(estimation_method))

        def Xcond(n, Sj, Sjc, xjc):
            cov = np.asarray(self.input_distribution.getCovariance())
            mean = np.asarray(self.input_distribution.getMean())
            if Sjc is None:
                cov_int = cov
                cov_int = cov_int.take(Sj, axis = 1)
                cov_int = cov_int[Sj, :]
                cov_int = ot.CovarianceMatrix(cov_int)
                distribution = ot.Normal(mean[Sj], cov_int)
                return np.asarray(distribution.getSample(n))
            else:
                return r_condMVN(n, mean=mean, cov=cov, dependent_ind=Sj, given_ind=Sjc, X_given=xjc)

        # Creation of the design matrix
        X = np.zeros((Nv + n_perms*(dim-1)*No*Ni, dim))
        X[:Nv, :] = self.input_distribution.getSample(Nv)

        for i_p, perm in enumerate(perms):
            idx_perm_sorted = np.argsort(perm)
            for j in range(dim-1):
                Sj = perm[:j+1] # set of the 0st-(j-1)th elements in perm
                Sjc = perm[j+1:] # set of the jth-dth elements in perm
                xjcM = Xcond(No, Sjc, None, None) # sampled values of the inputs in Sjc
                for l, xjc in enumerate(xjcM):
                    # sample values of inputs in Sj conditional on xjc
                    xj = Xcond(Ni, Sj, Sjc, xjc)
                    xx = np.c_[xj, [xjc]*Ni]
                    ind_inner = Nv + i_p*(dim-1)*No*Ni + j*No*Ni + l*Ni
                    X[ind_inner:ind_inner+Ni, :] = xx[:, idx_perm_sorted]

        self.output_sample = model(X)
        self.Nv = Nv
        self.No = No
        self.Ni = Ni
        self.perms = perms
        self.estimation_method = estimation_method

    def compute_indices(self):
        """
        """
        dim = self.dim
        Nv = self.Nv
        No = self.No
        Ni = self.Ni
        perms = self.perms
        estimation_method = self.estimation_method
        n_perms = len(perms)

        # Initialize Shapley, main and total Sobol effects for all players
        Sh = np.zeros(dim)
        Vsob = np.zeros(dim)
        Tsob = np.zeros(dim)
        nV = np.zeros(dim) # number of samples used to estimate V1,...,Vd
        nT = np.zeros(dim) # number of samples used to estimate T1,...,Td
    
        # Estimate Var[Y]
        Y = self.output_sample[:Nv]
        y = self.output_sample[Nv:]
        EY = np.mean(Y)
        VarY = np.var(Y, ddof=1)

        # Estimate Shapley, main and total Sobol effects
        cVar = np.zeros(No)

        for perm in perms:
            prevC = 0
            for j in range(dim):
                if (j == (dim-1)):
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
    
        Sh = Sh / n_perms / VarY
        if estimation_method == 'exact':
            Vsob = Vsob / (n_perms/dim) / VarY # averaging by number of permutations with j=d-1
            Vsob = 1 - Vsob 
            Tsob = Tsob / (n_perms/dim) / VarY # averaging by number of permutations with j=1 
        elif estimation_method == 'random':
            Vsob = Vsob / nV / VarY # averaging by number of permutations with j=d-1
            Vsob = 1 - Vsob 
            Tsob = Tsob / nT / VarY # averaging by number of permutations with j=1
    
        col = ['S%d' % (i+1) for i in range(dim)]
        effects = pd.DataFrame(np.array([Sh,Vsob,Tsob]), index = ['Shapley effects', 'First order Sobol', 'Total Sobol'], columns = col)

        return effects


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
    
    col = ['S%d' % (i+1) for i in range(d)]
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