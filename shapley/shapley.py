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
    
    cond_mean, cond_var = condMVN(mean, cov, dependent_ind, given_ind, X_given)
    distribution = ot.Normal(cond_mean, cond_var)
    return distribution.getSample(n)


def cond_sampling(dist, n_sample, idx, idx_c, x_cond):
    covariance = np.asarray(dist.getCovariance())
    mean = np.asarray(dist.getMean())
    return r_condMVN(n_sample, mean=mean, cov=covariance, dependent_ind=idx, given_ind=idx_c, X_given=x_cond)

def sub_sampling(dist, n_sample, idx):
    # Sampling of the complementary set
    covariance = np.asarray(dist.getCovariance())
    mean = np.asarray(dist.getMean())
    cov_sub = covariance.take(idx, axis=1)[idx, :]
    dist_sub = ot.Normal(mean[idx], ot.CovarianceMatrix(cov_sub))
    sample = np.asarray(dist_sub.getSample(n_sample))
    return sample

class ShapleyIndices(Indices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        Indices.__init__(self, input_distribution)
        # Add shapley function
        self.indice_func = shapley_indices

    def build_mc_sample(self, model, n_perms=3, Nv=10000, No=1000, Ni=3, n_realization=1):
        """
        """
        dim = self.dim
        if n_perms is None or n_perms > np.math.factorial(dim):
            estimation_method = 'exact'
            perms = list(ot.KPermutations(dim, dim).generate())
            n_perms = len(perms)
        elif n_perms > 0: 
            estimation_method = 'random'
            perms = [np.random.permutation(dim) for i in range(n_perms)]
        else:
            raise ValueError('Wrong value for n_perms: {0}'.format(n_perms))
        
        # Creation of the design matrix
        input_sample_1 = np.asarray(self.input_distribution.getSample(Nv))
        
        input_sample_2 = np.zeros((n_perms * (dim - 1) * No * Ni, dim))
        input_sample_3 = np.zeros((n_perms, dim - 1, No, Ni, dim))

        covariance = np.asarray(self.input_distribution.getCovariance())
        mean = np.asarray(self.input_distribution.getMean())
        for i_p, perm in enumerate(perms):
            idx_perm_sorted = np.argsort(perm)  # Sort the variable ids
            for j in range(dim - 1):
                # Normal set
                idx_j = perm[:j + 1]
                # Complementary set
                idx_j_c = perm[j + 1:]
                sample_j_c = sub_sampling(self.input_distribution, No, idx_j_c)
                for l, xjc in enumerate(sample_j_c):
                    # Sampling of the set conditionally to the complementary
                    # element
                    xj = cond_sampling(self.input_distribution, Ni, idx_j, idx_j_c, xjc)
                    xx = np.c_[xj, [xjc] * Ni]
                    ind_inner = i_p * (dim - 1) * No * Ni + j * No * Ni + l * Ni
                    input_sample_2[ind_inner:ind_inner + Ni, :] = xx[:, idx_perm_sorted]
                    input_sample_3[i_p, j, l] = xx[:, idx_perm_sorted]

        # Model evaluation
        X = np.r_[input_sample_1, input_sample_2]

        if n_realization == 1:
            output_sample = model(X)
        else:
            output_sample = model(X, n_realization)

        self.output_sample_1 = output_sample[:Nv]
        self.output_sample_2 = output_sample[Nv:]
        self.perms = perms
        self.estimation_method = estimation_method
        self.Nv = Nv
        self.No = No
        self.Ni = Ni
        self.n_realization = n_realization

    def compute_indices(self, n_boot):
        """
        """
        dim = self.dim
        Nv = self.Nv
        No = self.No
        Ni = self.Ni
        n_realization = self.n_realization
        perms = self.perms
        estimation_method = self.estimation_method
        n_perms = len(perms)

        # Initialize Shapley, main and total Sobol effects for all players
        Sh = np.zeros((dim, n_realization))
        Vsob = np.zeros((dim, n_realization))
        Tsob = np.zeros((dim, n_realization))
        nV = np.zeros((dim, n_realization)) # number of samples used to estimate V1,...,Vd
        nT = np.zeros((dim, n_realization)) # number of samples used to estimate T1,...,Td
    
        # Estimate Var[Y]
        var_y = np.var(self.output_sample_1, axis=0, ddof=1)

        # Estimate Shapley, main and total Sobol effects
        for i_nz in range(n_realization):
            if n_realization == 1:
                y = self.output_sample_2
            else:
                y = self.output_sample_2[:, i_nz]
            cVar = np.zeros((No, ))
            for perm in perms:
                prevC = 0
                for j in range(dim):
                    if j == dim - 1:
                        Chat = var_y[i_nz] if n_realization > 1 else var_y
                        delta = Chat - prevC
                        Vsob[perm[j], i_nz] = Vsob[perm[j], i_nz] + prevC # first order effect
                        nV[perm[j], i_nz] = nV[perm[j], i_nz] + 1
                    else:
                        for l in range(No):
                            y_l = y[:Ni]
                            cVar[l] = np.var(y_l, ddof=1)
                            y = y[Ni:]
                        Chat = np.mean(cVar)
                        delta = Chat - prevC
                    Sh[perm[j], i_nz] = Sh[perm[j], i_nz] + delta
                    prevC = Chat
                    if j == 0:
                        Tsob[perm[j], i_nz] = Tsob[perm[j], i_nz] + Chat # Total effect
                        nT[perm[j], i_nz] = nT[perm[j], i_nz] + 1
    
        Sh = Sh / n_perms / var_y
        if (estimation_method == 'exact'):
            Vsob = Vsob / (n_perms / dim) / var_y # averaging by number of permutations with j=d-1
            Vsob = 1 - Vsob 
            Tsob = Tsob / (n_perms / dim) / var_y # averaging by number of permutations with j=1
        else:
            Vsob = Vsob / nV / var_y # averaging by number of permutations with j=d-1
            Vsob = 1 - Vsob 
            Tsob = Tsob / nT / var_y # averaging by number of permutations with j=1
    
        col = ['S%d' % (i + 1) for i in range(dim)]
        #effects = pd.DataFrame(np.array([Sh,Vsob,Tsob]), index = ['Shapley effects', 'First order Sobol', 'Total Sobol'], columns = col)

        return Sh, Vsob, Tsob


class ShapleyKrigingIndices(KrigingIndices, ShapleyIndices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        KrigingIndices.__init__(self, input_distribution)
        ShapleyIndices.__init__(self, input_distribution)


def shapley_indices(method, n_perms, model, input_distribution, dim, Nv, No, Ni=3):
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
        perms = list(ot.KPermutations(dim, dim).generate())
        n_perms = len(perms) # number of permutation
    elif (method == 'random'):
        perms = [np.random.permutation(dim) for i in range(n_perms)]
    else:
        raise ValueError('Unknow permutation method {0}'.format(method))

    def Xcond(n, Sj, Sjc, xjc):
        cov = np.asarray(input_distribution.getCovariance())
        mean = np.asarray(input_distribution.getMean())
        if Sjc is None:
            cov_int = cov.take(Sj, axis = 1)
            cov_int = cov_int[Sj, :]
            cov_int = ot.CovarianceMatrix(cov_int)
            distribution = ot.Normal(mean[Sj], cov_int)
            return np.asarray(distribution.getSample(n))
        else:
            return r_condMVN(n, mean=mean, cov=cov, dependent_ind=Sj, given_ind=Sjc, X_given=xjc)
    
    # Creation of the design matrix
    input_sample_1 = np.asarray(input_distribution.getSample(Nv))

    input_sample_2 = np.zeros((n_perms * (dim - 1) * No * Ni, dim))
    for i_p, perm in enumerate(perms):
        idx_perm_sorted = np.argsort(perm)  # Sort the variable ids
        for j in range(dim - 1):
            # Normal set
            idx_j = perm[:j + 1]
            # Complementary set
            idx_j_c = perm[j + 1:]
            # Sampling of the complementary set
            sample_j_c = Xcond(No, idx_j_c, None, None)
            for l, xjc in enumerate(sample_j_c):
                # Sampling of the set conditionally to the complementary
                # element
                xj = Xcond(Ni, idx_j, idx_j_c, xjc)
                xx = np.c_[xj, [xjc] * Ni]
                ind_inner = i_p * (dim - 1) * No * Ni + j * No * Ni + l * Ni
                input_sample_2[ind_inner:ind_inner + Ni, :] = xx[:, idx_perm_sorted]

    # Model evaluation
    output_sample_1 = model(input_sample_1)
    output_sample_2 = model(input_sample_2)

    # Initialize Shapley, main and total Sobol effects for all players
    Sh = np.zeros(dim)
    Vsob = np.zeros(dim)
    Tsob = np.zeros(dim)
    nV = np.zeros(dim) # number of samples used to estimate V1,...,Vd
    nT = np.zeros(dim) # number of samples used to estimate T1,...,Td
    
    # Estimate Var[Y]
    var_y = np.var(output_sample_1, ddof=1)
    y = output_sample_2

    # Estimate Shapley, main and total Sobol effects
    cVar = np.zeros(No)
    for perm in perms:
        prevC = 0
        for j in range(dim):
            if j == (dim - 1):
                Chat = var_y
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
    
    Sh = Sh / n_perms / var_y
    if (method == 'exact'):
        Vsob = Vsob / (n_perms / dim) / var_y # averaging by number of permutations with j=d-1
        Vsob = 1 - Vsob 
        Tsob = Tsob / (n_perms / dim) / var_y # averaging by number of permutations with j=1
    else:
        Vsob = Vsob / nV / var_y # averaging by number of permutations with j=d-1
        Vsob = 1 - Vsob 
        Tsob = Tsob / nT / var_y # averaging by number of permutations with j=1
    
    col = ['S%d' % (i + 1) for i in range(dim)]
    effects = pd.DataFrame(np.array([Sh,Vsob,Tsob]), index = ['Shapley effects', 'First order Sobol', 'Total Sobol'], columns = col)

    return effects