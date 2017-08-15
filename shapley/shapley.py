import numpy as np
import itertools
import openturns as ot
import pandas as pd

from .base import Base
from .kriging import KrigingIndices
from .base import SensitivityResults

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
    """
    """
    covariance = np.asarray(dist.getCovariance())
    mean = np.asarray(dist.getMean())
    return r_condMVN(n_sample, mean=mean, cov=covariance, dependent_ind=idx, given_ind=idx_c, X_given=x_cond)

def sub_sampling(dist, n_sample, idx):
    """Sampling of a given set
    """
    covariance = np.asarray(dist.getCovariance())
    mean = np.asarray(dist.getMean())
    cov_sub = covariance.take(idx, axis=1)[idx, :]
    dist_sub = ot.Normal(mean[idx], ot.CovarianceMatrix(cov_sub))
    sample = np.asarray(dist_sub.getSample(n_sample))
    return sample

class ShapleyIndices(Base):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        Base.__init__(self, input_distribution)

    def build_mc_sample(self, model, n_perms=3, Nv=10000, No=1000, Ni=3):
        """
        """
        return self._build_mc_sample(model, n_perms, Nv, No, Ni, n_realization=1)

    def _build_mc_sample(self, model, n_perms, Nv, No, Ni, n_realization):
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
        self.output_sample_2 = output_sample[Nv:].reshape((n_perms, dim-1, No, Ni, n_realization))
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
        shapley_indices = np.zeros((dim, n_boot, n_realization))
        first_indices = np.zeros((dim, n_boot, n_realization))
        total_indices = np.zeros((dim, n_boot, n_realization))
        n_sob = np.zeros((dim, n_boot, n_realization))
        c_hat = np.zeros((n_perms, dim, n_boot, n_realization))

        variances = np.zeros((n_boot, ))
        for i in range(n_boot):
            # Bootstrap sample indexes
            if n_boot > 1:
                boot_var_idx = np.random.randint(0, Nv, size=(Nv, ))
                boot_Ni_idx = np.random.randint(0, Ni, size=(Ni, ))
                boot_No_idx = np.random.randint(0, No, size=(No, ))
            else:
                boot_var_idx = range(0, Nv)
                boot_Ni_idx = range(0, Ni)
                boot_No_idx = range(0, No)

            # Output variance
            var_y = self.output_sample_1[boot_var_idx].var(axis=0, ddof=1)
            variances[i] = var_y
            # Conditional variances
            output_sample_2 = self.output_sample_2[:, :, :, boot_Ni_idx]
            output_sample_2 = output_sample_2[:, :, boot_No_idx]
            c_var = output_sample_2.var(axis=3, ddof=1)

            # Conditional exceptations
            c_mean_var = c_var.mean(axis=2)

            c_hat[:, :, i] = np.concatenate((c_mean_var, [var_y.reshape(1, -1)]*n_perms), axis=1)

        # Cost variation
        delta_c = c_hat.copy()
        delta_c[:, 1:] = c_hat[:, 1:] - c_hat[:, :-1]
        
        # Estimate Shapley, main and total Sobol effects
        for i_p, perm in enumerate(perms):
            shapley_indices[perm] += delta_c[i_p] # Shapley effect
            total_indices[perm[0]] += c_hat[i_p, 0] # Total effect
            first_indices[perm[-1]] += c_hat[i_p, -2] # first order effect
            n_sob[perm] += 1

        N = n_perms / dim if estimation_method == 'exact' else n_sob

        shapley_indices = shapley_indices / n_perms / variances.reshape(1, n_boot, n_realization)
        total_indices = total_indices / N / variances.reshape(1, n_boot, n_realization)
        first_indices = 1. - first_indices / N / variances.reshape(1, n_boot, n_realization)
        
        shapley_indices = shapley_indices.reshape(dim, n_boot, n_realization)
        total_indices = total_indices.reshape(dim, n_boot, n_realization)
        first_indices = first_indices.reshape(dim, n_boot, n_realization)
    
        results = SensitivityResults(first_indices=first_indices, total_indices=total_indices,
                                     shapley_indices=shapley_indices)
        return results


class ShapleyKrigingIndices(KrigingIndices, ShapleyIndices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        KrigingIndices.__init__(self, input_distribution)
        ShapleyIndices.__init__(self, input_distribution)

        
    def build_mc_sample(self, model, n_perms=3, Nv=10000, No=1000, Ni=3, n_realization=10):
        """
        """
        return self._build_mc_sample(model, n_perms=3, Nv=10000, No=1000, Ni=3, n_realization=10)