import numpy as np
import openturns as ot

from .indices import BaseIndices, SensitivityResults


def condMVN_new(cov, dependent_ind, given_ind, X_given):
    """ Returns conditional mean and variance of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cov = np.asarray(cov)
    
    B = cov[:, dependent_ind]
    B = B[dependent_ind]
    
    C = cov[:, dependent_ind]
    C = C[given_ind]
    
    D = cov[:, given_ind]
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C), np.linalg.inv(D))
    
    condMean = np.dot(CDinv, X_given)
    condVar = B - np.dot(CDinv, C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean, condVar


def condMVN(mean, cov, dependent_ind, given_ind, X_given):
    """ Returns conditional mean and variance of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cov = np.array(cov)
    
    B = cov[:, dependent_ind]
    B = B[dependent_ind]
    
    C = cov[:, dependent_ind]
    C = C[given_ind]
    
    D = cov[:, given_ind]
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C), np.linalg.inv(D))
    
    condMean = mean[dependent_ind] + np.dot(CDinv, (X_given - mean[given_ind]))
    condVar = B - np.dot(CDinv, C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean, condVar


def cond_sampling_new(distribution, n_sample, idx, idx_c, x_cond):
    """
    """
    margins_dep = [distribution.getMarginal(int(i)) for i in idx]
    margins_cond = [distribution.getMarginal(int(i)) for i in idx_c]

    # Creates a conditioned variables that follows a Normal distribution
    u_cond = np.zeros(x_cond.shape)
    for i, marginal in enumerate(margins_cond):
        u_cond[i] = np.asarray(ot.Normal().computeQuantile(marginal.computeCDF(x_cond[i])))

    sigma = np.asarray(distribution.getCopula().getCorrelation())
    cond_mean, cond_var = condMVN_new(sigma, idx, idx_c, u_cond)
    
    n_dep = len(idx)
    dist_cond = ot.Normal(cond_mean, cond_var)
    sample_norm = np.asarray(dist_cond.getSample(int(n_sample)))
    sample_x = np.zeros((n_sample, n_dep))
    phi = lambda x: ot.Normal().computeCDF(x)
    for i in range(n_dep):
        u_i = np.asarray(phi(sample_norm[:, i].reshape(-1, 1))).ravel()
        sample_x[:, i] = np.asarray(margins_dep[i].computeQuantile(u_i)).ravel()

    return sample_x


def cond_sampling(distribution, n_sample, idx, idx_c, x_cond):
    """
    """
    cov = np.asarray(distribution.getCovariance())
    mean = np.asarray(distribution.getMean())
    cond_mean, cond_var = condMVN(mean, cov, idx, idx_c, x_cond)
    dist_cond = ot.Normal(cond_mean, cond_var)
    sample = dist_cond.getSample(n_sample)
    return sample


def sub_sampling(distribution, n_sample, idx):
    """Sampling from a subset of a given distribution.

    The function takes the margin and correlation matrix subset and creates a new copula
    and distribution function to sample.

    Parameters
    ----------


    Returns
    -------
    sample : array,
        The sample of the subset distribution.
    """
    # Margins of the subset
    margins_sub = [distribution.getMarginal(int(j)) for j in idx]
    # Get the correlation matrix
    sigma = np.asarray(distribution.getCopula().getCorrelation())
    # Takes only the subset of the correlation matrix
    copula_sub = ot.NormalCopula(ot.CorrelationMatrix(sigma[:, idx][idx, :]))
    # Creates the subset distribution
    dist_sub = ot.ComposedDistribution(margins_sub, copula_sub)
    # Sample
    sample = np.asarray(dist_sub.getSample(int(n_sample)))
    return sample


class ShapleyIndices(BaseIndices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        BaseIndices.__init__(self, input_distribution)

    def build_sample(self, model, n_perms, Nv, No, Ni, n_realization=1):
        """
        """
        dim = self.dim
        if n_perms is None:
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

        for i_p, perm in enumerate(perms):
            idx_perm_sorted = np.argsort(perm)  # Sort the variable ids
            for j in range(dim - 1):
                # Normal set
                idx_j = perm[:j + 1]
                # Complementary set
                idx_j_c = perm[j + 1:]
                sample_j_c = sub_sampling(self.input_distribution, No, idx_j_c)
                self.sample_j_c = sample_j_c
                for l, xjc in enumerate(sample_j_c):
                    # Sampling of the set conditionally to the complementary
                    # element
                    xj = cond_sampling_new(self.input_distribution, Ni, idx_j, idx_j_c, xjc)
                    xx = np.c_[xj, [xjc] * Ni]
                    ind_inner = i_p * (dim - 1) * No * Ni + j * No * Ni + l * Ni
                    input_sample_2[ind_inner:ind_inner + Ni, :] = xx[:, idx_perm_sorted]

        # Model evaluation
        X = np.r_[input_sample_1, input_sample_2]

        self.X = X
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
        self.model = model

    def compute_indices(self, n_boot):
        """
        """
        dim = self.dim
        Nv = self.Nv
        No = self.No
        n_realization = self.n_realization
        perms = self.perms
        estimation_method = self.estimation_method
        n_perms = len(perms)

        # Initialize Shapley, main and total Sobol effects for all players
        shapley_indices = np.zeros((dim, n_boot, n_realization))
        first_indices = np.zeros((dim, n_boot, n_realization))
        total_indices = np.zeros((dim, n_boot, n_realization))
        n_first = np.zeros((dim, n_boot, n_realization))
        n_total = np.zeros((dim, n_boot, n_realization))
        c_hat = np.zeros((n_perms, dim, n_boot, n_realization))
        
        # TODO: ugly... Do it better
        variance = np.zeros((n_boot, n_realization))

        for i in range(n_boot):
            # Bootstrap sample indexes
            # The first iteration is computed over the all sample.
            if i > 1:
                boot_var_idx = np.random.randint(0, Nv, size=(Nv, ))
                boot_No_idx = np.random.randint(0, No, size=(No, ))
            else:
                boot_var_idx = range(Nv)
                boot_No_idx = range(No)
                
            # Output variance
            var_y = self.output_sample_1[boot_var_idx].var(axis=0, ddof=1)

            variance[i] = var_y

            # Conditional variances
            output_sample_2 = self.output_sample_2[:, :, boot_No_idx]
            
            c_var = output_sample_2.var(axis=3, ddof=1)

            # Conditional exceptations
            c_mean_var = c_var.mean(axis=2)

            # Cost estimation
            c_hat[:, :, i] = np.concatenate((c_mean_var, [var_y.reshape(1, -1)]*n_perms), axis=1)

        # Cost variation
        delta_c = c_hat.copy()
        delta_c[:, 1:] = c_hat[:, 1:] - c_hat[:, :-1]
        
        # Estimate Shapley, main and total Sobol effects
        for i_p, perm in enumerate(perms):
            # Shapley effect
            shapley_indices[perm] += delta_c[i_p]
            # Total effect
            total_indices[perm[0]] += c_hat[i_p, 0]
            n_total[perm[0]] += 1
            # First order effect
            first_indices[perm[-1]] += c_hat[i_p, -2]
            n_first[perm[-1]] += 1

        shapley_indices = shapley_indices / n_perms / variance.reshape(1, n_boot, n_realization)

        N_first = n_perms / dim if estimation_method == 'exact' else n_first
        N_total = n_perms / dim if estimation_method == 'exact' else n_total
        
        total_indices = total_indices / N_first / variance.reshape(1, n_boot, n_realization)
        first_indices = 1. - first_indices / N_total / variance.reshape(1, n_boot, n_realization)
        
        shapley_indices = shapley_indices.reshape(dim, n_boot, n_realization)
        total_indices = total_indices.reshape(dim, n_boot, n_realization)
        first_indices = first_indices.reshape(dim, n_boot, n_realization)
    
        results = SensitivityResults(
                first_indices=first_indices, 
                total_indices=total_indices,
                shapley_indices=shapley_indices,
                true_first_indices=self.model.first_sobol_indices,
                true_total_indices=self.model.total_sobol_indices,
                true_shapley_indices=self.model.shapley_indices)
        return results