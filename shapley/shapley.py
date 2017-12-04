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
    """Shappley indices estimator.
    
    Estimates the Shapley indices for sensitivity analysis of model output. The
    estimation algorithm is inspired from [1] and slightly modified to 
    implement a bootstrap strategy. The bootstrap can be made on the random 
    permutation or the exact ones.
    
    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        An OpenTURNS distribution object.
    
    References
    ----------
    -- [1] Song, Eunhye, Barry L. Nelson, and Jeremy Staum, Shapley effects for
        global sensitivity analysis
        http://users.iems.northwestern.edu/~staum/ShapleyEffects.pdf
    """
    def __init__(self, input_distribution):
        BaseIndices.__init__(self, input_distribution)
        self._built_samples = False

    def build_sample(self, model, n_perms, n_var, n_outer, n_inner, n_realization=1):
        """Creates the input and output sample for the computation.
        
        Using Algorithm described in [1], the input sample are generated using
        the input distribution and are evaluated through the input model.
        
        Parameters
        ----------
        model : callable
            The input model function.
        
        n_perms : int or None
            The number of permutations. If None, the exact permutations method
            is considerd.
            
        n_var : int
            The sample size for the output variance estimation.
            
        n_outer : int
            The number of conditionnal variance estimations.
            
        n_inner : int
            The sample size for the conditionnal output variance estimation.
            
        n_realization : int, optional (default=1)
            The number of realization if the model is a random meta-model.
        
        References
        ----------
        -- [1] Song, Eunhye, Barry L. Nelson, and Jeremy Staum, Shapley effects for
            global sensitivity analysis
            http://users.iems.northwestern.edu/~staum/ShapleyEffects.pdf
        """
        assert callable(model), "The model function should be callable."
        assert isinstance(n_perms, (int, type(None))), \
            "The number of permutation should be an integer or None."
        assert isinstance(n_var, int), "n_var should be an integer."
        assert isinstance(n_outer, int), "n_outer should be an integer."
        assert isinstance(n_inner, int), "n_inner should be an integer."
        assert isinstance(n_realization, int), \
            "n_realization should be an integer."
        if isinstance(n_perms, int):
            assert n_perms > 0, "The number of permutation should be positive"
            
        assert n_var > 0, "n_var should be positive"
        assert n_outer > 0, "n_outer should be positive"
        assert n_inner > 0, "n_inner should be positive"
        assert n_realization > 0, "n_realization should be positive"            
        
        dim = self.dim
        
        if n_perms is None:
            estimation_method = 'exact'
            perms = list(ot.KPermutations(dim, dim).generate())
            n_perms = len(perms)
        else:
            estimation_method = 'random'
            perms = [np.random.permutation(dim) for i in range(n_perms)]
        
        # Creation of the design matrix
        input_sample_1 = np.asarray(self.input_distribution.getSample(n_var))
        input_sample_2 = np.zeros((n_perms * (dim - 1) * n_outer * n_inner, dim))

        for i_p, perm in enumerate(perms):
            idx_perm_sorted = np.argsort(perm)  # Sort the variable ids
            for j in range(dim - 1):
                # Normal set
                idx_j = perm[:j + 1]
                # Complementary set
                idx_j_c = perm[j + 1:]
                sample_j_c = sub_sampling(self.input_distribution, n_outer, idx_j_c)
                self.sample_j_c = sample_j_c
                for l, xjc in enumerate(sample_j_c):
                    # Sampling of the set conditionally to the complementary
                    # element
                    xj = cond_sampling_new(self.input_distribution, n_inner, idx_j, idx_j_c, xjc)
                    xx = np.c_[xj, [xjc] * n_inner]
                    ind_inner = i_p * (dim - 1) * n_outer * n_inner + j * n_outer * n_inner + l * n_inner
                    input_sample_2[ind_inner:ind_inner + n_inner, :] = xx[:, idx_perm_sorted]

        # Model evaluation
        X = np.r_[input_sample_1, input_sample_2]

        self.X = X
        if n_realization == 1:
            output_sample = model(X)
        else:
            output_sample = model(X, n_realization)
                
        self.output_sample_1 = output_sample[:n_var]
        self.output_sample_2 = output_sample[n_var:]\
            .reshape((n_perms, dim-1, n_outer, n_inner, n_realization))
        
        self.model = model
        self.estimation_method = estimation_method
        self.perms = perms
        self.n_var = n_var
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.n_realization = n_realization
        self._built_samples = True

    def compute_indices(self, n_boot=1):
        """Computes the Shapley indices.
        
        The Shapley indices are computed from the computed samples. In addition
        to the Shapley indices, the first-order and total Sobol' indices are
        also computed.
        
        Parameters
        ----------
        n_boot : int
            The number of bootstrap samples.
            
        Returns
        -------
        indice_results : instance of SensitivityResults
            The sensitivity results of the estimation.
        
        """
        assert self._built_samples, "The samples must be computed prior."
        assert isinstance(n_boot, int), "n_boot should be an integer."
        assert n_boot > 0, "n_boot should be positive."
        
        dim = self.dim
        n_var = self.n_var
        n_outer = self.n_outer
        n_realization = self.n_realization
        estimation_method = self.estimation_method
        perms = self.perms
        n_perms = len(perms)

        # Initialize Shapley, main and total Sobol effects for all players
        shapley_indices = np.zeros((dim, n_boot, n_realization))
        first_indices = np.zeros((dim, n_boot, n_realization))
        total_indices = np.zeros((dim, n_boot, n_realization))
        shapley_indices_2 = np.zeros((dim, n_realization))
        first_indices_2 = np.zeros((dim, n_realization))
        total_indices_2 = np.zeros((dim, n_realization))

        n_first = np.zeros((dim, n_boot, n_realization))
        n_total = np.zeros((dim, n_boot, n_realization))
        c_hat = np.zeros((n_perms, dim, n_boot, n_realization))

        if estimation_method == 'random':
            boot_perms = np.zeros((n_perms, n_boot), dtype=int)
        
        # TODO: ugly... Do it better
        variance = np.zeros((n_boot, n_realization))
        perms = np.asarray(perms)

        for i in range(n_boot):
            # Bootstrap sample indexes
            # The first iteration is computed over the all sample.
            if i > 0:
                boot_var_idx = np.random.randint(0, n_var, size=(n_var, ))
                if estimation_method == 'exact':
                    boot_No_idx = np.random.randint(0, n_outer, size=(n_outer, ))
                else:
                    boot_n_perms_idx = np.random.randint(0, n_perms, size=(n_perms, ))
                    boot_perms[:, i] = boot_n_perms_idx
            else:
                boot_var_idx = range(n_var)
                if estimation_method == 'exact':
                    boot_No_idx = range(n_outer)
                else:
                    boot_n_perms_idx = range(n_perms)
                    boot_perms[:, i] = boot_n_perms_idx
                
            # Output variance
            var_y = self.output_sample_1[boot_var_idx].var(axis=0, ddof=1)

            variance[i] = var_y

            # Conditional variances
            if estimation_method == 'exact':
                output_sample_2 = self.output_sample_2[:, :, boot_No_idx]
            else:
                output_sample_2 = self.output_sample_2[boot_n_perms_idx]
            
            c_var = output_sample_2.var(axis=3, ddof=1)

            # Conditional exceptations
            c_mean_var = c_var.mean(axis=2)

            # Cost estimation
            c_hat[:, :, i] = np.concatenate((c_mean_var, [var_y.reshape(1, -1)]*n_perms), axis=1)

        # Cost variation
        delta_c = c_hat.copy()
        delta_c[:, 1:] = c_hat[:, 1:] - c_hat[:, :-1]
        
        for i in range(n_boot):
            if estimation_method == 'random':
                boot_n_perms_idx = boot_perms[:, i]
                tmp_perms = perms[boot_n_perms_idx]
            else:
                tmp_perms = perms

            # Estimate Shapley, main and total Sobol effects
            for i_p, perm in enumerate(tmp_perms):
                # Shapley effect
                shapley_indices[perm, i] += delta_c[i_p, :, i]
                shapley_indices_2[perm] += delta_c[i_p, :, 0]**2

                # Total effect
                total_indices[perm[0], i] += c_hat[i_p, 0, i]
                total_indices_2[perm[0]] += c_hat[i_p, 0, 0]**2
                n_total[perm[0], i] += 1

                # First order effect
                first_indices[perm[-1], i] += c_hat[i_p, -2, i]
                first_indices_2[perm[-1]] += c_hat[i_p, -2, 0]**2
                n_first[perm[-1], i] += 1
            
        N_total = n_perms / dim if estimation_method == 'exact' else n_total
        N_first = n_perms / dim if estimation_method == 'exact' else n_first
        
        N_total_2 = n_perms / dim if estimation_method == 'exact' else n_total[:, 0]
        N_first_2 = n_perms / dim if estimation_method == 'exact' else n_first[:, 0]
        
        output_variance = variance[np.newaxis]
        shapley_indices = shapley_indices / n_perms / output_variance
        total_indices = total_indices / N_total / output_variance
        vsob = first_indices / N_first / output_variance
        first_indices = 1. - vsob
        if estimation_method == 'random':
            output_variance_2 = output_variance[:, 0]
            shapley_indices_2 = shapley_indices_2 / n_perms / output_variance_2**2
            shapley_indices_SE = np.sqrt((shapley_indices_2 - shapley_indices[:, 0]**2) / n_perms)
            total_indices_2 = total_indices_2 / N_total_2 / output_variance_2**2
            total_indices_SE = np.sqrt((total_indices_2 - total_indices[:, 0]**2) / N_total_2)
            vsob_2 = first_indices_2 / N_first_2 / output_variance_2**2
            first_indices_SE = np.sqrt((vsob_2 - vsob[:, 0]**2) / N_first_2)
        else:
            shapley_indices_SE = None
            total_indices_SE = None
            first_indices_SE = None

        indice_results = SensitivityResults(
                first_indices=first_indices, 
                total_indices=total_indices,
                shapley_indices=shapley_indices,
                true_first_indices=self.model.first_sobol_indices,
                true_total_indices=self.model.total_sobol_indices,
                true_shapley_indices=self.model.shapley_indices,
                shapley_indices_SE=shapley_indices_SE,
                total_indices_SE=total_indices_SE,
                first_indices_SE=first_indices_SE,
                estimation_method=estimation_method)
        return indice_results