import numpy as np

from .indices import BaseIndices, SensitivityResults


class SobolIndices(BaseIndices):
    """The class of Sobol indices.

    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        And OpenTURNS distribution object.
    """
    def __init__(self, input_distribution):
        BaseIndices.__init__(self, input_distribution)
        self.indice_func = sobol_indices

    # TODO: gather the two function and add an option for the 
    def build_sample(self, model, n_sample, n_realization=1):
        """Build the Monte-Carlo samples.

        Parameters
        ----------
        model : callable,
            The model function.
        n_sample : int,
            The sampling size of Monte-Carlo
        n_realization : int,
            The number of Gaussian Process realizations.
        """
        dim = self.dim
        
        # Simulate the two independent samples
        input_sample_1 = np.asarray(self._input_distribution.getSample(n_sample))
        input_sample_2 = np.asarray(self._input_distribution.getSample(n_sample))
        
        # The modified samples for each dimension
        
        all_output_sample_2t = np.zeros((dim, n_sample, n_realization))
        if n_realization == 1:
            output_sample_1 = model(input_sample_1)
            output_sample_2 = model(input_sample_2)
            output_sample_1 = np.c_[[output_sample_1]*dim].reshape(dim, n_sample, n_realization)
            output_sample_2 = np.c_[[output_sample_2]*dim].reshape(dim, n_sample, n_realization)
        else:
            output_sample_1 = np.zeros((dim, n_sample, n_realization))
            output_sample_2 = np.zeros((dim, n_sample, n_realization))

        X1 = input_sample_1
        X2 = input_sample_2
        for i in range(dim):
            X2t = X2.copy()
            X2t[:, i] = X1[:, i]

            if n_realization == 1:
                all_output_sample_2t[i] = model(X2t).reshape(n_sample, n_realization)
            else:
                output_sample_i = model(np.r_[X1, X2, X2t], n_realization)          ##model : function two parameters?
                output_sample_1[i] = output_sample_i[:n_sample, :]
                output_sample_2[i] = output_sample_i[n_sample:2*n_sample, :]
                all_output_sample_2t[i] = output_sample_i[2*n_sample:, :]
            
        self.all_output_sample_1 = output_sample_1
        self.all_output_sample_2 = output_sample_2
        self.all_output_sample_2t = all_output_sample_2t
        self.n_sample = n_sample
        self.n_realization = n_realization
        self.model = model
    
    def build_uncorr_sample(self, model, n_sample, n_realization):
        """         ## add some comment here too
        """
        dim = self.dim

        # Normal distribution
        norm_dist = ot.Normal(dim)

        # Independent samples
        U_1 = np.asarray(norm_dist.getSample(n_sample))
        U_2 = np.asarray(norm_dist.getSample(n_sample))

        all_output_sample_1 = np.zeros((dim, n_sample, n_realization))
        all_output_sample_2 = np.zeros((dim, n_sample, n_realization))
        all_output_sample_2t = np.zeros((dim, n_sample, n_realization))
        all_output_sample_2t1 = np.zeros((dim, n_sample, n_realization))
        
        n_pairs = int(dim*(dim-1) / 2)
        for i in range(dim):
            # Copy of the input dstribution
            margins = [ot.Distribution(self._input_distribution.getMarginal(j)) for j in range(dim)]
            copula = ot.Copula(self._input_distribution.getCopula())

            # 1) Pick and Freeze
            U_3_i = U_2.copy()
            U_3_i[:, 0] = U_1[:, 0]
            U_4_i = U_2.copy()
            U_4_i[:, -1] = U_1[:, -1]
            
            # 2) Permute the margins and the copula
            order_i = np.roll(range(dim), -i)
            order_i_inv = np.roll(range(dim), i)
            order_cop = np.roll(range(n_pairs), i)
            margins_i = [margins[j] for j in order_i]
            params_i = np.asarray(copula.getParameter())[order_cop]

            copula.setParameter(params_i)
            dist = ot.ComposedDistribution(margins_i, copula)

            # 3) Inverse Transformation
            tmp = dist.getInverseIsoProbabilisticTransformation()
            inv_rosenblatt_transform_i = lambda u: np.asarray(tmp(u))

            X_1_i = inv_rosenblatt_transform_i(U_1)
            X_2_i = inv_rosenblatt_transform_i(U_2)
            X_3_i = inv_rosenblatt_transform_i(U_3_i)
            X_4_i = inv_rosenblatt_transform_i(U_4_i)
            assert X_1_i.shape[1] == dim, "Wrong dimension"

            X_1_i = X_1_i[:, order_i_inv]
            X_2_i = X_2_i[:, order_i_inv]
            X_3_i = X_3_i[:, order_i_inv]
            X_4_i = X_4_i[:, order_i_inv]
            
            # 4) Model evaluations
            X = np.r_[X_1_i, X_2_i, X_3_i, X_4_i]
            if n_realization == 1:
                output_sample_i = model(X).reshape(4*n_sample, n_realization)
            else:
                output_sample_i = model(X, n_realization)

            all_output_sample_1[i] = output_sample_i[:n_sample]
            all_output_sample_2[i] = output_sample_i[n_sample:2*n_sample]
            all_output_sample_2t[i] = output_sample_i[2*n_sample:3*n_sample]
            all_output_sample_2t1[i] = output_sample_i[3*n_sample:]

        self.all_output_sample_1 = all_output_sample_1
        self.all_output_sample_2 = all_output_sample_2
        self.all_output_sample_2t = all_output_sample_2t
        self.all_output_sample_2t1 = all_output_sample_2t1
        self.n_sample = n_sample
        self.n_realization = n_realization
        self.model = model

    def compute_indices(self, n_boot, estimator, indice_type='classic'):
        """
        """
        dim = self.dim
        n_sample = self.n_sample
        n_realization = self.n_realization

        first_indices = np.zeros((dim, n_boot, n_realization))
        total_indices = np.zeros((dim, n_boot, n_realization))

        if indice_type in ['classic', 'full']:
            dev = 0
            sample_Y2t = self.all_output_sample_2t
        elif indice_type == 'ind':
            dev = 1
            sample_Y2t = self.all_output_sample_2t1
        else:
            raise ValueError('Unknow type of indice {0}'.format(type))

        # TODO: cythonize this, takes too much memory when n_boot is large
        boot_idx = None
        for i in range(dim):
            if n_boot > 1:
                boot_idx = np.zeros((n_boot, n_sample), dtype=int)
                boot_idx[0] = range(n_sample)
                boot_idx[1:] = np.random.randint(0, n_sample, size=(n_boot-1, n_sample))

            Y1 = self.all_output_sample_1[i]
            Y2 = self.all_output_sample_2[i]
            Y2t = sample_Y2t[i]
            first, total = self.indice_func(Y1, Y2, Y2t, boot_idx=boot_idx, estimator=estimator)
            if first is not None:
                first = first.reshape(n_boot, n_realization)
            if total is not None:
                total = total.reshape(n_boot, n_realization)

            first_indices[i-dev], total_indices[i-dev] = first, total

        if np.isnan(total_indices).all():
            total_indices = None

        results = SensitivityResults(
            first_indices=first_indices,
            total_indices=total_indices,
            true_first_indices=self.model.first_sobol_indices,
            true_total_indices=self.model.total_sobol_indices,
            true_shapley_indices=self.model.shapley_indices)
        return results


def sobol_indices(Y1, Y2, Y2t, boot_idx=None, estimator='sobol2002'):
    """Compute the Sobol indices from the to

    Parameters
    ----------
    Y1 : array,
        The 

    Returns
    -------
    first_indice : int or array,
        The first order sobol indice estimation.

    total_indice : int or array,
        The total sobol indice estimation.
    """
    n_sample = Y1.shape[0]
    assert n_sample == Y2.shape[0], "Matrices should have the same sizes"
    assert n_sample == Y2t.shape[0], "Matrices should have the same sizes"
    assert estimator in _ESTIMATORS, 'Unknow estimator {0}'.format(estimator)

    estimator = _ESTIMATORS[estimator]

    # When boot_idx is None, it reshapes the Y as (1, -1).
    first_indice, total_indice = estimator(Y1[boot_idx], Y2[boot_idx], Y2t[boot_idx])

    return first_indice, total_indice


m = lambda x : x.mean(axis=1)
s = lambda x : x.sum(axis=1)
v = lambda x : x.var(axis=1)


def sobol_estimator(Y1, Y2, Y2t):
    """
    """
    mean2 = m(Y1)**2
    var = v(Y1)

    var_indiv = m(Y2t * Y1) - mean2
    first_indice = var_indiv / var
    total_indice = None

    return first_indice, total_indice


def sobol2002_estimator(Y1, Y2, Y2t):
    """
    """
    n_sample = Y1.shape[1]
    mean2 = s(Y1*Y2)/(n_sample - 1)
    var = v(Y1)

    var_indiv = s(Y2t * Y1)/(n_sample - 1) - mean2
    var_total = s(Y2t * Y2)/(n_sample - 1) - mean2
    first_indice = var_indiv / var
    total_indice = 1. - var_total  / var

    return first_indice, total_indice


def sobol2007_estimator(Y1, Y2, Y2t):
    """
    """
    var = v(Y1)

    var_indiv = m((Y2t - Y2) * Y1)
    var_total = m((Y2t - Y1) * Y2)
    first_indice = var_indiv / var
    total_indice = 1. - var_total / var

    return first_indice, total_indice


def soboleff1_estimator(Y1, Y2, Y2t):
    """
    """
    mean2 = m(Y1) * m(Y2t)
    var = m(Y1**2) - m(Y1)**2

    var_indiv = m(Y2t * Y1) - mean2
    var_total = m(Y2t * Y2) - mean2

    first_indice = var_indiv / var
    total_indice = 1. - var_total  / var

    return first_indice, total_indice


def soboleff2_estimator(Y1, Y2, Y2t):
    """
    """
    mean2 = m((Y1 + Y2t)/2.)**2
    var = m((Y1**2 + Y2t**2 )/2.) - m((Y1 + Y2t)/2.)**2

    var_indiv = m(Y2t * Y1) - mean2
    var_total = m(Y2t * Y2) - mean2

    first_indice = var_indiv / var
    total_indice = 1. - var_total  / var

    return first_indice, total_indice


def sobolmara_estimator(Y1, Y2, Y2t):
    """
    """
    diff = Y2t - Y2
    var = v(Y1)


    var_indiv = m(Y1 * diff)
    var_total = m(diff ** 2)

    first_indice = var_indiv / var
    total_indice = var_total / var / 2.

    return first_indice, total_indice


_ESTIMATORS = {
    'sobol': sobol_estimator,
    'sobol2002': sobol2002_estimator,
    'sobol2007': sobol2007_estimator,
    'soboleff1': soboleff1_estimator,
    'soboleff2': soboleff2_estimator,
    'sobolmara': sobolmara_estimator,
    }