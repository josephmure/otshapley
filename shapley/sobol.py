import numpy as np
import openturns as ot

from numpy.random import randint

from .indices import BaseIndices, SensitivityResults


class SobolIndices(BaseIndices):
    """The Sobol indices.

    Estimate with a Monte-Carlo sampling, the first-order and total sobol
    indices. The classical method in addition to the uncorrelated sampling 
    using the Rosenblatt Transformation are implemented.

    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        An OpenTURNS distribution object.
    """

    def __init__(self, input_distribution):
        BaseIndices.__init__(self, input_distribution)
        self.indice_func = sobol_indices

    # TODO: gather the two function and add an option for the
    def build_sample(self, model, n_sample, n_realization=1):
        """Creates the Monte-Carlo samples for independent variables.

        A pick and freeze strategy is done considering the distribution of the 
        input sample. This method creates the input samples and evaluate them 
        through the model to create the output sample. Note that the variables
        should be considered independent.

        Parameters
        ----------
        model : callable
            The model function.

        n_sample : int
            The sampling size of the Monte-Carlo estimation.

        n_realization : int, optional (default=1)
            The number of realization of the meta-model.
        """

        assert callable(model), "The model should be a function"
        assert isinstance(n_sample, int), \
            "The number of sample should be an integer"
        assert isinstance(n_realization, int), \
            "The number of realization should be an integer"
        assert n_sample > 0, \
            "The number of sample should be positive: %d<0" % (n_sample)
        assert n_realization > 0, \
            "The number of realization should be positive: %d<0" % (
                n_realization)
        dim = self.dim

        # Simulate the two independent samples
        input_sample_1 = np.asarray(
            self._input_distribution.getSample(n_sample))
        input_sample_2 = np.asarray(
            self._input_distribution.getSample(n_sample))

        # The modified samples for each dimension
        all_output_sample_2t = np.zeros((dim, n_sample, n_realization))
        if n_realization == 1:
            output_sample_1 = model(input_sample_1)
            output_sample_2 = model(input_sample_2)
            output_sample_1 = np.c_[[output_sample_1]
                                    * dim].reshape(dim, n_sample, n_realization)
            output_sample_2 = np.c_[[output_sample_2]
                                    * dim].reshape(dim, n_sample, n_realization)
        else:
            output_sample_1 = np.zeros((dim, n_sample, n_realization))
            output_sample_2 = np.zeros((dim, n_sample, n_realization))

        X1 = input_sample_1
        X2 = input_sample_2
        for i in range(dim):
            X2t = X2.copy()
            X2t[:, i] = X1[:, i]

            if n_realization == 1:
                all_output_sample_2t[i] = model(
                    X2t).reshape(n_sample, n_realization)
            else:
                # model : function two parameters?
                output_sample_i = model(np.r_[X1, X2, X2t], n_realization)
                output_sample_1[i] = output_sample_i[:n_sample, :]
                output_sample_2[i] = output_sample_i[n_sample:2*n_sample, :]
                all_output_sample_2t[i] = output_sample_i[2*n_sample:, :]

        self.all_output_sample_1 = output_sample_1
        self.all_output_sample_2 = output_sample_2
        self.all_output_sample_2t = all_output_sample_2t
        self.n_sample = n_sample
        self.n_realization = n_realization
        self.model = model

    def build_uncorr_sample(self, model, n_sample, n_realization=1):
        """Creates the Monte-Carlo samples for correlated variables.

        A pick and freeze strategy is done considering the distribution of the 
        input sample. This method creates the input samples and evaluate them 
        through the model to create the output sample. Note that the variables
        should be considered independent.

        Parameters
        ----------
        model : callable
            The model function.

        n_sample : int
            The sampling size of the Monte-Carlo estimation.

        n_realization : int, optional (default=1)
            The number of realization of the meta-model.  

        References
        ----------
        .. [1] Thierry A Mara, Stefano Tarantola, Paola Annoni, Non-parametric
            methods for global sensitivity analysis of model output with dependent inputs
            https://hal.archives-ouvertes.fr/hal-01182302/file/Mara15EMS_HAL.pdf
        """
        assert callable(model), "The model should be a function"
        assert isinstance(n_sample, int), \
            "The number of sample should be an integer"
        assert isinstance(n_realization, int), \
            "The number of realization should be an integer"
        assert n_sample > 0, \
            "The number of sample should be positive: %d<0" % (n_sample)
        assert n_realization > 0, \
            "The number of realization should be positive: %d<0" % (
                n_realization)
        dim = self.dim
        n_pairs = int(dim*(dim-1) / 2)

        # Gaussian distribution
        norm_dist = ot.Normal(dim)

        # Independent samples
        U_1 = np.asarray(norm_dist.getSample(n_sample))
        U_2 = np.asarray(norm_dist.getSample(n_sample))

        all_output_sample_1 = np.zeros((dim, n_sample, n_realization))
        all_output_sample_2 = np.zeros((dim, n_sample, n_realization))
        all_output_sample_2t = np.zeros((dim, n_sample, n_realization))
        all_output_sample_2t1 = np.zeros((dim, n_sample, n_realization))

        for i in range(dim):
            # Copy of the input dstribution
            margins = [ot.Distribution(
                self._input_distribution.getMarginal(j)) for j in range(dim)]
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

            def inv_rosenblatt_transform_i(u): return np.asarray(tmp(u))

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
            self.debug_X = X
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

    def compute_indices(self, n_boot=500, estimator='soboleff2', indice_type='classic'):
        """Computes the Sobol' indices with the pick and freeze strategy.

        The method computes the indices from the previously created samples.

        Parameters
        ----------            
        n_boot : int, optional (default=500)
            The bootstrap sample size.

        estimator : str, optional (default='soboleff2')
            The estimator method for the pick and freeze strategy. Available
            estimators :

            - 'sobol': initial pick and freeze from [1],
            - 'sobol2002': from [2],
            - 'sobol2007': from [3],
            - 'soboleff1': first estimator of [4],
            - 'soboleff2': second estimator of [4],
            - 'sobolmara': from [5],

        indice_type : str, optional (default='classic')
            The type of indices to compute. It can be:

            - 'classic': the pick and freeze for independent variables,
            - 'ind': the independent Sobol' indices,
            - 'full': the full Sobol' indices.

        Returns
        -------
        results : SensitivityResults instance
            The computed Sobol' indices.

        References
        ----------
        .. [1] Sobol 93
        .. [2] Saltelli & al. 2002
        .. [3] Sobol 2007
        .. [4] Janon
        .. [5] TODO: check the sources
        """
        assert isinstance(n_boot, int), \
            "The number of bootstrap should be an integer"
        assert isinstance(estimator, str), \
            "The estimator name should be an string"
        assert isinstance(indice_type, str), \
            "The type of indices name should be an string"
        assert n_boot > 0, \
            "The number of boostrap should be positive: %d<0" % (n_boot)
        assert estimator in _ESTIMATORS, "Unknow estimator %s" % (estimator)
        assert indice_type in _DELTA_INDICES, \
            "Unknow of indice: %s" % (indice_type)

        dim = self.dim
        n_sample = self.n_sample
        n_realization = self.n_realization

        first_indices = np.zeros((dim, n_boot, n_realization))
        total_indices = np.zeros((dim, n_boot, n_realization))

        dev = _DELTA_INDICES[indice_type]
        if indice_type in ['classic', 'full']:
            sample_Y2t = self.all_output_sample_2t
        elif indice_type == 'ind':
            sample_Y2t = self.all_output_sample_2t1
        else:
            raise ValueError('Unknow type of indice: {0}'.format(indice_type))

        # TODO: cythonize this, takes too much memory when n_boot is large
        boot_idx = None
        for i in range(dim):
            if n_boot > 1:
                boot_idx = np.zeros((n_boot, n_sample), dtype=int)
                boot_idx[0] = range(n_sample)
                boot_idx[1:] = randint(
                    0, n_sample, size=(n_boot-1, n_sample))

            Y1 = self.all_output_sample_1[i]
            Y2 = self.all_output_sample_2[i]
            Y2t = sample_Y2t[i]
            first, total = self.indice_func(
                Y1, Y2, Y2t, boot_idx=boot_idx, estimator=estimator)
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


# TODO: cythonize this, it takes too much memory in vectorial
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
    first_indice, total_indice = estimator(
        Y1[boot_idx], Y2[boot_idx], Y2t[boot_idx])

    return first_indice, total_indice


def m(x): return x.mean(axis=1)


def s(x): return x.sum(axis=1)


def v(x): return x.var(axis=1)


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
    total_indice = 1. - var_total / var

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
    total_indice = 1. - var_total / var

    return first_indice, total_indice


def soboleff2_estimator(Y1, Y2, Y2t):
    """
    """
    mean2 = m((Y1 + Y2t)/2.)**2
    var = m((Y1**2 + Y2t**2)/2.) - m((Y1 + Y2t)/2.)**2

    var_indiv = m(Y2t * Y1) - mean2
    var_total = m(Y2t * Y2) - mean2

    first_indice = var_indiv / var
    total_indice = 1. - var_total / var

    return first_indice, total_indice


def sobolmara_estimator(Y1, Y2, Y2t):
    """
    """
    if True:
        diff = Y2t - Y2
        var = v(Y1)

        var_indiv = m(Y1 * diff)
        var_total = m(diff ** 2)

        first_indice = var_indiv / var
        total_indice = var_total / var / 2.
    else:
        n_sample, n_boot, n_realization = Y2.shape
        first_indice = np.zeros((n_boot, n_realization))
        total_indice = np.zeros((n_boot, n_realization))
        for i in range(n_realization):
            diff = Y2t[:, :, i] - Y2[:, :, i]
            var = v(Y1[:, :, i])

            var_indiv = m(Y1[:, :, i] * diff)
            var_total = m(diff ** 2)

            first_indice[:, i] = var_indiv / var
            total_indice[:, i] = var_total / var / 2.

    return first_indice, total_indice


_ESTIMATORS = {
    'sobol': sobol_estimator,
    'sobol2002': sobol2002_estimator,
    'sobol2007': sobol2007_estimator,
    'soboleff1': soboleff1_estimator,
    'soboleff2': soboleff2_estimator,
    'sobolmara': sobolmara_estimator
}

_DELTA_INDICES = {
    'classic': 0,
    'full': 0,
    'ind': 1,
}
