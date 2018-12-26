import numpy as np
import openturns as ot

from shapley.model import ProbabilisticModel
from shapley.tests.utils import get_id


def is_independent(dist):
    """Check if the distribution has independent inputs.

    Parameters
    ----------
    dist : ot.Distribution,
        An multivariate OpenTURNS distribution object.

    Return
    ------
    is_ind : bool,
        True if the distribution is independent, False otherwise.
    """
    is_ind = np.all(np.tril(np.asarray(dist.getCorrelation()), k=-1) == 0.)
    return is_ind


def ishigami_variance(a, b):
    return 0.5 + a**2/8 + b**2*np.pi**8/18 + b * np.pi**4/5


def ishigami_partial_variance(a, b):
    v1 = 0.5 * (1 + b * np.pi**4 / 5)**2
    v2 = a**2 / 8
    v3 = 0
    return v1, v2, v3


def ishigami_total_variance(a, b):
    v13 = b**2 * np.pi**8 * 8 / 225
    v1, v2, v3 = ishigami_partial_variance(a, b)
    return v1 + v13, v2, v3 + v13


class Ishigami(ProbabilisticModel):
    """This class collect all the information about the Ishigami test function
    for sensitivity analysis.
    """

    def __init__(self, a=7., b=0.1):
        dim = 3
        margins = [ot.Uniform(-np.pi, np.pi)]*dim
        copula = ot.IndependentCopula(dim)
        ProbabilisticModel.__init__(
            self,
            model_func=ishigami_func,
            input_distribution=ot.ComposedDistribution(margins, copula)
        )
        self.a = a
        self.b = b
        self.name = 'Ishigami'

    @ProbabilisticModel.first_sobol_indices.getter
    def first_sobol_indices(self):
        """
        """
        if is_independent(self._input_distribution):
            a, b = self.a, self.b
            var_y = ishigami_variance(a, b)
            partial_var = ishigami_partial_variance(a, b)
            si = [vi / var_y for vi in partial_var]
            return np.asarray(si)
        else:
            return None

    @ProbabilisticModel.total_sobol_indices.getter
    def total_sobol_indices(self):
        """
        """
        if is_independent(self._input_distribution):
            a, b = self.a, self.b
            var_y = ishigami_variance(a, b)
            total_var = ishigami_total_variance(a, b)
            si = [vi / var_y for vi in total_var]
            return np.asarray(si)
        else:
            return None

    @ProbabilisticModel.shapley_indices.getter
    def shapley_indices(self):
        """
        """
        if is_independent(self._input_distribution):
            return np.asarray([0.437, 0.441, 0.12])
        else:
            return None

    @ProbabilisticModel.output_variance.getter
    def output_variance(self):
        a, b = self.a, self.b
        var_y = ishigami_variance(a, b)
        return var_y


class Gfunction(ProbabilisticModel):
    """This class collect all the information about the Gfunction test function
    for sensitivity analysis.
    """

    def __init__(self, dim, beta=None):
        margins = [ot.Uniform(0, 1)]*dim
        copula = ot.IndependentCopula(dim)
        ProbabilisticModel.__init__(
            self,
            model_func=g_func,
            input_distribution=ot.ComposedDistribution(margins, copula)
        )
        self.beta = beta
        self.name = 'G-function'

    @property
    def _partial_variances(self):
        return (1. / 3) / (1 + self.beta)**2

    @property
    def beta(self):
        """
        """
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta is None:
            beta = np.ones((self.dim, ))
        else:
            beta = np.asarray(beta)

        self.model_func = lambda x: g_func(x, beta)
        self._beta = beta

    @ProbabilisticModel.first_sobol_indices.getter
    def first_sobol_indices(self):
        """
        """
        if is_independent(self._input_distribution):
            return self._partial_variances / self.output_variance
        else:
            return None

    @ProbabilisticModel.total_sobol_indices.getter
    def total_sobol_indices(self):
        """
        """
        if is_independent(self._input_distribution):
            v = self._partial_variances
            vt = v.copy()
            for i in range(self.dim):
                vt[i] *= (1 + v[np.arange(self.dim) != i]).prod()

            return vt / self.output_variance
        else:
            return None

    @ProbabilisticModel.output_variance.getter
    def output_variance(self):
        v = self._partial_variances
        return (1 + v).prod() - 1


class ProductGaussian(ProbabilisticModel):
    """
    """

    def __init__(self, dim, beta=None):
        margins = [ot.Normal()]*dim
        copula = ot.NormalCopula(dim)
        ProbabilisticModel.__init__(
            self,
            model_func=product_func,
            input_distribution=ot.ComposedDistribution(margins, copula))
        self.beta = beta
        self.name = 'Product Gaussian'

    @property
    def beta(self):
        """
        """
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta is None:
            beta = np.ones((self.dim, ))
        else:
            beta = np.asarray(beta)

        self._beta = beta

    @property
    def first_sobol_indices(self):
        """
        """
        dim = self.dim
        beta = self.beta
        sigma = np.asarray(self.input_distribution.getCovariance())
        sigma_x = np.sqrt(sigma.diagonal())
        theta = np.asarray(self.copula.getParameter())
        var_y = (1. + theta**2) * (beta[0] *
                                   beta[1] * sigma_x[0] * sigma_x[1])**2
        s_corr = np.zeros((dim, ))
        if dim == 2:
            s_corr[0] = 2 * (theta * beta[0] * beta[1] *
                             sigma_x[0] * sigma_x[1])**2
            s_corr[1] = s_corr[0]
            indices = (s_corr) / var_y
            return indices

    @first_sobol_indices.setter
    def first_sobol_indices(self, indices):
        self._first_sobol_indices = indices

    @property
    def total_sobol_indices(self):
        """
        """
        dim = self.dim
        beta = self.beta
        sigma = np.asarray(self.input_distribution.getCovariance())
        sigma_x = np.sqrt(sigma.diagonal())
        theta = np.asarray(self.copula.getParameter())
        var_y = (1. + theta**2) * (beta[0] *
                                   beta[1] * sigma_x[0] * sigma_x[1])**2
        s_corr = np.zeros((dim, ))
        if dim == 2:
            s_corr[0] = (1. - theta**2) * beta[0]**2 * \
                beta[1]**2 * sigma_x[0]**2 * sigma_x[1]**2
            s_corr[1] = s_corr[0]
            indices = (s_corr) / var_y
            return indices

    @total_sobol_indices.setter
    def total_sobol_indices(self, indices):
        self._total_sobol_indices = indices

    @property
    def shapley_indices(self):
        """
        """
        dim = self.dim
        beta = self.beta
        sigma = np.asarray(self.input_distribution.getCovariance())
        sigma_x = np.sqrt(sigma.diagonal())
        theta = np.asarray(self.copula.getParameter())
        var_y = (1. + theta**2) * (beta[0] *
                                   beta[1] * sigma_x[0] * sigma_x[1])**2
        s_corr = np.zeros((dim, ))
        if dim == 2:
            s_corr[0] = 0.5*(1. + theta**2) * (beta[0] *
                                               beta[1] * sigma_x[0] * sigma_x[1])**2
            s_corr[1] = s_corr[0]
            indices = (s_corr) / var_y
            return indices

    @shapley_indices.setter
    def shapley_indices(self, indices):
        self._shapley_indices = indices


class AdditiveGaussian(ProbabilisticModel):
    """This class collect all the information about the Additive Gaussian test 
    function for sensitivity analysis.
    """

    def __init__(self, dim, means=None, std=None, beta=None):

        if means is None:
            means = np.zeros((dim, ))
        if std is None:
            std = np.ones((dim, ))

        margins = [ot.Normal(means[i], std[i]) for i in range(dim)]
        copula = ot.NormalCopula(dim)
        ProbabilisticModel.__init__(
            self,
            model_func=additive_func,
            input_distribution=ot.ComposedDistribution(margins, copula))
        self.beta = beta
        self.name = 'Additive Gaussian'
        self.type_indice = 'ind'

    @property
    def beta(self):
        """
        """
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta is None:
            beta = np.ones((self.dim, ))
        else:
            beta = np.asarray(beta)

        self.model_func = lambda x: additive_func(x, beta)
        self._beta = beta

    @ProbabilisticModel.output_variance.getter
    def output_variance(self):
        covariance = np.asarray(self.input_distribution.getCovariance())
        var_y = (self.beta.dot(covariance)).dot(self.beta)
        return var_y

    @ProbabilisticModel.first_sobol_indices.getter
    def first_sobol_indices(self):
        var_y = self.output_variance
        beta = self.beta
        dim = self.dim
        sigma = np.asarray(self.input_distribution.getCovariance())
        corr = np.asarray(self.input_distribution.getCorrelation())
        input_variance = sigma.diagonal()
        input_std = np.sqrt(input_variance)
        indices = np.zeros((dim, ))
        if self.type_indice == 'ind':
            for j in range(dim):
                c_j = np.asarray([i for i in range(dim) if i != j])
                inv_j = np.linalg.inv(sigma[c_j, :][:, c_j])
                tmp_j = sigma - (sigma[:, c_j].dot(inv_j)).dot(sigma[c_j, :])
                var_j = (beta.dot(tmp_j)).dot(beta)
                indices[j] = var_j
            return indices / var_y
        else:
            for j in range(dim):
                tmp_j = beta[j] * input_std[j]
                for j2 in range(dim):
                    if j2 != j:
                        tmp_j += beta[j2] * input_std[j2] * corr[j, j2]
                indices[j] = tmp_j**2
            return indices / var_y

    @ProbabilisticModel.total_sobol_indices.getter
    def total_sobol_indices(self):
        return self.first_sobol_indices

    @property
    def shapley_indices(self):
        """Return the Shapley indices
        """
        dim = self.dim
        beta = self.beta
        sigma = np.asarray(self.input_distribution.getCovariance())
        var_y = (beta.dot(sigma)).dot(beta)
        sigma_x = np.sqrt(sigma.diagonal())

        # Effects without correlation
        s_uncorr = (beta * sigma_x)**2

        # Effects with correlation
        theta = np.asarray(self.copula.getParameter())
        dep_pair = theta != 0
        rho = theta[dep_pair]
        if dim == 3 and len(rho) <= 1:
            rho = rho.item() if len(rho) == 1 else 0
            s_corr = np.zeros((dim, ))
            for i, j in [[1, 2], [2, 1]]:
                s_corr[i] = rho * beta[i] * beta[j] * sigma_x[i] * \
                    sigma_x[j] + 0.5 * rho**2 * (s_uncorr[j] - s_uncorr[i])

            indices = (s_uncorr + s_corr) / var_y
            return indices
        elif dim == 2:
            indices = np.asarray([1./dim]*dim)
            return indices
        else:
            return self._shapley_indices

    @shapley_indices.setter
    def shapley_indices(self, indices):
        self._shapley_indices = indices


def product_func(x, beta=None):
    """Product function.

    Parameters
    ----------
    x : array,
        The input variables. The shape should be d x n, with d the dimension and n the sample-size.

    Returns
    -------
    y : float or array,
        The function output. If n > 1, the function returns an array.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        dim = x.shape[0]
    else:
        n_sample, dim = x.shape

    if beta is None:
        beta = np.ones((dim, ))
    else:
        beta = np.asarray(beta)
    y = np.prod(beta * x, axis=1)
    return y


def additive_func(x, beta=None):
    """Additive function.

    Parameters
    ----------
    x : array,
        The input variables. The shape should be d x n, with d the dimension and n the sample-size.

    Returns
    -------
    y : float or array,
        The function output. If n > 1, the function returns an array.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        dim = x.shape[0]
    else:
        n_sample, dim = x.shape

    if beta is None:
        beta = np.ones((dim, ))
    else:
        beta = np.asarray(beta)
    y = np.dot(x, beta)
    return y


def g_func(x, beta=None):
    """G function.

    Parameters
    ----------
    x : array,
        The input variables. The shape should be d x n, with d the dimension and n the sample-size.

    Returns
    -------
    y : float or array,
        The function output. If n > 1, the function returns an array.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        dim = x.shape[0]
    else:
        n_sample, dim = x.shape

    if beta is None:
        beta = np.ones((dim, ))
    else:
        beta = np.asarray(beta)

    gi = (np.abs(4*x - 2) + beta) / (1 + beta)
    y = np.prod(gi, axis=1)
    return y


def ishigami_func(x, a=7, b=0.1):
    """Ishigami function.

    Parameters
    ----------
    x : array,
        The input variables. The shape should be 3 x n.

    Returns
    -------
    y : float or array,
        The function output. If n > 1, the function returns an array.
    """
    x = np.asarray(x).squeeze()
    if x.shape[0] == x.size:
        dim = x.shape[0]
        y = np.sin(x[0]) + a*np.sin(x[1])**2 + b*x[2]**4 * np.sin(x[0])
    else:
        dim = x.shape[1]
        y = np.sin(x[:, 0]) + a*np.sin(x[:, 1])**2 + b*x[:, 2]**4 * np.sin(x[:, 0])

    assert dim == 3, "Dimension problem %d != %d " % (3, ndim)

    return y
