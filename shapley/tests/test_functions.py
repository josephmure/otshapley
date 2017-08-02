import numpy as np
import openturns as ot

class Ishigami(object):
    """This class collect all the information about the Ishigami test function for sensitivity analysis.
    """
    def __init__(self):
        self._ndim = 3
        self._margins = [ot.Uniform(-np.pi, np.pi)]*self.ndim
        self._copula = ot.IndependentCopula(self.ndim)
        self._first_order_sobol_indices = [0.314, 0.442, 0.]

    def __call__(self, x):
        """Ishigami function.

        Parameters
        ----------
        x : array,
            The input variables. The shape should be 3 x n.

        Returns
        -------
        y : int or array,
            The function output.
        """
        x = np.asarray(x).squeeze()
        if x.ndim == 1:
            ndim = x.shape[0]
        else:
            ndim = x.shape[1]

        assert ndim == self._ndim, "Dimension problem %d != %d " % (self.ndim, ndim)

        if ndim == 1:
            y = np.sin(x[0]) + 7*np.sin(x[1])**2 + 0.1*x[2]**4 * np.sin(x[0])
        else:
            y = np.sin(x[:, 0]) + 7*np.sin(x[:, 1])**2 + 0.1*x[:, 2]**4 * np.sin(x[:, 0])

        return y

    @property
    def first_order_sobol_indices(self):
        """The true first order sobol indices.
        """
        return self._first_order_sobol_indices

    @property
    def ndim(self):
        """Problem dimension.
        """
        return self._ndim
    @property
    def input_distribution(self):
        """The OpenTURNS input distribution.
        """
        return ot.ComposedDistribution(self._margins, self._copula)

    @property
    def margins(self):
        """The problem margins.
        """
        return selt._margins

    @property
    def copula(self):
        """
        """
        return self._copula

    @copula.setter
    def copula(self, copula):
        assert isinstance(copula, ot.CopulaImplementation), "The copula should be an OpenTURNS implementation."
        self._copula = copula

def additive_linear(x, beta=None):
    """
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
    return np.dot(x, beta)

ishigami_true_indices = [0.314, 0.442, 0.]