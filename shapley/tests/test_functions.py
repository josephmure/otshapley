import numpy as np
import openturns as ot

from shapley.base import ProbabilisticModel

class Ishigami(ProbabilisticModel):
    """This class collect all the information about the Ishigami test function for sensitivity analysis.
    """
    def __init__(self):
        dim = 3
        margins = [ot.Uniform(-np.pi, np.pi)]*dim
        copula = ot.IndependentCopula(dim)
        input_distribution = ot.ComposedDistribution(margins, copula)
        ProbabilisticModel.__init__(self, model_func=ishigami_func, input_distribution=input_distribution)
        self._first_order_sobol_indices = [0.314, 0.442, 0.]

def ishigami_func(x):
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

    assert ndim == 3, "Dimension problem %d != %d " % (3, ndim)

    if ndim == 1:
        y = np.sin(x[0]) + 7*np.sin(x[1])**2 + 0.1*x[2]**4 * np.sin(x[0])
    else:
        y = np.sin(x[:, 0]) + 7*np.sin(x[:, 1])**2 + 0.1*x[:, 2]**4 * np.sin(x[:, 0])

    return y

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