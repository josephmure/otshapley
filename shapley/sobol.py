import openturns as ot
import numpy as np

from .indices import Indices
from .kriging import KrigingIndices


class SobolIndices(Indices):
    """The class of Sobol indices.

    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        And OpenTURNS distribution object.
    """
    def __init__(self, input_distribution):
        Indices.__init__(self, input_distribution)
        self.indice_func = sobol_indices


class SobolKrigingIndices(SobolIndices, KrigingIndices):
    """Estimation of the Sobol indices using Gaussian Process approximation.
    """
    def __init__(self, input_distribution):
        SobolIndices.__init__(self, input_distribution)
        KrigingIndices.__init__(self, input_distribution)


def sobol_indices(Y1, Y2, Y2t, n_boot=1, boot_idx=None, estimator='sobol2002'):
    """Compute the Sobol indices from the to

    Parameters
    ----------
    Y : array,
        The 

    Returns
    -------
    indice : int or array,
        The first order sobol indice estimation.
    """
    n_sample = Y1.shape[0]
    assert n_sample == Y2.shape[0], "Matrices should have the same sizes"
    assert n_sample == Y2t.shape[0], "Matrices should have the same sizes"
    assert estimator in _ESTIMATORS, 'Unknow estimator {0}'.format(estimator)

    estimator = _ESTIMATORS[estimator]

    if boot_idx is None:
        boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))

    first_indice = np.zeros((n_boot, ))
    total_indice = np.zeros((n_boot, ))
    first_indice[0], total_indice[0] = estimator(Y1, Y2, Y2t)
    first_indice[1:], total_indice[1:] = estimator(Y1[boot_idx], Y2[boot_idx], Y2t[boot_idx])

    if n_boot == 1:
        first_indice = first_indice.item()
        total_indice = total_indice.item()

    return first_indice, total_indice


def first_order_full_sobol_indice(Y1, Y2, Y2i, n_boot=1, boot_idx=None, estimator='mara'):
    """Compute the Sobol indices from the to

    Parameters
    ----------
    """
    n_sample = Y1.shape[0]

    if estimator == 'mara':
        estimator = mara_estimator

    first_indice = np.zeros((n_boot, ))
    first_indice[0] = estimator(Y1, Y2, Y2i)
    if boot_idx is None:
        boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))
    if n_boot > 1:
        first_indice[1:] = estimator(Y1[boot_idx], Y2[boot_idx], Y2i[boot_idx])

    return first_indice if n_boot > 1 else first_indice.item()


m = lambda x : x.mean(axis=1)
s = lambda x : x.sum(axis=1)
v = lambda x : x.var(axis=1)

def sobol_estimator(Y1, Y2, Y2t):
    """
    """
    if Y1.ndim == 1:
        Y1 = Y1.reshape(1, -1)
        Y2t = Y2t.reshape(1, -1)

    mean2 = m(Y1)**2
    var = v(Y1)

    var_indiv = m(Y2t * Y1) - mean2
    first_indice = var_indiv / var
    total_indice = None

    return first_indice, total_indice

def sobol2002_estimator(Y1, Y2, Y2t):
    """
    """
    if Y1.ndim == 1:
        Y1 = Y1.reshape(1, -1)
        Y2 = Y2.reshape(1, -1)
        Y2t = Y2t.reshape(1, -1)

    n_sample = Y1.shape[1]
    mean2 = m(Y1*Y2)
    var = v(Y1)

    var_indiv = s(Y2t * Y1)/(n_sample - 1) - mean2
    var_total = s(Y2t * Y2)/(n_sample - 1) - mean2
    first_indice = var_indiv / var
    total_indice = 1. - var_total  / var

    return first_indice, total_indice


def sobol2007_estimator(Y1, Y2, Y2t):
    """
    """
    if Y1.ndim == 1:
        Y1 = Y1.reshape(1, -1)
        Y2 = Y2.reshape(1, -1)
        Y2t = Y2t.reshape(1, -1)

    n_sample = Y1.shape[1]
    mean2 = m(Y1*Y2)
    var = v(Y1)

    var_indiv = s((Y2t - Y2) * Y1)/(n_sample - 1)
    var_total = s((Y2t - Y1) * Y2)/(n_sample - 1)
    first_indice = var_indiv / var
    total_indice = 1. - var_total / var

    return first_indice, total_indice


def soboleff1_estimator(Y1, Y2, Y2t):
    """
    """
    if Y1.ndim == 1:
        Y1 = Y1.reshape(1, -1)
        Y2 = Y2.reshape(1, -1)
        Y2t = Y2t.reshape(1, -1)

    n_sample = Y1.shape[1]
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
    if Y1.ndim == 1:
        Y1 = Y1.reshape(1, -1)
        Y2 = Y2.reshape(1, -1)
        Y2t = Y2t.reshape(1, -1)

    n_sample = Y1.shape[1]
    mean2 = m((Y1 + Y2t)/2.)**2
    var = m((Y1**2 + Y2t**2 )/2.) - m((Y1 + Y2t)/2.)**2

    var_indiv = m(Y2t * Y1) - mean2
    var_total = m(Y2t * Y2) - mean2

    first_indice = var_indiv / var
    total_indice = 1. - var_total  / var

    return first_indice, total_indice

_ESTIMATORS = {
    'sobol': sobol_estimator,
    'sobol2002': sobol2002_estimator,
    'sobol2007': sobol2007_estimator,
    'soboleff1': soboleff1_estimator,
    'soboleff2': soboleff2_estimator
    }