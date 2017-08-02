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
        self.first_order_indice_func = first_order_sobol_indice
        self.first_order_full_indice_func = first_order_full_sobol_indice


class SobolKrigingIndices(KrigingIndices, SobolIndices):
    """Estimation of the Sobol indices using Gaussian Process approximation.
    """
    def __init__(self, input_distribution):
        KrigingIndices.__init__(self, input_distribution)
        SobolIndices.__init__(self, input_distribution)


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


def first_order_sobol_indice(Y, Yt, n_boot=1, boot_idx=None, estimator='janon1'):
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
    n_sample = Y.shape[0]
    assert n_sample == Yt.shape[0], "Matrices should have the same sizes"

    if estimator == 'janon1':
        estimator = estimator_first_order_janon1
    elif estimator == 'janon2':
        estimator = estimator_first_order_janon2
    elif estimator == 'sobol':
        estimator = estimator_first_order_sobol
    else:
        raise ValueError('Unknow estimator {0}'.format(estimator))

    if boot_idx is None:
        boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))

    indice = np.zeros((n_boot, ))
    indice[0] = estimator(Y, Yt)
    indice[1:] = estimator(Y[boot_idx], Yt[boot_idx])

    return indice if n_boot > 1 else indice.item()


def estimator_first_order_janon1(Y, Yt):
    """
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
        Yt = Yt.reshape(1, -1)
        
    m = lambda x : x.mean(axis=1)
    partial = m(Y * Yt) - m(Y) * m(Yt)
    total = m(Y**2) - m(Y)**2
    return partial / total


def estimator_first_order_janon2(Y, Yt):
    """
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
        Yt = Yt.reshape(1, -1)
        
    m = lambda x : x.mean(axis=1)
    partial = m(Y*Yt) - m((Y + Yt)/2.)**2
    total = m((Y**2 + Yt**2)/2.) - m((Y + Yt)/2.)**2
    return partial / total

def estimator_first_order_sobol(Y, Yt):
    """
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
        Yt = Yt.reshape(1, -1)

    m = lambda x : x.mean(axis=1)
    partial = m(Y * Yt) - m(Y)**2
    total = m(Y**2) - m(Y)**2
    return partial / total

def mara_estimator(Y1, Y2, Y2i):
    """
    """
    if Y1.ndim == 1:
        Y1 = Y1.reshape(1, -1)
        Y2 = Y2.reshape(1, -1)
        Y2i = Y2i.reshape(1, -1)

    m = lambda x : x.mean(axis=1)
    v = lambda x : x.var(axis=1)
    partial = m(Y1 *(Y2i - Y2))
    total = (v(Y1) + v(Y2))/2.
    return partial/total