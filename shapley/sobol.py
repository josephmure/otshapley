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


class SobolKrigingIndices(KrigingIndices, SobolIndices):
    """Estimation of the Sobol indices using Gaussian Process approximation.
    """
    def __init__(self, input_distribution):
        KrigingIndices.__init__(self, input_distribution)
        SobolIndices.__init__(self, input_distribution)


def first_order_sobol_indices(output_sample_1, all_output_sample_2, n_boot=1, boot_idx=None, estimator='janon'):
    """
    """
    dim = all_output_sample_2.shape[1]
    first_indices = np.zeros((dim, n_boot))
    Y = output_sample_1
    for i in range(dim):
        Yt = all_output_sample_2[:, i]
        first_indices[i, :] = first_order_sobol_indice(Y, Yt, n_boot=n_boot)
    return first_indices


def first_order_sobol_indice(Y, Yt, n_boot=1, boot_idx=None, estimator='janon'):
    """Compute the Sobol indices from the to

    Parameters
    ----------
    """
    n_sample = Y.shape[0]
    assert n_sample == Yt.shape[0], "Matrices should have the same sizes"

    if estimator == 'janon':
        estimator = janon_estimator

    first_indice = np.zeros((n_boot, ))
    first_indice[0] = estimator(Y, Yt)
    if boot_idx is None:
        boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))
    first_indice[1:] = estimator(Y[boot_idx], Yt[boot_idx])

    return first_indice if n_boot > 1 else first_indice.item()


def janon_estimator(Y, Yt):
    """
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
        Yt = Yt.reshape(1, -1)

    partial = (Y * Yt).mean(axis=1) - ((Y + Yt).mean(axis=1)*0.5)**2
    total = (Y**2).mean(axis=1) - ((Y + Yt).mean(axis=1)*0.5)**2
    return partial / total