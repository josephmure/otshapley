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

    def build_mc_sample(self, model, n_sample):
        """Build the Monte-Carlo samples.

        Parameters
        ----------
        model : callable,
            The model function.
        n_sample : int,
            The sampling size of Monte-Carlo
        """
        Indices.build_mc_sample(self, model=model, n_sample=n_sample, n_realization=1)

    def build_uncorrelated_mc_sample(self, model, n_sample=100):
        """Build the Monte-Carlo samples.

        Parameters
        ----------
        model : callable,
            The model function.
        n_sample : int,
            The sampling size of Monte-Carlo
        """
        Indices.build_uncorrelated_mc_sample(self, model=model, n_sample=n_sample, n_realization=1)
    
    def compute_indices(self, n_boot=100, estimator='soboleff2'):
        """Compute the indices.

        Parameters
        ----------
        n_boot : int,
            The number of bootstrap samples.
        estimator : str,
            The type of estimator to use.
        
        Returns
        -------
        indices : list,
            The list of computed indices.
        """
        return Indices.compute_indices(self, n_boot=n_boot, estimator=estimator, calculation_method='monte-carlo')

    def compute_full_indices(self, n_boot, estimator):
        """
        """
        return Indices.compute_full_indices(self, n_boot=n_boot, estimator=estimator, calculation_method='monte-carlo')

    def compute_ind_indices(self, n_boot, estimator):
        """
        """
        return Indices.compute_ind_indices(self, n_boot=n_boot, estimator=estimator, calculation_method='monte-carlo')

class SobolKrigingIndices(KrigingIndices, Indices):
    """Estimation of the Sobol indices using Gaussian Process approximation.
    """
    def __init__(self, input_distribution):
        KrigingIndices.__init__(self, input_distribution)
        Indices.__init__(self, input_distribution)
        self.indice_func = sobol_indices

    def build_mc_sample(self, model, n_sample=100, n_realization=10):
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
        Indices.build_mc_sample(self, model, n_sample, n_realization)

    def build_uncorrelated_mc_sample(self, model, n_sample=100, n_realization=10):
        """Build the Monte-Carlo samples.

        Parameters
        ----------
        model : callable,
            The model function.
        n_sample : int,
            The sampling size of Monte-Carlo
        """
        Indices.build_uncorrelated_mc_sample(self, model=model, n_sample=n_sample, n_realization=n_realization)
    
    def compute_indices(self, n_boot=100, estimator='soboleff2'):
        """Compute the indices.

        Parameters
        ----------
        n_boot : int,
            The number of bootstrap samples.
        estimator : str,
            The type of estimator to use.
        
        Returns
        -------
        indices : list,
            The list of computed indices.
        """
        return Indices.compute_indices(self, n_boot=n_boot, 
                                       estimator=estimator, 
                                       calculation_method='kriging-mc')

    def compute_full_indices(self, n_boot, estimator):
        """
        """
        return Indices.compute_full_indices(self, n_boot=n_boot, estimator=estimator, calculation_method='kriging-mc')

    def compute_ind_indices(self, n_boot, estimator):
        """
        """
        return Indices.compute_ind_indices(self, n_boot=n_boot, estimator=estimator, calculation_method='kriging-mc')

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
    #var_indiv = m(Y2t * Y1) - mean2
    #var_total = m(Y2t * Y2) - mean2
    first_indice = var_indiv / var
    total_indice = 1. - var_total  / var

    return first_indice, total_indice


def sobol2007_estimator(Y1, Y2, Y2t):
    """
    """
    n_sample = Y1.shape[1]
    mean2 = m(Y1*Y2)
    var = v(Y1)

    #var_indiv = s((Y2t - Y2) * Y1)/(n_sample - 1)
    #var_total = s((Y2t - Y1) * Y2)/(n_sample - 1)
    var_indiv = m((Y2t - Y2) * Y1)
    var_total = m((Y2t - Y1) * Y2)
    first_indice = var_indiv / var
    total_indice = 1. - var_total / var

    return first_indice, total_indice


def soboleff1_estimator(Y1, Y2, Y2t):
    """
    """
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
    n_sample = Y1.shape[1]
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
    n_sample = Y1.shape[1]
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