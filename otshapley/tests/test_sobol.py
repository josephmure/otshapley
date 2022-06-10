from itertools import product

import numpy as np
import openturns as ot
import pytest

from otshapley.sobol import SobolIndices
from otshapley.tests.test_functions import Ishigami, AdditiveGaussian
from otshapley.tests.utils import true_gaussian_full_ind_sobol

N_SAMPLE_WITHOUT_BOOT = 80000
N_SAMPLE_WITH_BOOT = 1000
N_BOOT = 1000

ESTIMATORS = ['sobol', 'sobol2002', 'sobol2007', 'soboleff1', 'soboleff2']

@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_sobol_ishigami_independence_no_boot(estimator):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    model = Ishigami()
    sobol = SobolIndices(input_distribution=model.input_distribution)
    sobol.build_sample(model=model, n_sample=N_SAMPLE_WITHOUT_BOOT)
    sobol_results = sobol.compute_indices(n_boot=1, estimator=estimator)
    first_indices_mc = sobol_results.first_indices
    total_indices_mc = sobol_results.total_indices

    np.testing.assert_array_almost_equal(first_indices_mc, model.first_sobol_indices, decimal=2)
    if estimator != 'sobol':
        np.testing.assert_array_almost_equal(total_indices_mc, model.total_sobol_indices, decimal=2)
    
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_sobol_ishigami_independence_boot(estimator):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    model = Ishigami()
    sobol = SobolIndices(input_distribution=model.input_distribution)
    sobol.build_sample(model=model, n_sample=N_SAMPLE_WITH_BOOT)
    sobol_results = sobol.compute_indices(n_boot=N_BOOT, estimator=estimator)

    quantiles_first = np.percentile(sobol_results.full_first_indices, [1, 99], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], model.first_sobol_indices)
    np.testing.assert_array_less(model.first_sobol_indices, quantiles_first[1, :])
    if estimator != 'sobol':
        quantiles_total = np.percentile(sobol_results.full_total_indices, [1, 99], axis=1)
        np.testing.assert_array_less(quantiles_total[0, :], model.total_sobol_indices)
        np.testing.assert_array_less(model.total_sobol_indices, quantiles_total[1, :])

THETAS = [[0.5, 0.8, 0.], [-0.5, 0.2, -0.7], [-0.49, -0.49, -0.49]]
INDICE_TYPES = ['full', 'ind']

ESTIMATORS_THETAS_TYPES = list(product(ESTIMATORS, THETAS, INDICE_TYPES))


# Tests from Mara & Tarantola 2012/2015
@pytest.mark.parametrize("estimator, theta, ind_type", ESTIMATORS_THETAS_TYPES)
def test_full_ind_sobol_gaussian_dependent_boot(estimator, theta, ind_type):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    dim = 3
    model = AdditiveGaussian(dim=dim, beta=[1., 1., 1.])
    model.copula_parameters = theta
    
    true_full_indices, true_ind_indices = true_gaussian_full_ind_sobol(theta, dim=dim)
    true_indices = {'full': true_full_indices,
                    'ind': true_ind_indices}

    sobol = SobolIndices(input_distribution=model.input_distribution)
    sobol.build_uncorr_sample(model=model, n_sample=N_SAMPLE_WITH_BOOT)
    sobol_results = sobol.compute_indices(n_boot=N_BOOT, estimator=estimator, indice_type=ind_type)
    sobol_results.true_first_indices = true_indices[ind_type]
    sobol_results.true_total_indices = true_indices[ind_type]

    quantiles_first = np.percentile(sobol_results.full_first_indices, [1, 99], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], sobol_results.true_first_indices)
    np.testing.assert_array_less(sobol_results.true_first_indices, quantiles_first[1, :])
    if estimator != 'sobol':
        quantiles_total = np.percentile(sobol_results.full_total_indices, [1, 99], axis=1)
        np.testing.assert_array_less(quantiles_total[0, :], sobol_results.true_total_indices)
        np.testing.assert_array_less(sobol_results.true_total_indices, quantiles_total[1, :])
