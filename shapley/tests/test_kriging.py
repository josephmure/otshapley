from itertools import product

import openturns as ot
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import pytest

from shapley.tests import Ishigami, AdditiveGaussian
from shapley.sobol import SobolKrigingIndices
from shapley.tests.utils import true_gaussian_full_ind_sobol

MODEL_BUDGET = 300

N_SAMPLE = 200
N_BOOT = 200
N_REALIZATION = 100
ESTIMATOR = 'soboleff2'

SAMPLINGS = ['lhs']
BASIS_TYPES = ['linear', 'constant', 'quadratic']
OT_KERNELS = ['exponential', 'generalized-exponential']
SK_KERNELS = ['matern', 'RBF']

def test_sobol_kriging_ishigami_ind_bench_matern_OT():
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    sobol_kriging = SobolKrigingIndices(input_distribution=ishigami.input_distribution)
    meta_model = sobol_kriging.build_meta_model(model=ishigami, n_sample=300, 
                                                basis_type='linear', kernel='matern', sampling='monte-carlo')
    sobol_kriging.build_mc_sample(model=meta_model, n_sample=500, n_realization=100)
    sobol_results = sobol_kriging.compute_indices(n_boot=200, estimator=ESTIMATOR)

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(ishigami.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami.first_order_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(ishigami.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_total[0, :], ishigami.total_sobol_indices)
    np.testing.assert_array_less(ishigami.total_sobol_indices, quantiles_total[1, :])


# Spherical has low kriging variance. It's hard to make it converge with a bad model.
def test_sobol_kriging_ishigami_ind_bench_spherical_OT():
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    sobol_kriging = SobolKrigingIndices(input_distribution=ishigami.input_distribution)
    meta_model = sobol_kriging.build_meta_model(model=ishigami, n_sample=300, 
                                                basis_type='quadratic', kernel='spherical', sampling='monte-carlo')
    sobol_kriging.build_mc_sample(model=meta_model, n_sample=500, n_realization=100)
    sobol_results = sobol_kriging.compute_indices(n_boot=200, estimator=ESTIMATOR)

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(ishigami.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami.first_order_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(ishigami.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_total[0, :], ishigami.total_sobol_indices)
    np.testing.assert_array_less(ishigami.total_sobol_indices, quantiles_total[1, :])
    
SAMPLING_BASIS_KERNELS = list(product(SAMPLINGS, BASIS_TYPES, OT_KERNELS))

@pytest.mark.parametrize("sampling, basis_type, kernel", SAMPLING_BASIS_KERNELS)
def test_sobol_kriging_ishigami_ind_OT(sampling, basis_type, kernel):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    sobol_kriging = SobolKrigingIndices(input_distribution=ishigami.input_distribution)
    meta_model = sobol_kriging.build_meta_model(model=ishigami, n_sample=MODEL_BUDGET, 
                                                basis_type=basis_type, kernel=kernel, sampling=sampling, library='OT')
    sobol_kriging.build_mc_sample(model=meta_model, n_sample=N_SAMPLE, n_realization=N_REALIZATION)

    sobol_results = sobol_kriging.compute_indices(n_boot=N_BOOT, estimator=ESTIMATOR)

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(ishigami.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami.first_order_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(ishigami.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_total[0, :], ishigami.total_sobol_indices)
    np.testing.assert_array_less(ishigami.total_sobol_indices, quantiles_total[1, :])

SAMPLING_BASIS_KERNELS = list(product(SAMPLINGS, SK_KERNELS))

@pytest.mark.parametrize("sampling, kernel", SAMPLING_BASIS_KERNELS)
def test_sobol_kriging_ishigami_ind_SK(sampling, kernel):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    sobol_kriging = SobolKrigingIndices(input_distribution=ishigami.input_distribution)
    meta_model = sobol_kriging.build_meta_model(model=ishigami, n_sample=MODEL_BUDGET, kernel=kernel, sampling=sampling, library='sklearn')
    sobol_kriging.build_mc_sample(model=meta_model, n_sample=N_SAMPLE, n_realization=N_REALIZATION)

    sobol_results = sobol_kriging.compute_indices(n_boot=N_BOOT, estimator=ESTIMATOR)

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(ishigami.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami.first_order_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(ishigami.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_total[0, :], ishigami.total_sobol_indices)
    np.testing.assert_array_less(ishigami.total_sobol_indices, quantiles_total[1, :])

    
THETAS = [[0.5, 0.8, 0.], [-0.5, 0.2, -0.7], [-0.49, -0.49, -0.49]]
INDICE_TYPES = ['full', 'ind']

#theta = THETAS[0]
#ind_type = INDICE_TYPES[1]

THETAS_TYPES = list(product(THETAS, INDICE_TYPES))

# Tests from Mara & Tarantola 2012/2015
@pytest.mark.parametrize("theta, ind_type", THETAS_TYPES)
def test_full_ind_sobol_kriging_gaussian_dep_SK(theta, ind_type):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    dim = 3
    model = AdditiveGaussian(dim=dim, beta=[1., 1., 1.])
    model.copula_parameters = theta

    sobol_kriging = SobolKrigingIndices(input_distribution=model.input_distribution)
    meta_model = sobol_kriging.build_meta_model(model=model, n_sample=MODEL_BUDGET, kernel='matern', sampling='monte-carlo', library='sklearn')

    sobol_kriging.build_uncorrelated_mc_sample(model=meta_model, n_sample=300, n_realization=N_REALIZATION)
    
    if ind_type == 'full':
        sobol_results = sobol_kriging.compute_full_indices(n_boot=N_BOOT, estimator=ESTIMATOR)
        model.first_order_sobol_indices = true_gaussian_full_ind_sobol(theta, dim)[0]
        model.total_sobol_indices = model.first_order_sobol_indices
    elif ind_type == 'ind':
        sobol_results = sobol_kriging.compute_ind_indices(n_boot=N_BOOT, estimator=ESTIMATOR)
        model.first_order_sobol_indices = true_gaussian_full_ind_sobol(theta, dim)[1]
        model.total_sobol_indices = model.first_order_sobol_indices
    else:
        raise ValueError('Unknow value {0}'.format(ind_type))

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(model.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], model.first_order_sobol_indices)
    np.testing.assert_array_less(model.first_order_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(model.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_total[0, :], model.total_sobol_indices)
    np.testing.assert_array_less(model.total_sobol_indices, quantiles_total[1, :])