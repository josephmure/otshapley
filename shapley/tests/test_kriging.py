from itertools import product

import openturns as ot
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
import pytest

from shapley.tests import Ishigami
from shapley.sobol import SobolKrigingIndices

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

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami.first_order_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
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

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami.first_order_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
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

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami.first_order_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
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

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami.first_order_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_total[0, :], ishigami.total_sobol_indices)
    np.testing.assert_array_less(ishigami.total_sobol_indices, quantiles_total[1, :])