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

MODEL_BUDGET = 100

N_SAMPLE = 200
N_BOOT = 100
N_REALIZATION = 50
ESTIMATOR = 'janon2'

SAMPLINGS = ['lhs', 'monte-carlo']
BASIS_TYPES = ['linear', 'constant', 'quadratic']
KERNELS = ['exponential', 'generalized-exponential']

SAMPLING_BASIS_KERNELS = list(product(SAMPLINGS, BASIS_TYPES, KERNELS))

def test_sobol_kriging_ishigami_ind_bench_matern():
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    sobol_kriging = SobolKrigingIndices(input_distribution=ishigami.input_distribution)
    meta_model = sobol_kriging.build_meta_model(model=ishigami, n_sample=300, 
                                                basis_type='linear', kernel='matern', sampling='lhs')
    sobol_kriging.build_mc_sample(model=meta_model, n_sample=500, n_realization=100)
    first_sobol_indices = sobol_kriging.compute_indices(n_boot=200, estimator=ESTIMATOR)
    quantiles = np.percentile(first_sobol_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami._first_order_sobol_indices, quantiles[1, :])


# Spherical has low kriging variance. It's hard to make it converge with a bad model.
def test_sobol_kriging_ishigami_ind_bench_spherical():
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    sobol_kriging = SobolKrigingIndices(input_distribution=ishigami.input_distribution)
    meta_model = sobol_kriging.build_meta_model(model=ishigami, n_sample=8, 
                                                basis_type='linear', kernel='spherical', sampling='lhs')
    sobol_kriging.build_mc_sample(model=meta_model, n_sample=500, n_realization=100)
    first_sobol_indices = sobol_kriging.compute_indices(n_boot=200, estimator=ESTIMATOR)
    quantiles = np.percentile(first_sobol_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami._first_order_sobol_indices, quantiles[1, :])


@pytest.mark.parametrize("sampling, basis_type, kernel", SAMPLING_BASIS_KERNELS)
def test_sobol_kriging_ishigami_ind(sampling, basis_type, kernel):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    sobol_kriging = SobolKrigingIndices(input_distribution=ishigami.input_distribution)
    meta_model = sobol_kriging.build_meta_model(model=ishigami, n_sample=MODEL_BUDGET, 
                                                basis_type=basis_type, kernel=kernel, sampling=sampling)
    sobol_kriging.build_mc_sample(model=meta_model, n_sample=N_SAMPLE, n_realization=N_REALIZATION)
    first_sobol_indices = sobol_kriging.compute_indices(n_boot=N_BOOT, estimator=ESTIMATOR)
    quantiles = np.percentile(first_sobol_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami._first_order_sobol_indices, quantiles[1, :])