import time

import numpy as np
import pandas as pd
import openturns as ot

from shapley.sobol import SobolIndices
from shapley.tests.test_functions import Ishigami

SOBOL_ESTIMATORS = ['janon1', 'janon2', 'sobol']

N_SAMPLE_WITH_BOOT = 50000
N_SAMPLE_WITHOUT_BOOT = 1000
N_BOOT = 1000

def test_sobol_ishigami_independence_janon1_no_boot():
    ishigami = Ishigami()
    sobol = SobolIndices(input_distribution=ishigami.input_distribution)
    sobol.build_mc_sample(model=ishigami, n_sample=N_SAMPLE_WITH_BOOT)
    first_sobol_indices = sobol.compute_indices(n_boot=1, estimator='janon1')
    np.testing.assert_array_almost_equal(first_sobol_indices.mean(axis=1), ishigami._first_order_sobol_indices, decimal=2)

def test_sobol_ishigami_independence_janon1_boot():
    ishigami = Ishigami()
    sobol = SobolIndices(input_distribution=ishigami.input_distribution)
    sobol.build_mc_sample(model=ishigami, n_sample=N_SAMPLE_WITHOUT_BOOT)
    first_sobol_indices = sobol.compute_indices(n_boot=N_BOOT, estimator='janon1')
    quantiles = np.percentile(first_sobol_indices, [5, 95], axis=1)
    np.testing.assert_array_less(quantiles[0, :], ishigami._first_order_sobol_indices)
    np.testing.assert_array_less(ishigami._first_order_sobol_indices, quantiles[1, :])

def test_sobol_ishigami_independence_janon2_no_boot():
    ishigami = Ishigami()
    sobol = SobolIndices(input_distribution=ishigami.input_distribution)
    sobol.build_mc_sample(model=ishigami, n_sample=N_SAMPLE_WITH_BOOT)
    first_sobol_indices = sobol.compute_indices(n_boot=1, estimator='janon2')
    np.testing.assert_array_almost_equal(first_sobol_indices.mean(axis=1), ishigami._first_order_sobol_indices, decimal=2)

def test_sobol_ishigami_independence_janon2_boot():
    ishigami = Ishigami()
    sobol = SobolIndices(input_distribution=ishigami.input_distribution)
    sobol.build_mc_sample(model=ishigami, n_sample=N_SAMPLE_WITHOUT_BOOT)
    first_sobol_indices = sobol.compute_indices(n_boot=N_BOOT, estimator='janon2')
    quantiles = np.percentile(first_sobol_indices, [5, 95], axis=1)
    np.testing.assert_array_less(quantiles[0, :], ishigami._first_order_sobol_indices)
    np.testing.assert_array_less(ishigami._first_order_sobol_indices, quantiles[1, :])

def test_sobol_ishigami_independence_sobol_no_boot():
    ishigami = Ishigami()
    sobol = SobolIndices(input_distribution=ishigami.input_distribution)
    sobol.build_mc_sample(model=ishigami, n_sample=N_SAMPLE_WITH_BOOT)
    first_sobol_indices = sobol.compute_indices(n_boot=1, estimator='sobol')
    np.testing.assert_array_almost_equal(first_sobol_indices.mean(axis=1), ishigami._first_order_sobol_indices, decimal=2)

def test_sobol_ishigami_independence_sobol_boot():
    ishigami = Ishigami()
    sobol = SobolIndices(input_distribution=ishigami.input_distribution)
    sobol.build_mc_sample(model=ishigami, n_sample=N_SAMPLE_WITHOUT_BOOT)
    first_sobol_indices = sobol.compute_indices(n_boot=N_BOOT, estimator='sobol')
    quantiles = np.percentile(first_sobol_indices, [5, 95], axis=1)
    np.testing.assert_array_less(quantiles[0, :], ishigami._first_order_sobol_indices)
    np.testing.assert_array_less(ishigami._first_order_sobol_indices, quantiles[1, :])
