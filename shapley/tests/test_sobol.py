import time

import numpy as np
import pandas as pd
import openturns as ot
import pytest

from shapley.sobol import SobolIndices
from shapley.tests.test_functions import Ishigami

N_SAMPLE_WITHOUT_BOOT = 70000
N_SAMPLE_WITH_BOOT = 5000
N_BOOT = 1000

ESTIMATORS = ['janon1', 'janon2', 'sobol']

@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_sobol_ishigami_independence_no_boot(estimator):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    sobol = SobolIndices(input_distribution=ishigami.input_distribution)
    sobol.build_mc_sample(model=ishigami, n_sample=N_SAMPLE_WITHOUT_BOOT)
    first_sobol_indices = sobol.compute_indices(n_boot=1, estimator=estimator)
    np.testing.assert_array_almost_equal(first_sobol_indices.mean(axis=1), ishigami.first_order_sobol_indices, decimal=2)
    
@pytest.mark.parametrize("estimator", ESTIMATORS)
def test_sobol_ishigami_independence_boot(estimator):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    sobol = SobolIndices(input_distribution=ishigami.input_distribution)
    sobol.build_mc_sample(model=ishigami, n_sample=N_SAMPLE_WITH_BOOT)
    first_sobol_indices = sobol.compute_indices(n_boot=N_BOOT, estimator=estimator)
    quantiles = np.percentile(first_sobol_indices, [5, 95], axis=1)
    np.testing.assert_array_less(quantiles[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami.first_order_sobol_indices, quantiles[1, :])