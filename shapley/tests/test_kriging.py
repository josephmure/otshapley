import openturns as ot
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt

from shapley.tests import Ishigami
from shapley.sobol import SobolKrigingIndices

MODEL_BUDGETS = [100, 100, 500]

N_SAMPLE = 1000
N_BOOT = 300
N_REALIZATION = 100

BASIS_TYPES = ['linear', 'constant']
KERNELS = ['matern', 'exponential', 'generalized-exponential', 'spherical']
SAMPLINGS = ['lhs', 'monte-carlo']
ESTIMATOR = 'janon2'

def test_sobol_kriging_ishigami_independence():
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    ishigami = Ishigami()
    dim = ishigami.ndim
    sobol_kriging = SobolKrigingIndices(input_distribution=ishigami.input_distribution)
    basis_type = 'linear'
    sampling = 'lhs'
    kernel = 'spherical'
    meta_model = sobol_kriging.build_meta_model(model=ishigami, n_sample=MODEL_BUDGETS[0], 
                                                basis_type=basis_type, kernel=kernel, sampling=sampling)
    sobol_kriging.build_mc_sample(model=meta_model, n_sample=N_SAMPLE, n_realization=N_REALIZATION)
    first_sobol_indices = sobol_kriging.compute_indices(n_boot=N_BOOT, estimator=ESTIMATOR)
    quantiles = np.percentile(first_sobol_indices.reshape(ishigami.ndim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles[0, :], ishigami.first_order_sobol_indices)
    np.testing.assert_array_less(ishigami._first_order_sobol_indices, quantiles[1, :])