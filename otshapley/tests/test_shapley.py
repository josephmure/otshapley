import numpy as np
import openturns as ot

from otshapley import ShapleyIndices
from otshapley.tests.test_functions import Ishigami, AdditiveGaussian

N_BOOT = 1000
N_PERMS = None
THETA = [0., 0., 0.5]

def test_shapley_gaussian_no_boot():
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    dim = 3
    model = AdditiveGaussian(dim=dim, beta=None)
    model.margins = [ot.Normal()]*(dim-1) + [ot.Normal(0, 2.)]
    theta = THETA
    model.copula_parameters = theta
    
    shapley = ShapleyIndices(input_distribution=model.input_distribution)
    shapley.build_sample(model=model, n_perms=None, n_var=10000, n_outer=1000, n_inner=100)
    shapley_results = shapley.compute_indices(n_boot=1)
    first_indices = shapley_results.first_indices
    total_indices = shapley_results.total_indices
    shapley_indices = shapley_results.shapley_indices

    np.testing.assert_array_almost_equal(first_indices.ravel(), model.first_sobol_indices, decimal=2)
    np.testing.assert_array_almost_equal(total_indices.ravel(), model.total_sobol_indices, decimal=2)
    np.testing.assert_array_almost_equal(shapley_indices.ravel(), model.shapley_indices, decimal=2)
    
    
def test_shapley_gaussian_boot():
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    dim = 3
    model = AdditiveGaussian(dim=dim, beta=None)
    model.margins = [ot.Normal()]*(dim-1) + [ot.Normal(0, 2.)]
    theta = THETA
    model.copula_parameters = theta
    
    shapley = ShapleyIndices(input_distribution=model.input_distribution)
    shapley.build_sample(model=model, n_perms=None, n_var=1000, n_outer=100, n_inner=10)
    shapley_results = shapley.compute_indices(n_boot=N_BOOT)

    quantiles_first = np.percentile(shapley_results.full_first_indices, [1, 99], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], model.first_sobol_indices)
    np.testing.assert_array_less(model.first_sobol_indices, quantiles_first[1, :])
    
    quantiles_total = np.percentile(shapley_results.full_total_indices, [1, 99], axis=1)
    np.testing.assert_array_less(quantiles_total[0, :], model.total_sobol_indices)
    np.testing.assert_array_less(model.total_sobol_indices, quantiles_total[1, :])
    
    quantiles_shapley = np.percentile(shapley_results.full_shapley_indices, [1, 99], axis=1)
    np.testing.assert_array_less(quantiles_shapley[0, :], model.shapley_indices)
    np.testing.assert_array_less(model.shapley_indices, quantiles_shapley[1, :])
        
    
def test_shapley_ishigami_ind_boot():
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    model = Ishigami()
    dim = model.dim
    model.copula = ot.NormalCopula(dim)
    
    shapley = ShapleyIndices(input_distribution=model.input_distribution)
    shapley.build_sample(model=model, n_perms=None, n_var=1000, n_outer=100, n_inner=10)
    shapley_results = shapley.compute_indices(n_boot=N_BOOT)

    quantiles_first = np.percentile(shapley_results.full_first_indices, [1, 99], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], model.first_sobol_indices)
    np.testing.assert_array_less(model.first_sobol_indices, quantiles_first[1, :])
    
    quantiles_total = np.percentile(shapley_results.full_total_indices, [1, 99], axis=1)
    np.testing.assert_array_less(quantiles_total[0, :], model.total_sobol_indices)
    np.testing.assert_array_less(model.total_sobol_indices, quantiles_total[1, :])
    
    quantiles_shapley = np.percentile(shapley_results.full_shapley_indices, [1, 99], axis=1)
    np.testing.assert_array_less(quantiles_shapley[0, :], model.shapley_indices)
    np.testing.assert_array_less(model.shapley_indices, quantiles_shapley[1, :])
