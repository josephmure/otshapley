from itertools import product

import openturns as ot
import numpy as np
import pytest

from shapley.sobol import SobolIndices
from shapley.tests import Ishigami, AdditiveGaussian
from shapley.tests.utils import true_gaussian_full_ind_sobol
from shapley.kriging import KrigingModel

MODEL_BUDGET = 300

N_SAMPLE = 200
N_BOOT = 200
N_REALIZATION = 100
ESTIMATOR = 'soboleff2'

SAMPLINGS = ['lhs']
OT_BASIS = ['linear', 'constant', 'quadratic']
OT_KERNELS = ['matern', 'exponential', 'generalized-exponential']
SK_KERNELS = ['matern', 'RBF']
LIBRAIRIES = ['sklearn', 'gpflow']

@pytest.mark.parametrize("kernel, library", list(product(SK_KERNELS, LIBRAIRIES)))
def test_sobol_kriging_ishigami_ind(kernel, library):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    model = Ishigami()
    
    model_gp = KrigingModel(model=model, input_distribution=model.input_distribution)
    model_gp.generate_sample(n_sample=MODEL_BUDGET, sampling='lhs', sampling_type='uniform')
    model_gp.build(library=library, kernel=kernel, basis_type='linear')
    
    sobol = SobolIndices(input_distribution=model.input_distribution)
    sobol.build_sample(model=model_gp, n_sample=N_SAMPLE, n_realization=N_REALIZATION)
    sobol_results = sobol.compute_indices(n_boot=N_BOOT, estimator=ESTIMATOR)

    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(model.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], model.first_sobol_indices)
    np.testing.assert_array_less(model.first_sobol_indices, quantiles_first[1, :])

    quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(model.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_total[0, :], model.total_sobol_indices)
    np.testing.assert_array_less(model.total_sobol_indices, quantiles_total[1, :])
    
if False:
    @pytest.mark.parametrize("kernel, basis", list(product(OT_KERNELS, OT_BASIS)))
    def test_sobol_kriging_ishigami_ind_bench_OT(kernel, basis):
        ot.RandomGenerator.SetSeed(0)
        np.random.seed(0)
        model = Ishigami()
        
        model_gp = KrigingModel(model=model, input_distribution=model.input_distribution)
        model_gp.generate_sample(n_sample=MODEL_BUDGET, sampling='lhs', sampling_type='uniform')
        model_gp.build(library='OT', kernel=kernel, basis_type=basis)
        
        sobol = SobolIndices(input_distribution=model.input_distribution)
        sobol.build_sample(model=model_gp, n_sample=N_SAMPLE, n_realization=N_REALIZATION)
        sobol_results = sobol.compute_indices(n_boot=N_BOOT, estimator=ESTIMATOR)

        quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(model.dim, -1), [5, 95], axis=1)
        np.testing.assert_array_less(quantiles_first[0, :], model.first_sobol_indices)
        np.testing.assert_array_less(model.first_sobol_indices, quantiles_first[1, :])

        quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(model.dim, -1), [5, 95], axis=1)
        np.testing.assert_array_less(quantiles_total[0, :], model.total_sobol_indices)
        np.testing.assert_array_less(model.total_sobol_indices, quantiles_total[1, :])
             
THETAS = [[0.5, 0.8, 0.], [-0.5, 0.2, -0.7], [-0.49, -0.49, -0.49]]
INDICE_TYPES = ['full', 'ind']    
THETAS_TYPES = list(product(THETAS, INDICE_TYPES))

# Tests from Mara & Tarantola 2012/2015
@pytest.mark.parametrize("theta, ind_type", THETAS_TYPES)
def test_full_ind_sobol_kriging_gaussian_dep_gpflow(theta, ind_type):
    ot.RandomGenerator.SetSeed(0)
    np.random.seed(0)
    dim = 3
    model = AdditiveGaussian(dim=dim, beta=[1., 1., 1.])
    model.copula_parameters = theta
    
    true_full_indices, true_ind_indices = true_gaussian_full_ind_sobol(theta, dim=dim)
    true_indices = {'full': true_full_indices,
                    'ind': true_ind_indices}
    
    model_gp = KrigingModel(model=model, input_distribution=model.input_distribution)
    model_gp.generate_sample(n_sample=MODEL_BUDGET, sampling='lhs', sampling_type='uniform')
    model_gp.build(library='gpflow', kernel='matern', basis_type='linear')

    sobol = SobolIndices(input_distribution=model.input_distribution)
    sobol.build_uncorr_sample(model=model_gp, n_sample=300, n_realization=N_REALIZATION)
    sobol_results = sobol.compute_indices(n_boot=N_BOOT, estimator=ESTIMATOR, indice_type=ind_type)
    sobol_results.true_first_indices = true_indices[ind_type]
    sobol_results.true_total_indices = true_indices[ind_type]


    quantiles_first = np.percentile(sobol_results.full_first_indices.reshape(model.dim, -1), [5, 95], axis=1)
    np.testing.assert_array_less(quantiles_first[0, :], sobol_results.true_first_indices)
    np.testing.assert_array_less(sobol_results.true_first_indices, quantiles_first[1, :])
    if ESTIMATOR != 'sobol':
        quantiles_total = np.percentile(sobol_results.full_total_indices.reshape(model.dim, -1), [5, 95], axis=1)
        np.testing.assert_array_less(quantiles_total[0, :], sobol_results.true_total_indices)
        np.testing.assert_array_less(sobol_results.true_total_indices, quantiles_total[1, :])
