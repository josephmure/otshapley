import openturns as ot
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from shapley import ShapleyIndices
from shapley.tests import AdditiveGaussian
from shapley.tests.utils import true_gaussian_full_ind_sobol

if None:
    ## Model parameters
    dim = 3
    beta = None
    corr = -0.5
    margins = [ot.Normal()]*(dim-1) + [ot.Normal(0, 2.)]

    ## Estimation parameters
    Nv = 10000
    No = 500
    max_Ni = 10

    min_Ni = 2
    n_boot = 500
    n_run = 20
    n_Ni = 5

    ## Plot parameters
    ylim = [0, 1.0]
    alpha = [2.75, 97.5]
    figsize = (8, 5)

    ## The model
    model = AdditiveGaussian(dim=dim, beta=beta)
    model.margins = margins
    model.copula_parameters = [0., 0., corr]

    ## Initialization
    true_results = {
        'Shapley': model.shapley_indices,
        'First Sobol': model.first_order_sobol_indices,
        'Total Sobol': model.total_sobol_indices
    }

    all_Ni = np.linspace(min_Ni, max_Ni, n_Ni, dtype=int)
    n_Ni = len(all_Ni)

    all_shapley_results = np.zeros((n_Ni, n_run, dim, n_boot))
    all_first_results = np.zeros((n_Ni, n_run, dim, n_boot))
    all_total_results = np.zeros((n_Ni, n_run, dim, n_boot))

    for i_ni, Ni in enumerate(all_Ni):
        print('Ni:', Ni)
        for i_run in range(n_run):
            shapley = ShapleyIndices(model.input_distribution)
            shapley.build_mc_sample(model=model, n_perms=None, Nv=Nv, No=No, Ni=Ni)
            shapley_results = shapley.compute_indices(n_boot=n_boot)
            all_shapley_results[i_ni, i_run] = shapley_results.full_shapley_indices
            all_first_results[i_ni, i_run] = shapley_results.full_first_indices
            all_total_results[i_ni, i_run] = shapley_results.full_total_indices

    results = {
        'Shapley': all_shapley_results,
        'First Sobol': all_first_results,
        'Total Sobol': all_total_results
    }

    fig, ax = plt.subplots(figsize=figsize)
    plot_error_precision(results, all_Ni, true_results, ax=ax, ylim=ylim, alpha=alpha)
    fig.tight_layout()

    if savefigs:
        fig.savefig('./output/gaussian_precision_%d_rho_%.2f_Nv_%d_No_%d_maxNi_%d_nrun%d.pdf' % (len(results), corr, Nv, No, max_Ni, n_run))