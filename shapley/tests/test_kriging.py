import openturns as ot
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt

from shapley.kriging import KrigingIndices, compute_indices, create_df_from_gp_indices, create_df_from_indices
from shapley.tests.test_functions import ishigami, ot_ishigami, ishigami_true_indices
from shapley.plots import set_style_paper


def test_sobol_kriging_ishigami_independence():
    dim = 3
    model = ishigami
    margins = [ot.Uniform(-np.pi, np.pi)]*dim
    copula = ot.IndependentCopula(dim)
    input_distribution = ot.ComposedDistribution(margins, copula)


def test_sobol_kriging_shapley_independence():
    pass