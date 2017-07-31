import openturns as ot
import numpy as np
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt

from shapley.kriging import KrigingIndices
from shapley.tests.test_functions import ishigami, ishigami_true_indices


def test_sobol_kriging_ishigami_independence():
    dim = 3
    model = ishigami
    margins = [ot.Uniform(-np.pi, np.pi)]*dim
    copula = ot.IndependentCopula(dim)
    input_distribution = ot.ComposedDistribution(margins, copula)


def test_sobol_kriging_shapley_independence():
    pass