import time

import numpy as np
import pandas as pd
import openturns as ot

from shapley.kriging import SobolIndices
from shapley.tests.test_functions import ishigami, ishigami_true_indices


def test_sobol_kriging_ishigami_independence():
    dim = 3
    model = ishigami
    margins = [ot.Uniform(-np.pi, np.pi)]*dim
    copula = ot.IndependentCopula(dim)
    input_distribution = ot.ComposedDistribution(margins, copula)
    sobol = SobolIndices(input_distribution=input_distribution)
    sobol



def test_sobol_kriging_shapley_independence():
    pass