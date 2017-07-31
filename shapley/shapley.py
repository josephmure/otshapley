import numpy as np
import itertools

from .indices import Indices
from .kriging import KrigingIndices


class ShapleyIndices(Indices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        Indices.__init__(self, input_distribution)
        # Add shapley function
        self.first_order_indice_func = None


class ShapleyKrigingIndices(KrigingIndices, ShapleyIndices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        KrigingIndices.__init__(self, input_distribution)
        ShapleyIndices.__init__(self, input_distribution)