import numpy as np
import itertools

from .indices import Indices
from .kriging import KrigingIndices


class ShapleyIndices(Indices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        super(self.__class__, self).__init__(input_distribution)


class ShapleyKrigingIndices(KrigingIndices, ShapleyIndices):
    """Shappley indices object estimator.
    """
    def __init__(self, input_distribution):
        KrigingIndices.__init__(self, input_distribution)
        ShapleyIndices.__init__(self, input_distribution)