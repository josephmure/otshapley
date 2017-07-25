import numpy as np
import itertools

class Shapley(object):
    """Shappley indices object estimator
    """
    def __init__(self, model, dim):
        self.model = model
        self.dim = dim

    def compute_indices(self):
        """
        """
        dim = self.dim
        perms = itertool.permutations(range(dim), dim)
        n_perms = len(perms)


    @property
    def dim(self):
        """
        """
        return self._dim

    @dim.setter
    def dim(self, dim):
        assert dim > 0
        self._dim = dim
        self._dim = dim