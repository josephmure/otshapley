import openturns as ot
import numpy as np

from .base import Base


class Indices(Base):
    """Template APIs of the sensitivity indices computation.
    """
    def __init__(self, input_distribution):
        Base.__init__(self, input_distribution)

    def build_mc_sample(self, model, n_sample):
        """Build the Monte-Carlo samples.

        Parameters
        ----------
        model : callable,
            The model function.
        n_sample : int,
            The sampling size of Monte-Carlo

        Return
        ------
        """
        dim = self.dim

        # Simulate the two independent samples
        input_sample_1 = np.asarray(self._input_distribution.getSample(n_sample))
        input_sample_2 = np.asarray(self._input_distribution.getSample(n_sample))
        
        # The modified samples for each dimension
        all_output_sample_2 = np.zeros((n_sample, dim))

        X = input_sample_1
        for i in range(dim):
            Xt = input_sample_2.copy()
            Xt[:, i] = X[:, i]
            all_output_sample_2[:, i] = model(Xt)

        self.output_sample_1 = model(input_sample_1)
        self.all_output_sample_2 = all_output_sample_2

    def build_uncorrelated_mc_sample(self, model, n_sample):
        """
        """
        dim = self.dim
        input_sample_1 = np.asarray(self._input_distribution.getSample(n_sample))
        input_sample_2 = np.asarray(self._input_distribution.getSample(n_sample))

        dist_transformation = self._input_distribution.getIsoProbabilisticTransformation()
        inv_dist_transformation = self._input_distribution.getInverseIsoProbabilisticTransformation()

        def rosenblatt_transformation(x):
            normal_transformed_x = dist_transformation(x)
            norm = ot.Normal()
            transformed_sample = np.zeros((n_sample, dim))
            for i in range(dim):
                transformed_sample[:, i] = np.asarray(norm.computeCDF(normal_transformed_x[:, i])).squeeze()
            return transformed_sample

        input_sample_1_uncorr = np.asarray(dist_transformation(input_sample_1))
        input_sample_2_uncorr = np.asarray(dist_transformation(input_sample_2))

        # The modified samples for each dimension
        all_output_sample_2 = np.zeros((n_sample, dim))

        X = input_sample_1_uncorr
        for i in range(dim):
            Xt = input_sample_2_uncorr.copy()
            Xt[:, i] = X[:, i]
            Xt = np.asarray(inv_dist_transformation(Xt))
            all_output_sample_2[:, i] = model(Xt)

        X = np.asarray(inv_dist_transformation(X))
        self.output_sample_1 = model(X)
        self.all_output_sample_2 = all_output_sample_2

    def compute_indices(self, n_boot=1, estimator='janon2'):
        """Compute the indices.

        Parameters
        ----------
        n_boot : int,
            The number of bootstrap samples.
        estimator : str,
            The type of estimator to use.
        
        Returns
        -------
        indices : list,
            The list of computed indices.
        """
        dim = self.dim
        first_indices = np.zeros((dim, n_boot))
        Y = self.output_sample_1
        for i in range(dim):
            Yt = self.all_output_sample_2[:, i]
            first_indices[i, :] = self.first_order_indice_func(Y, Yt, n_boot=n_boot, estimator=estimator)

        return first_indices
