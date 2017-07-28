import numpy as np
import pandas as pd
import openturns as ot

from .base import Base


class KrigingIndices(Base):
    """Estimate indices using a kriging based metamodel.

    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        And OpenTURNS distribution object.
    """
    def __init__(self, input_distribution):
        Base.__init__(self, input_distribution)
        
    def build_meta_model(self, model, n_sample=100, basis_type='linear', kernel='matern', sampling='lhs'):
        """Build the Kriging model.

        Parameters
        ----------
        model : callable,
            The model function to approximate.
        n_sample : int,
            The sampling size.
        basis_type : str,
            The type of basis to use for the kriging model.
        kernel : str,
            The kernel to use.
        sampling : str,
            The sampling method to use.

        Returns
        -------
        meta_model : callable,
            A stochastic function of the built kriging model.
        """
        dim = self.dim
        if basis_type == 'linear':
            basis = ot.LinearBasisFactory(dim).build()
        elif basis_type == 'constant':
            basis = ot.ConstantBasisFactory(dim).build()

        if kernel == 'matern':
            covariance = ot.MaternModel(dim)

        if sampling == 'lhs':
            lhs = ot.LHSExperiment(self._input_distribution, n_sample)
            input_sample = lhs.generate()
        elif sampling == 'monte-carlo':
            input_sample = self._input_distribution.getSample(n_sample)

        output_sample = model(input_sample).reshape(-1, 1)

        # Build the meta_model
        kriging_algo = ot.KrigingAlgorithm(input_sample, output_sample, covariance, basis)
        kriging_algo.run()
        kriging_result = kriging_algo.getResult()

        # The resulting meta_model function
        def meta_model(X, n_realization=1):
            kriging_vector = ot.KrigingRandomVector(kriging_result, X)
            output = np.asarray(kriging_vector.getSample(n_realization)).T
            return output.squeeze()

        return meta_model
        
    def build_mc_sample(self, model, n_sample=100, n_realization=10, evaluate_together=True):
        """Build the Monte-Carlo samples.

        Parameters
        ----------
        model : callable,
            The model function.
        n_sample : int,
            The sampling size of Monte-Carlo
        n_realization : int,
            The number of Gaussian Process realizations.
        evaluate_together : bool,
            If True, the GP evaluates the two input samples together.

        Return
        ------
        """
        dim = self.dim
        input_sample_1 = np.asarray(self._input_distribution.getSample(n_sample))
        input_sample_2 = np.asarray(self._input_distribution.getSample(n_sample))
        
        # The modified samples for each dimension
        all_output_sample_2 = np.zeros((n_sample, dim, n_realization))
        output_sample_1 = np.zeros((n_sample, dim, n_realization))

        X = input_sample_1
        for i in range(dim):
            Xt = input_sample_2.copy()
            Xt[:, i] = X[:, i]
            if evaluate_together:
                output_sample_i = model(np.r_[X, Xt], n_realization)
                output_sample_1[:, i, :] = output_sample_i[:n_sample, :]
                all_output_sample_2[:, i, :] = output_sample_i[n_sample:, :]
            else:
                output_sample_1[:, i, :] = model(X, n_realization)
                all_output_sample_2[:, i, :] = model(Xt, n_realization)
            
        self.output_sample_1 = output_sample_1
        self.all_output_sample_2 = all_output_sample_2

    def compute_indices(self, n_boot=50, estimator='janon', indiv_bootstraps=False):
        """Compute the indices.

        Parameters
        ----------
        n_sample : int,
            The number of sample.
        n_realization : int,
            The number of gaussian process realizations.
        n_bootstrap : int,
            The number of bootstrap samples.
        """
        dim = self.dim
        n_sample = self.output_sample_1.shape[0]
        n_realization = self.output_sample_1.shape[2]

        boot_idx = None
        first_indices = np.zeros((dim, n_realization, n_boot))
        for i in range(dim):
            if not indiv_bootstraps:
                boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))
            for i_nz in range(n_realization):
                Y = self.output_sample_1[:, i, i_nz]
                Yt = self.all_output_sample_2[:, i, i_nz]
                first_indices[i, i_nz, :] = self.first_order_indice_func(Y, Yt, n_boot=n_boot, boot_idx=boot_idx, estimator=estimator)

        return first_indices