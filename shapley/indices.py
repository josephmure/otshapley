import openturns as ot
import numpy as np

from .base import Base, SensitivityResults


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

        X1 = input_sample_1
        X2 = input_sample_2
        for i in range(dim):
            X2t = X2.copy()
            X2t[:, i] = X1[:, i]

            all_output_sample_2[:, i] = model(X2t)

        self.output_sample_1 = model(input_sample_1)
        self.output_sample_2 = model(input_sample_2)
        self.all_output_sample_2 = all_output_sample_2

    def compute_indices(self, n_boot=100, estimator='soboleff2'):
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
        total_indices = np.zeros((dim, n_boot))
        Y1 = self.output_sample_1
        Y2 = self.output_sample_2
        for i in range(dim):
            Y2t = self.all_output_sample_2[:, i]
            first_indices[i, :], total_indices[i, :] = self.indice_func(Y1, Y2, Y2t, n_boot=n_boot, estimator=estimator)

        results = SensitivityResults(first_indices=first_indices, total_indices=total_indices, calculation_method='monte-carlo')
        return results

    def build_uncorrelated_mc_sample(self, model, n_sample):
        """
        """
        dim = self.dim

        # Normal distribution
        norm_dist = ot.Normal(dim)

        # Independent samples
        U_1 = np.asarray(norm_dist.getSample(n_sample))
        U_2 = np.asarray(norm_dist.getSample(n_sample))

        # The modified samples for each dimension
        all_output_sample_1 = np.zeros((n_sample, dim))
        all_output_sample_2 = np.zeros((n_sample, dim))
        all_output_sample_2t = np.zeros((n_sample, dim))
        all_output_sample_2t1 = np.zeros((n_sample, dim))

        for i in range(dim):
            # Copy of the input dstribution
            margins = [ot.Distribution(self._input_distribution.getMarginal(i)) for i in range(dim)]
            copula = ot.Copula(self._input_distribution.getCopula())

            # 1) Pick and Freeze
            U_3_i = U_2.copy()
            U_3_i[:, 0] = U_1[:, 0]
            U_4_i = U_2.copy()
            U_4_i[:, -1] = U_1[:, -1]
            
            # 2) Permute the margins and the copula
            order_i = np.roll(range(dim), -i)
            order_cop = np.roll(range(dim), i)
            margins_i = [margins[j] for j in order_i]
            params_i = np.asarray(copula.getParameter())[order_cop]

            copula.setParameter(params_i)
            dist = ot.ComposedDistribution(margins_i, copula)

            # 3) Inverse Transformation
            tmp = dist.getInverseIsoProbabilisticTransformation()
            inv_rosenblatt_transform_i = lambda u: np.asarray(tmp(u))

            X_1_i = inv_rosenblatt_transform_i(U_1)
            X_2_i = inv_rosenblatt_transform_i(U_2)
            X_3_i = inv_rosenblatt_transform_i(U_3_i)
            X_4_i = inv_rosenblatt_transform_i(U_4_i)
            assert X_1_i.shape[1] == dim, "Wrong dimension"

            # 4) Model evaluations
            all_output_sample_1[:, i] = model(X_1_i)
            all_output_sample_2[:, i] = model(X_2_i)
            all_output_sample_2t[:, i] = model(X_3_i)
            all_output_sample_2t1[:, i] = model(X_4_i)

        self.all_output_sample_1 = all_output_sample_1
        self.all_output_sample_2 = all_output_sample_2
        self.all_output_sample_2t = all_output_sample_2t
        self.all_output_sample_2t1 = all_output_sample_2t1
        
    def compute_full_indices(self, n_boot=1, estimator='soboleff2'):
        """
        """
        dim = self.dim
        first_indices = np.zeros((dim, n_boot))
        total_indices = np.zeros((dim, n_boot))
        for i in range(dim):
            Y1 = self.all_output_sample_1[:, i]
            Y2 = self.all_output_sample_2[:, i]
            Y2t = self.all_output_sample_2t[:, i]
            first_indices[i, :], total_indices[i, :] = self.indice_func(Y1, Y2, Y2t, n_boot=n_boot, estimator=estimator)

        results = SensitivityResults(first_indices=first_indices, total_indices=total_indices, calculation_method='monte-carlo')
        return results

    def compute_ind_indices(self, n_boot=1, estimator='soboleff2'):
        """
        """
        dim = self.dim
        first_indices = np.zeros((dim, n_boot))
        total_indices = np.zeros((dim, n_boot))
        for i in range(dim):
            Y1 = self.all_output_sample_1[:, i]
            Y2 = self.all_output_sample_2[:, i]
            Y2t = self.all_output_sample_2t1[:, i]
            first_indices[i-1, :], total_indices[i-1, :] = self.indice_func(Y1, Y2, Y2t, n_boot=n_boot, estimator=estimator)

        results = SensitivityResults(first_indices=first_indices, total_indices=total_indices, calculation_method='monte-carlo')
        return results
