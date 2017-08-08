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
        for i in range(dim):
            X2t = input_sample_2.copy()
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

        margins = [ot.Distribution(self._input_distribution.getMarginal(i)) for i in range(dim)]
        copula = ot.Copula(self._input_distribution.getCopula())

        # Independent samples
        U_1 = np.asarray(norm_dist.getSample(n_sample))
        U_2 = np.asarray(norm_dist.getSample(n_sample))

        # The modified samples for each dimension
        all_output_sample_1 = np.zeros((n_sample, dim))
        all_output_sample_2 = np.zeros((n_sample, dim))
        all_output_sample_3 = np.zeros((n_sample, dim))
        all_output_sample_4 = np.zeros((n_sample, dim))

        for i in range(dim):
            # 1) Permutations
            U_1_i = U_1
            U_2_i = U_2

            # 2) Pick and Freeze
            U_3_i = U_2_i.copy()
            U_3_i[:, 0] = U_1_i[:, 0]
            U_4_i = U_2_i.copy()
            U_4_i[:, -1] = U_1_i[:, -1]

            order_i = np.roll(range(dim), -i)

            margins_i = [margins[j] for j in order_i]
            params = np.asarray(copula.getParameter())
            if i == 0:
                order_cop = [0, 1, 2]
            elif i == 2:
                order_cop = [2, 0, 1]
            else:
                order_cop = [2, 0, 1]

            copula.setParameter(params[order_cop])
            dist = ot.ComposedDistribution(margins_i, copula)

            # Inverse rosenblatt distribution
            inv_dist_transformation = dist.getInverseIsoProbabilisticTransformation()
            inv_rosenblatt_transform = lambda u: np.asarray(inv_dist_transformation(u))

            # 3) Inverse Transformation
            X_1_i = inv_rosenblatt_transform(U_1_i)
            X_2_i = inv_rosenblatt_transform(U_2_i)
            X_3_i = inv_rosenblatt_transform(U_3_i)
            X_4_i = inv_rosenblatt_transform(U_4_i)

            order_i = list()
            if i > 0:
                order_i = list(range(dim-i, dim))

            order_i += list(range(dim-i))
            print(order_i)
            
            X_1_i = X_1_i[:, order_i]
            X_2_i = X_2_i[:, order_i]
            X_3_i = X_3_i[:, order_i]
            X_4_i = X_4_i[:, order_i]

            assert X_1_i.shape[1] == dim, "Wrong dimension"

            # 4) Model evaluations
            all_output_sample_1[:, i] = model(X_1_i)
            all_output_sample_2[:, i] = model(X_2_i)
            all_output_sample_3[:, i] = model(X_3_i)
            all_output_sample_4[:, i] = model(X_4_i)

        self.all_output_sample_1 = all_output_sample_1
        self.all_output_sample_2 = all_output_sample_2
        self.all_output_sample_3 = all_output_sample_3
        self.all_output_sample_4 = all_output_sample_4

    def compute_uncorrelated_indices(self, n_boot=1, estimator='soboleff2'):
        """
        """
        dim = self.dim
        first_indices = np.zeros((dim, n_boot))
        total_indices = np.zeros((dim, n_boot))
        for i in range(dim):
            Y1 = self.all_output_sample_1[:, i]
            Y2 = self.all_output_sample_2[:, i]
            Y2t = self.all_output_sample_3[:, i]
            first_indices[i, :], total_indices[i, :] = self.indice_func(Y1, Y2, Y2t, n_boot=n_boot, estimator=estimator)

        results = SensitivityResults(first_indices=first_indices, total_indices=total_indices, calculation_method='monte-carlo')
        return results