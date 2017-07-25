import numpy as np
import openturns as ot


class KrigingIndices(object):
    """Estimate the shaplay and sobol indices using a kriging based metamodel.
    """
    def __init__(self, model, input_distribution):
        self.model = model
        self.input_distribution = input_distribution
        self.dim = input_distribution.getDimension()
        
    def build_model(self, n_sample_kriging=100, basis_type='linear', kernel='matern'):
        """Build the Kriging model
        """
        dim = self.dim
        input_sample = self.input_distribution.getSample(n_sample_kriging)
        output_sample = self.model(input_sample)
        
        if basis_type == 'linear':
            basis = ot.LinearBasisFactory(dim).build()
            
        if kernel == 'matern':
            covariance = ot.MaternModel(dim)
            
        kriging_algo = ot.KrigingAlgorithm(input_sample, output_sample, covariance, basis)
        kriging_algo.run()
        self.kriging_result = kriging_algo.getResult()

    def kriging_function(self, X):
        """
        """
        return ot.KrigingRandomVector(self.kriging_result, X)
        
    def compute_indices(self, n_sample=200, n_realization=10, n_bootstrap=50, with_second_order=False):
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
        input_sample_1 = np.asarray(self.input_distribution.getSample(n_sample))
        input_sample_2 = np.asarray(self.input_distribution.getSample(n_sample))

        for i in range(self.dim):
            X = input_sample_1
            Xt = input_sample_2.copy()
            Xt[:, i] = input_sample_1[:, i]
            input_design = np.r_[X, Xt]
            output_design = np.r_[Y, Xt]
            indices[i] = janon_estimator(self.kriging_function, X, Xt)

            kriging_vector = self.kriging_function(input_design)
            output_designs = np.asarray(kriging_vector.getSample(n_realization)).T
        
            first_indices = np.zeros((self.dim, n_realization, n_bootstrap))
            n_design = input_design.shape[0]
            for i_nz in range(n_realization):
                output_design = output_designs[:, i_nz].reshape(-1, 1)
                sensitivity_normal = ot.SaltelliSensitivityAlgorithm(input_design, output_design, n_sample)
                first_indices[:, i_nz, 0] = np.asarray(sensitivity_normal.getFirstOrderIndices())
                for i_b in range(1, n_bootstrap):
                    # Bootstrap of design space
                    boot_idx = np.random.choice(range(n_design), n_design)
                    input_design_boot = input_design[boot_idx]
                    output_design_boot = output_design[boot_idx]

                    # Compute the indices
                    sensitivity_boot = ot.SaltelliSensitivityAlgorithm(input_design_boot, output_design_boot, n_sample)
                    first_indices[:, i_nz, i_b] = np.asarray(sensitivity_boot.getFirstOrderIndices())
                
        return first_indices


    
def compute_indices(func, input_sample_1, input_sample_2):
    """
    """
    indices = np.zeros((dim,))
    for i in range(dim):
        X = input_sample_1
        Xt = input_sample_2.copy()
        Xt[:, i] = input_sample_1[:, i]
        indices[i] = janon_estimator(func, X, Xt)
    return indices