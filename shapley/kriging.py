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
        self.kriging_algo = kriging_algo
        self.kriging_result = kriging_algo.getResult()

    def kriging_function(self, X, n_realization=1):
        """
        """
        kriging_vector = ot.KrigingRandomVector(self.kriging_result, X)
        return np.asarray(kriging_vector.getSample(n_realization)).T
        
    def compute_indices(self, n_sample=200, n_realization=10, n_bootstrap=50):
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
        first_indices = np.zeros((self.dim, n_realization, n_bootstrap))

        # Loop for each indices...
        for i in range(self.dim):
            # Create the input design
            X = input_sample_1
            Xt = input_sample_2.copy()
            Xt[:, i] = X[:, i]
            input_design = np.r_[X, Xt]

            # Sample the realizations of the input design
            output_designs = self.kriging_function(input_design, n_realization=n_realization)

            # TODO: make this as a cython function
            for i_nz in range(n_realization):
                output_design = output_designs[:, i_nz]
                Y = output_design[:n_sample]
                Yt = output_design[n_sample:]
                first_indices[i, i_nz, 0] = janon_estimator(Y, Yt)
                for i_b in range(1, n_bootstrap):
                    # Bootstrap of design space
                    boot_idx = np.random.choice(range(n_sample), n_sample)
                    
                    Y_boot = Y[boot_idx]
                    Yt_boot = Yt[boot_idx]
                    # Compute the indices
                    first_indices[i, i_nz, i_b] = janon_estimator(Y_boot, Yt_boot)
                
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


def janon_estimator(Y, Yt):
    """
    """
    partial = (Y * Yt).mean() - ((Y + Yt).mean()*0.5)**2
    total = (Y**2).mean() - ((Y + Yt).mean()*0.5)**2
    return partial / total