import numpy as np
import pandas as pd
import openturns as ot


class KrigingIndices(object):
    """Estimate the shaplay and sobol indices using a kriging based metamodel.
    """
    def __init__(self, model, input_distribution):
        self.model = model
        self.input_distribution = input_distribution
        self.dim = input_distribution.getDimension()
        
    def build_model(self, n_sample_kriging=100, basis_type='linear', kernel='matern', sampling='lhs'):
        """Build the Kriging model.
        """
        dim = self.dim
        if basis_type == 'linear':
            basis = ot.LinearBasisFactory(dim).build()
        elif basis_type == 'constant':
            basis = ot.ConstantBasisFactory(dim).build()
        if kernel == 'matern':
            covariance = ot.MaternModel(dim)

        if sampling == 'lhs':
            lhs = ot.LHSExperiment(self.input_distribution, n_sample_kriging)
            input_sample = lhs.generate()
        elif sampling == 'monte-carlo':
            input_sample = self.input_distribution.getSample(n_sample_kriging)

        output_sample = self.model(input_sample)

        kriging_algo = ot.KrigingAlgorithm(input_sample, output_sample, covariance, basis)
        kriging_algo.run()
        self.kriging_algo = kriging_algo
        self.kriging_result = kriging_algo.getResult()

    def kriging_function(self, X, n_realization=1):
        """
        """
        kriging_vector = ot.KrigingRandomVector(self.kriging_result, X)
        return np.asarray(kriging_vector.getSample(n_realization)).T
        
    def compute_indices(self, n_sample=200, n_realization=10, n_boot=50):
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
        first_indices = np.zeros((self.dim, n_realization, n_boot))

        # Loop for each indices...
        for i in range(self.dim):
            # Create the input design
            X = input_sample_1
            Xt = input_sample_2.copy()
            Xt[:, i] = X[:, i]
            input_design = np.r_[X, Xt]

            # Sample the realizations of the input design
            output_designs = self.kriging_function(input_design, n_realization=n_realization)

            boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))
            # TODO: make this as a cython function
            for i_nz in range(n_realization):
                output_design = output_designs[:, i_nz]
                Y = output_design[:n_sample]
                Yt = output_design[n_sample:]
                first_indices[i, i_nz, :] = compute_indice(Y, Yt, n_boot=n_boot, boot_idx=boot_idx)
                
        return first_indices

def compute_indices(func, input_sample_1, input_sample_2, n_boot=1):
    """
    """
    dim = input_sample_1.shape[1]
    first_indices = np.zeros((dim, n_boot))
    for i in range(dim):
        X = input_sample_1
        Xt = input_sample_2.copy()
        Xt[:, i] = X[:, i]
        Y = func(X)
        Yt = func(Xt)
        first_indices[i, :] = compute_indice(Y, Yt, n_boot=n_boot)
    return first_indices
    
def compute_indice(Y, Yt, n_boot=1, boot_idx=None):
    """
    """
    n_sample = Y.shape[0]
    assert n_sample == Yt.shape[0], "Matrices should have the same sizes"

    first_indice = np.zeros((n_boot, ))
    first_indice[0] = janon_estimator(Y, Yt)
    if boot_idx is None:
        boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))
    first_indice[1:] = janon_estimator(Y[boot_idx], Yt[boot_idx])

    if n_boot == 1:
        return first_indice.item()
    else:
        return first_indice


def janon_estimator(Y, Yt):
    """
    """
    if Y.ndim == 1:
        Y = Y.reshape(1, -1)
        Yt = Yt.reshape(1, -1)

    partial = (Y * Yt).mean(axis=1) - ((Y + Yt).mean(axis=1)*0.5)**2
    total = (Y**2).mean(axis=1) - ((Y + Yt).mean(axis=1)*0.5)**2
    return partial / total


def create_df_from_gp_indices(first_indices, mean_method=True):
    """
    """
    dim, n_realization, n_boot = first_indices.shape
    columns = ['S_%d' % (i+1) for i in range(dim)]
    if mean_method:
        df1 = pd.DataFrame(first_indices.mean(axis=2).T, columns=columns)
        df2 = pd.DataFrame(first_indices.mean(axis=1).T, columns=columns)
    else:
        df1 = pd.DataFrame(first_indices[:, :, 0].T, columns=columns)
        df2 = pd.DataFrame(first_indices[:, 0, :].T, columns=columns)

    df = pd.concat([df1, df2])
    df['Error'] = pd.DataFrame(['Kriging error']*n_realization + ['MC error']*n_boot)
    df = pd.melt(df, id_vars=['Error'], value_vars=columns, var_name='Variables', value_name='Indice values')
    return df

def create_df_from_indices(first_indices):
    """
    """
    dim, n_boot = first_indices.shape
    columns = ['S_%d' % (i+1) for i in range(dim)]
    df = pd.DataFrame(first_indices.T, columns=columns)
    df = pd.melt(df, value_vars=columns, var_name='Variables', value_name='Indice values')
    return df