import numpy as np
import pandas as pd
import openturns as ot


class Indices(object):
    """Template APIs of the sensitivity indices computation.
    """
    def __init__(self, input_distribution):
        self.input_distribution = input_distribution
        self.dim = input_distribution.getDimension()
        self.first_order_indice_func = None

    def build_mc_sample(self, n_sample, model, n_realization=1):
        """Build the Monte-Carlo samples.
        """
        dim = self.dim
        input_sample_1 = np.asarray(self.input_distribution.getSample(n_sample))
        input_sample_2 = np.asarray(self.input_distribution.getSample(n_sample))
        
        # The modified samples for each dimension
        all_input_sample_2 = np.zeros((dim, n_sample, dim))
        all_output_sample_2 = np.zeros((n_sample, dim))
        for i in range(dim):
            Xt = input_sample_2.copy()
            Xt[:, i] = input_sample_1[:, i]
            Yt = model(Xt)
            all_input_sample_2[:, :, i] = Xt.T
            all_output_sample_2[:, i] = Yt
        
        self.output_sample_1 = model(input_sample_1)
        self.all_output_sample_2 = all_output_sample_2

    def compute_indices(self, n_boot=1, estimator='janon'):
        """Compute the indices
        """
        dim = self.dim
        first_indices = np.zeros((dim, n_boot))
        Y = self.output_sample_1
        for i in range(dim):
            Yt = self.all_output_sample_2[:, i]
            first_indices[i, :] = self.first_order_indice_func(Y, Yt, n_boot=n_boot, estimator=estimator)

        return first_indices


class SobolIndices(Indices):
    """
    """
    def __init__(self, input_distribution):
        super(self.__class__, self).__init__(input_distribution)
        self.first_order_indice_func = first_order_sobol_indice


class KrigingIndices(object):
    """Estimate indices using a kriging based metamodel.
    """
    def __init__(self, input_distribution):
        self.input_distribution = input_distribution
        self.dim = input_distribution.getDimension()
        
    def build_meta_model(self, model, n_sample_kriging=100, basis_type='linear', kernel='matern', sampling='lhs'):
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

        output_sample = model(input_sample).reshape(-1, 1)

        # Build the meta_model
        kriging_algo = ot.KrigingAlgorithm(input_sample, output_sample, covariance, basis)
        kriging_algo.run()
        kriging_result = kriging_algo.getResult()

        # The resulting meta_model function
        def meta_model(X, n_realization=1):
            kriging_vector = ot.KrigingRandomVector(kriging_result, X)
            output = np.asarray(kriging_vector.getSample(n_realization)).T
            return output if n_realization > 1 else output.ravel()

        return meta_model
        
    def compute_indices(self, n_boot=50, n_realization=1, estimator='janon'):
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
        first_indices = np.zeros((dim, n_realization, n_boot))
        Y = self.output_sample_1
        for i in range(dim):
            Yt = self.all_output_sample_2[:, i]
            n_sample = Yt.shape[0]
            for i_nz in range(n_realization):
                boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))
                first_indices[i, i_nz, :] = self.first_order_indice_func(Y, Yt, n_boot=n_boot, boot_idx=boot_idx, estimator=estimator)

        return first_indices


class SobolKrigingIndices(KrigingIndices, SobolIndices):
    """
    """
    def __init__(self, input_distribution):
        KrigingIndices.__init__(self, input_distribution)
        SobolIndices.__init__(self, input_distribution)


def first_order_sobol_indices(output_sample_1, all_output_sample_2, n_boot=1, boot_idx=None, estimator='janon'):
    """
    """
    dim = all_output_sample_2.shape[1]
    first_indices = np.zeros((dim, n_boot))
    Y = output_sample_1
    for i in range(dim):
        Yt = all_output_sample_2[:, i]
        first_indices[i, :] = first_order_sobol_indice(Y, Yt, n_boot=n_boot)
    return first_indices
    
def first_order_sobol_indice(Y, Yt, n_boot=1, boot_idx=None, estimator='janon'):
    """Compute the Sobol indices from the to

    Parameters
    ----------
    """
    n_sample = Y.shape[0]
    assert n_sample == Yt.shape[0], "Matrices should have the same sizes"

    if estimator == 'janon':
        estimator = janon_estimator

    first_indice = np.zeros((n_boot, ))
    first_indice[0] = estimator(Y, Yt)
    if boot_idx is None:
        boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))
    first_indice[1:] = estimator(Y[boot_idx], Yt[boot_idx])

    return first_indice if n_boot > 1 else first_indice.item()


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