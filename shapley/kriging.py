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
        basis_type : str or ot.CovarianceModelImplementation,
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
        basis = get_basis(basis_type, dim)
        covariance = get_covariance(kernel, dim)

        if sampling == 'lhs':
            lhs = ot.LHSExperiment(self._input_distribution, n_sample)
            input_sample = lhs.generate()
        elif sampling == 'monte-carlo':
            input_sample = self._input_distribution.getSample(n_sample)
        else:
            raise ValueError('Unknow sampling type {0}'.format(sampling))

        output_sample = model(input_sample).reshape(-1, 1)

        # Build the meta_model
        kriging_algo = ot.KrigingAlgorithm(input_sample, output_sample, covariance, basis)
        kriging_algo.run()
        kriging_result = kriging_algo.getResult()
        self.kriging_algo = kriging_algo
        self.kriging_result = kriging_result
        self.meta_model_mean = kriging_result.getMetaModel()

        # The resulting meta_model function
        def meta_model(X, n_realization=1):
            kriging_vector = ot.KrigingRandomVector(kriging_result, X)
            output = np.asarray(kriging_vector.getSample(n_realization)).T
            return output.squeeze()
        
        self.meta_model_input_sample = np.asarray(input_sample)
        self.meta_model_output_sample = output_sample.squeeze()

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

    def compute_indices(self, n_boot=100, estimator='janon2', indiv_bootstraps=False):
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



def test_q2(ytrue, ypred):
    """Compute the Q2 test.

    Parameters
    ----------
    ytrue : array,
        The true output values.
    ypred : array,
        The predicted output values.

    Returns
    -------
    q2 : float,
        The estimated Q2.
    """
    ymean = ytrue.mean()
    up = ((ytrue - ypred)**2).sum()
    down = ((ytrue - ymean)**2).sum()
    q2 = 1. - up / down
    return q2


def get_basis(basis_type, dim):
    """
    """
    if basis_type == 'constant':
        basis = ot.ConstantBasisFactory(dim).build()
    elif basis_type == 'linear':
        basis = ot.LinearBasisFactory(dim).build()
    elif basis_type == 'quadratic':
        basis = ot.QuadraticBasisFactory(dim).build()
    else:
        raise ValueError('Unknow basis type {0}'.format(basis_type))

    return basis


def compute_q2_cv(model, meta_model, input_distribution, n_sample=100, sampling='lhs'):
    """Cross validation Q2 test.
    """
    if sampling =='lhs':
        lhs = ot.LHSExperiment(input_distribution, n_sample)
        val_input_sample = lhs.generate()
    elif sampling == 'monte-carlo':
        val_input_sample = input_distribution.getSample(n_sample)
        
    val_output_sample = model(val_input_sample).reshape(-1, 1)

    ytrue = val_output_sample.squeeze()
    ypred = np.asarray(meta_model(val_input_sample)).squeeze()
    
    q2 = test_q2(ytrue, ypred)
    return q2


def compute_q2_loo(input_sample, output_sample, basis_type='linear', kernel='matern'):
    """Leave One Out estimation of Q2.
    """
    n_sample, dim = input_sample.shape
    basis = get_basis(basis_type, dim)
    covariance = get_covariance(kernel, dim)
    
    ypred = np.zeros((n_sample, ))
    
    for i in range(n_sample):
        xi = input_sample[i, :]
        input_sample_i = np.delete(input_sample, i, axis=0)
        output_sample_i = np.delete(output_sample, i, axis=0).reshape(-1, 1)
        
        kriging_algo = ot.KrigingAlgorithm(input_sample_i, output_sample_i, covariance, basis)
        kriging_algo.run()
        meta_model_mean = kriging_algo.getResult().getMetaModel()
        
        ypred[i] = np.asarray(meta_model_mean(xi))

    ytrue = output_sample.squeeze()
    q2 = test_q2(ytrue, ypred)
    
    return q2


def get_covariance(kernel, dim=None):
    """
    """
    if isinstance(kernel, ot.CovarianceModelImplementation):
        covariance = kernel
    else:
        assert dim is not None, "Dimension should be given"
        if kernel == 'matern':
            covariance = ot.MaternModel(dim)
        elif kernel == 'exponential':
            covariance = ot.ExponentialModel(dim)
        elif kernel == 'generalized-exponential':
            covariance = ot.GeneralizedExponential(dim)
        elif kernel == 'spherical':
            covariance = ot.SphericalModel(dim)
        else:
            raise ValueError('Unknow kernel {0}'.format(kernel))

    return covariance