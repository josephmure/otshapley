import numpy as np
import pandas as pd
import openturns as ot
from sklearn.gaussian_process import GaussianProcessRegressor
from .base import Base, ProbabilisticModel


class KrigingIndices(Base):
    """Estimate indices using a kriging based metamodel.

    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        And OpenTURNS distribution object.
    """
    def __init__(self, input_distribution):
        Base.__init__(self, input_distribution)
        
    def build_meta_model(self, model, n_sample=100, basis_type='linear', kernel='matern', sampling='lhs', library='OT'):
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
        meta_model = KrigingModel(model=model, input_distribution=self._input_distribution)
        meta_model.generate_sample(n_sample=n_sample, sampling=sampling)
        meta_model.build(kernel=kernel, basis_type=basis_type, library=library)
        return meta_model
        
    def build_mc_sample(self, model, n_sample=100, n_realization=10):
        """Build the Monte-Carlo samples.

        Parameters
        ----------
        model : callable,
            The model function.
        n_sample : int,
            The sampling size of Monte-Carlo
        n_realization : int,
            The number of Gaussian Process realizations.

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
            #if n_realization == 1:
            #    output_sample_i = model(np.r_[X, Xt])
            #    output_sample_1[:, i, :] = output_sample_i[:n_sample].reshape(-1, 1)
            #    all_output_sample_2[:, i, :] = output_sample_i[n_sample:].reshape(-1, 1)
            #else:
            output_sample_i = model(np.r_[X, Xt], n_realization)
            output_sample_1[:, i, :] = output_sample_i[:n_sample, :]
            all_output_sample_2[:, i, :] = output_sample_i[n_sample:, :]
            
        self.output_sample_1 = output_sample_1
        self.all_output_sample_2 = all_output_sample_2

    def compute_indices(self, n_boot=100, estimator='janon2', same_bootstrap=True):
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
            if same_bootstrap:
                boot_idx = np.random.randint(low=0, high=n_sample, size=(n_boot-1, n_sample))
            for i_nz in range(n_realization):
                Y = self.output_sample_1[:, i, i_nz]
                Yt = self.all_output_sample_2[:, i, i_nz]
                first_indices[i, i_nz, :] = self.first_order_indice_func(Y, Yt, n_boot=n_boot, boot_idx=boot_idx, estimator=estimator)

        return first_indices


class KrigingModel(ProbabilisticModel):
    """Class to build a kriging model.
    
    Parameters
    ----------
    model : callable,
        The true function.
    input_distribution : ot.DistributionImplementation,
        The input distribution for the sampling of the observations.
    """
    def __init__(self, model, input_distribution):
        self.true_model = model
        ProbabilisticModel.__init__(self, model_func=None, input_distribution=input_distribution)

    def generate_sample(self, n_sample=50, sampling='lhs'):
        """Generate the sample to build the model.

        Parameters
        ----------
        n_sample : int,
            The sampling size.
        sampling : str,
            The sampling method to use.
        """
        if sampling == 'lhs':
            lhs = ot.LHSExperiment(self._input_distribution, n_sample)
            input_sample = lhs.generate()
        elif sampling == 'monte-carlo':
            input_sample = self._input_distribution.getSample(n_sample)
        else:
            raise ValueError('Unknow sampling type {0}'.format(sampling))

        self.input_sample = np.asarray(input_sample)
        self.output_sample = self.true_model(input_sample)

    def build(self, library='sklearn', kernel='matern', basis_type='linear'):
        """Build the Kriging model.

        Parameters
        ----------
        """
        self.library = library
        if library == 'OT':
            self.covariance = kernel
            self.basis = basis_type
            kriging_algo = ot.KrigingAlgorithm(self.input_sample, self.output_sample.reshape(-1, 1), self.covariance, self.basis)
            kriging_algo.run()
            self.kriging_result = kriging_algo.getResult()

            # The resulting meta_model function
            def meta_model(X, n_realization=1):
                kriging_vector = ot.KrigingRandomVector(self.kriging_result, X)
                output = np.asarray(kriging_vector.getSample(n_realization)).T
                return output
            
            def predict(X):
                """Predict the kriging model in a deterministic way.
                """
                kriging_model = self.kriging_result.getMetaModel()
                prediction = np.asarray(kriging_model(X)).squeeze()
                return prediction
        elif library == 'sklearn':
            kriging_result = GaussianProcessRegressor()
            kriging_result.fit(self.input_sample, self.output_sample)
            self.kriging_result = kriging_result
            def meta_model(X, n_realization=1):
                return kriging_result.sample_y(X, n_samples=n_realization)
            predict = kriging_result.predict
        else:
            raise ValueError('Unknow library {0}'.format(library))

        self.predict = predict
        self.model_func = meta_model

    def __call__(self, X, n_realization=1):
        y = self._model_func(X, n_realization)
        return y

    @property
    def input_sample(self):
        """The input sample to build the model.
        """
        return self._input_sample
    
    @input_sample.setter
    def input_sample(self, sample):
        n_sample, dim = sample.shape
        assert dim == self._ndim, "Dimension should be the same as the input_distribution"
        self._n_sample = n_sample
        self._input_sample = sample

    @property
    def output_sample(self):
        """The output sample to build the model.
        """
        return self._output_sample
    
    @output_sample.setter
    def output_sample(self, sample):
        n_sample = sample.shape[0]
        assert n_sample == self._n_sample, "Samples should be the same sizes"
        self._output_sample = sample

    @property
    def covariance(self):
        """Covariance model.
        """
        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._covariance = get_covariance(covariance, self._ndim)

    @property
    def basis(self):
        """Basis model.
        """
        return self._basis

    @basis.setter
    def basis(self, basis):
        self._basis = get_basis(basis, self._ndim)

    def compute_score_q2_loo(self):
        """Leave One Out estimation of Q2.
        """
        q2 = q2_loo(self.input_sample, self.output_sample, self.basis, self.covariance)
        self.score_q2_loo = q2
        return q2

    def compute_score_q2_cv(self, n_sample=100, sampling='lhs'):
        """Cross Validation estimation of Q2.
        """
        x = self.get_input_sample(n_sample, sampling=sampling)
        ytrue = self.true_model(x)
        ypred = self.predict(x)
        q2 = q2_cv(ytrue, ypred)
        self.score_q2_cv = q2
        return q2

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


def q2_cv(ytrue, ypred):
    """Cross validation Q2 test.

    Parameters
    ----------
    ytrue : array,
        The true values.
    """
       
    ytrue = ytrue.squeeze()
    ypred = ypred.squeeze()
    q2 = max(0., test_q2(ytrue, ypred))
    return q2
def q2_loo(input_sample, output_sample, basis, covariance):
    """Leave One Out estimation of Q2.
    """
    n_sample, dim = input_sample.shape
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
    q2 = max(0., test_q2(ytrue, ypred))
    
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