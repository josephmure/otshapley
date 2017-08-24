import numpy as np
import pandas as pd
import openturns as ot
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from .base import Base, ProbabilisticModel, SensitivityResults

MAX_N_SAMPLE = 2000

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

        self._basis = None
        self._covariance = None

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

# before, the default library is OT and now and sklearn?

    def build(self, library='sklearn', kernel='matern', basis_type='linear'):
        """Build the Kriging model.

        Parameters
        ----------
        """
        self.library = library
        if library == 'OT':
            self.covariance = kernel
            self.basis = basis_type
            kriging_algo = ot.KrigingAlgorithm(self.input_sample, self.output_sample.reshape(-1, 1), self.covariance, self.basis)   ## cov and basis reversed wrt the doc
            kriging_algo.run()
            self.kriging_result = kriging_algo.getResult()

            # The resulting meta_model function
            def meta_model(X, n_realization=1):
                n_sample = X.shape[0]
                if n_sample < MAX_N_SAMPLE:
                    results = np.asarray(kriging_vector.getSample(n_realization)).T
                else:
                    state = np.random.randint(0, 1E7)
                    for max_n in range(MAX_N_SAMPLE, 1, -1):
                        if n_sample % max_n == 0:
                            print(max_n)
                            break
                    results = []
                    for i_p, X_p in enumerate(np.split(X, int(n_sample/max_n), axis=0)):
                        ot.RandomGenerator.SetSeed(state)
                        kriging_vector = ot.KrigingRandomVector(self.kriging_result, X_p)
                        results.append(np.asarray(kriging_vector.getSample(n_realization)).T)
                        print('i_p:', i_p)
                    results = np.concatenate(results)
                return results
                
            
            def predict(X):
                """Predict the kriging model in a deterministic way.
                """
                kriging_model = self.kriging_result.getMetaModel()
                prediction = np.asarray(kriging_model(X)).squeeze()
                return prediction
        elif library == 'sklearn':
            self.covariance = kernel
            kriging_result = GaussianProcessRegressor(kernel=self.covariance)
            kriging_result.fit(self.input_sample, self.output_sample)
            self.kriging_result = kriging_result

            def meta_model(X, n_realization=1):
                """
                """
                n_sample = X.shape[0]
                if n_sample < MAX_N_SAMPLE:
                    results = kriging_result.sample_y(X, n_samples=n_realization)
                else:
                    print('Sample size is too large. A loop is done to save memory.')
                    state = np.random.randint(0, 1E7)
                    
                    for max_n in range(MAX_N_SAMPLE, 1, -1):
                        if n_sample % max_n == 0:
                            print(max_n)
                            break
                    results = []
                    for i_p, X_p in enumerate(np.split(X, int(n_sample/max_n), axis=0)):
                        results.append(kriging_result.sample_y(X_p, n_samples=n_realization, random_state=state))
                        print('i_p:', i_p)
                    results = np.concatenate(results)
                return results
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
        assert dim == self._dim, "Dimension should be the same as the input_distribution"
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
        self._covariance = get_covariance(covariance, self.library, self._dim)

    @property
    def basis(self):
        """Basis model.
        """
        return self._basis

    @basis.setter
    def basis(self, basis):
        self._basis = get_basis(basis, self._dim)

    def compute_score_q2_loo(self):
        """Leave One Out estimation of Q2.
        """
        q2 = q2_loo(self.input_sample, self.output_sample, self.library, self.covariance, self.basis)
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
    q2 = max(0., test_q2(ytrue, ypred))                     ## thus useless ?
    return q2

def q2_loo(input_sample, output_sample, library, covariance, basis=None):
    """Leave One Out estimation of Q2.
    """
    n_sample, dim = input_sample.shape
    ypred = np.zeros((n_sample, ))
    
    for i in range(n_sample):
        xi = input_sample[i, :]
        input_sample_i = np.delete(input_sample, i, axis=0)
        output_sample_i = np.delete(output_sample, i, axis=0).reshape(-1, 1)
        
        if library == 'OT':
            kriging_algo = ot.KrigingAlgorithm(input_sample_i, output_sample_i, covariance, basis)
            kriging_algo.run()
            meta_model_mean = kriging_algo.getResult().getMetaModel()
        elif library == 'sklearn':
            kriging_result = GaussianProcessRegressor(kernel=covariance)
            kriging_result.fit(input_sample_i, output_sample_i.ravel())
            meta_model_mean = kriging_result.predict
        else:
            raise ValueError('Unknow library {0}'.format(library))
        
        ypred[i] = np.asarray(meta_model_mean(xi.reshape(1, -1)))
        
    ytrue = output_sample.squeeze()

    q2 = test_q2(ytrue, ypred)
    
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


def get_covariance(kernel, library, dim=None):
    """
    """
    if isinstance(kernel, ot.CovarianceModelImplementation):
        covariance = kernel
    else:
        if library == 'OT':
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
                raise ValueError('Unknow kernel {0} for library {1}'.format(kernel, library))
        elif library == 'sklearn':
            if kernel == 'matern':
                covariance = kernels.Matern()
            elif kernel == 'exponential':
                covariance = kernels.ExpSineSquared()
            elif kernel == 'RBF':
                covariance = kernels.RBF()
            else:
                raise ValueError('Unknow kernel {0} for library {1}'.format(kernel, library))
        else:
            raise ValueError('Unknow library {0}'.format(library))
    return covariance