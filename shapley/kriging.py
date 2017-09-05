import numpy as np
import pandas as pd
import openturns as ot
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from .indices import BaseIndices, SensitivityResults
from .model import ProbabilisticModel, MetaModel
from .utils import test_q2

MAX_N_SAMPLE = 15000

class KrigingIndices(BaseIndices):
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


class KrigingModel(MetaModel):
    """Class to build a kriging model.
    
    Parameters
    ----------
    model : callable,
        The true function.
    input_distribution : ot.DistributionImplementation,
        The input distribution for the sampling of the observations.
    """
    def __init__(self, model, input_distribution):
        MetaModel.__init__(self, model=model, input_distribution=input_distribution)

        self._basis = None
        self._covariance = None
        self._input_sample = None
        self._output_sample = None

    def build(self, library='sklearn', kernel='matern', basis_type='linear'):
        """Build the Kriging model.

        Parameters
        ----------
        library: str,
            The used library to build the metamodel.
        """
        assert self.input_sample is not None, "No input sample given"
        assert self.output_sample is not None, "No output sample given"
        self.library = library
        if library == 'OT':
            self.covariance = kernel
            self.basis = basis_type
            kriging_algo = ot.KrigingAlgorithm(self.input_sample, self.output_sample.reshape(-1, 1), self.covariance, self.basis)
            kriging_algo.run()
            self.kriging_result = kriging_algo.getResult()

            # The resulting meta_model function
            def meta_model(X, n_realization):
                n_sample = X.shape[0]
                if n_sample <= MAX_N_SAMPLE:
                    kriging_vector = ot.KrigingRandomVector(self.kriging_result, X)
                    results = np.asarray(kriging_vector.getSample(n_realization)).T
                else:
                    state = np.random.randint(0, 1E7)
                    for max_n in range(MAX_N_SAMPLE, 1, -1):
                        if n_sample % max_n == 0:
                            break
                    results = []
                    print("%d splits of size %d" % (int(n_sample/max_n), max_n))
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
                            break
                    results = []
                    print("%d splits of size %d" % (int(n_sample/max_n), max_n))
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