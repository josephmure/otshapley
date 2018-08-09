import numpy as np
import openturns as ot
import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
try:
    import gpflow
    from gpflow.params import Parameter
    from gpflow.mean_functions import MeanFunction
    from gpflow.decors import params_as_tensors
    from gpflow import settings
except:
    print('Could not load gpflow')

from .model import MetaModel
from .utils import test_q2

MAX_N_SAMPLE = 15000
MAXITER = 10000


class QuadraticTrendGPflow(MeanFunction):
    """
    Class to calculate a mean quadratic function for the trend of the process
    using the gpflow package
    y_i = A x_i + b
    """
    def __init__(self, A=None, b=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be 2*D x Q, b must be a vector of length Q.
        """
        A = np.ones((2, 1)) if A is None else A
        b = np.zeros(1) if b is None else b
        MeanFunction.__init__(self)
        self.A = Parameter(np.atleast_2d(A), dtype = settings.float_type)
        self.b = Parameter(b, dtype = settings.float_type)

    @params_as_tensors
    def __call__(self, X):
        X = tf.concat([X, tf.pow(X,2)], 1)
        return tf.matmul(X, self.A) + self.b


class KrigingModel(MetaModel):
    """Class to build a kriging model.
    
    This class generate a Gaussian Process.
    
    Parameters
    ----------
    model : callable,
        The true function.
    input_distribution : ot.DistributionImplementation,
        The input distribution for the sampling of the observations.
    """
    def __init__(self, model=None, input_distribution=None):
        MetaModel.__init__(self, model=model, input_distribution=input_distribution)
        self._basis = None
        self._covariance = None
        self._input_sample = None
        self._output_sample = None

    def build(self, library='gpflow', kernel='matern', basis_type='linear'):
        """Build the Kriging model.

        Parameters
        ----------
        library: str, optional (default='gpflow')
            The used library to build the meta-model. The available libraries
            are :
            - "gpflow": a binding between GPy and tensorflow,
            - "sklearn": the classical scikit-learn library,
            - "OT": the OpenTURNS library.
            
        kernel: str, optional (default='matern')
            The kernel covariance of the Gaussian Process. The possible kernels
            are :
            - "matern": a matern 5/2.
        """
        assert self.input_sample is not None, "No input sample were given"
        assert self.output_sample is not None, "No output sample were given"
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
                if n_sample > MAX_N_SAMPLE:
                    raise MemoryError('Sample size is too large.')
                else:
                    kriging_vector = ot.KrigingRandomVector(self.kriging_result, X)
                    results = np.asarray(kriging_vector.getSample(n_realization)).T
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

            def meta_model(X, n_realization):
                """
                """
                n_sample = X.shape[0]
                if n_sample > MAX_N_SAMPLE:
                    raise MemoryError('Sample size is too large.')
                else:
                    results = kriging_result.sample_y(X, n_samples=n_realization)
                    
                return results
            
            predict = kriging_result.predict
        elif library == 'gpflow':
            self.covariance = kernel
            self.basis = basis_type
            gp = gpflow.models.GPR(
                self.input_sample, self.output_sample[:, np.newaxis], kern=self.covariance, mean_function=self.basis)
            gp.likelihood.variance = 1.E-6
            gp.likelihood.trainable = True
            gp.compile()
            gpflow.train.ScipyOptimizer().minimize(gp, maxiter = MAXITER)

            def meta_model(X, n_realization):
                """
                """
                results = gp.predict_f_samples(X, n_realization)
                return results.squeeze().T

            def predict(X):
                """
                """
                results = gp.predict_y(X)[0]
                return results.squeeze()
            self.gp_qflow = gp
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
        self._basis = get_basis(basis, self._dim, self.library)

    def compute_score_q2_loo(self):
        """Leave One Out estimation of Q2.
        """
        q2 = q2_loo(self.input_sample, self.output_sample, self.library, self.covariance, self.basis)
        self.score_q2_loo = q2
        return q2

    def __call__(self, X, n_realization=1):
        if n_realization == 1:
            y = self.predict(X)
        else:
            y = self._model_func(X, n_realization)
        return y

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
        elif library == 'gpflow':
            gp = gpflow.models.GPR(input_sample_i, output_sample_i.reshape(-1, 1), covariance, basis)
            gp.likelihood.variance = 1.E-6
            gp.optimize()
            meta_model_mean = lambda X: gp.predict_y(X)[0].squeeze()
        else:
            raise ValueError('Unknow library {0}'.format(library))
        
        ypred[i] = np.asarray(meta_model_mean(xi.reshape(1, -1)))
        
    ytrue = output_sample.squeeze()

    q2 = test_q2(ytrue, ypred)
    
    return q2

def get_basis(basis_type, dim, library):
    """
    """
    if library == 'OT':
        if basis_type == 'constant':
            basis = ot.ConstantBasisFactory(dim).build()
        elif basis_type == 'linear':
            basis = ot.LinearBasisFactory(dim).build()
        elif basis_type == 'quadratic':
            basis = ot.QuadraticBasisFactory(dim).build()
        else:
            raise ValueError('Unknow basis type {0}'.format(basis_type))
    elif library == 'gpflow':
        if basis_type == 'constant':
            basis = gpflow.mean_functions.Constant()
        elif basis_type == 'linear':
            basis = gpflow.mean_functions.Linear(np.ones((dim, 1)), 0.)
        elif basis_type == 'quadratic':
            basis = QuadraticTrendGPflow(np.ones((2*dim, 1)), 0.)
        elif basis_type == 'sum-product':
            const1_1 = gpflow.mean_functions.Constant(1)
            const1_2 = gpflow.mean_functions.Constant(1)
            const2_1 = gpflow.mean_functions.Constant(1)
            const2_2 = gpflow.mean_functions.Constant(1)
            
            basis = gpflow.mean_functions.Additive(
                gpflow.mean_functions.Product(const1_1, const2_1),
                gpflow.mean_functions.Product(const1_2, const2_2))

        else:
            raise ValueError('Unknow basis type {0}'.format(basis_type))
        return basis
    else:
        raise ValueError('Unknow library {0}'.format(library))


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
        elif library == 'gpflow':
            if kernel == 'matern':
                covariance = gpflow.kernels.Matern52(dim, ARD = True)
            elif kernel == 'exponential':
                covariance = gpflow.kernels.Exponential(dim, ARD = True)
            elif kernel == 'RBF':
                covariance = gpflow.kernels.RBF(dim, ARD = True)
            else:
                raise ValueError('Unknow kernel {0} for library {1}'.format(kernel, library))
        else:
            raise ValueError('Unknow library {0}'.format(library))
    return covariance
