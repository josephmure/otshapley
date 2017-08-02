import openturns as ot
import numpy as np


class Base(object):
    """Base class.

    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        And OpenTURNS distribution object.
    """
    def __init__(self, input_distribution):
        self.input_distribution = input_distribution
        self.first_order_indice_func = None
        self.total_indice_func = None

    @property
    def input_distribution(self):
        """The OpenTURNS input distribution.
        """
        return self._input_distribution

    @input_distribution.setter
    def input_distribution(self, dist):
        assert isinstance(dist, ot.DistributionImplementation), \
            "The distribution should be an OpenTURNS Distribution object. Given %s" % (type(dist))
        self._input_distribution = dist

    @property
    def dim(self):
        """The input dimension.
        """
        return self._input_distribution.getDimension()

    @property
    def first_order_indice_func(self):
        """Function for 1st indice computation.
        """
        return self._first_order_indice_func

    @first_order_indice_func.setter
    def first_order_indice_func(self, func):
        assert callable(func) or func is None, \
            "First order indice function should be callable or None."

        self._first_order_indice_func = func

    @property
    def total_indice_func(self):
        """Function for 1st indice computation.
        """
        return self._first_order_indice_func

    @total_indice_func.setter
    def total_indice_func(self, func):
        assert callable(func) or func is None, \
            "Total indice function should be callable or None."

        self._first_order_indice_func = func

class Model(object):
    """Class to create Model object.

    Parameters
    ----------
    model_func : callable,
        The model function.
    """
    def __init__(self, model_func):
        self.model_func = model_func

    @property
    def model_func(self):
        """The model function.
        """
        return self._model_func

    @model_func.setter
    def model_func(self, func):
        assert callable(func), "The function should be callable"
        self._model_func = func

    def __call__(self, x):
        y = self._model_func(x)
        return y


class ProbabilisticModel(Model):
    """Create probabilistic model instances.

    Parameters
    ----------
    model_func : callable,
        The model function.
    input_distribution : ot.DistributionImplementation,
        The input distribution
    """
    def __init__(self, model_func, input_distribution):
        Model.__init__(self, model_func=model_func)
        self.input_distribution = input_distribution
        self._first_order_sobol_indices = None

    @property
    def copula(self):
        """The problem copula.
        """
        return self._copula
    
    @copula.setter
    def copula(self, copula):
        assert isinstance(copula, ot.CopulaImplementation), "The copula should be an OpenTURNS implementation."
        self._input_distribution = ot.ComposedDistribution(self._margins, copula)
        self._copula = copula

    @property
    def margins(self):
        """The problem margins.
        """
        return selt._margins

    @margins.setter
    def margins(self, margins):
        assert isinstance(margins, list), "It should be a list"
        for marginal in margins:
            assert isinstance(marginal, ot.DistributionImplementation), "The marginal should be an OpenTURNS implementation."
        self._input_distribution = ot.ComposedDistribution(margins, self._copula)
        self._margins = margins

    @property
    def ndim(self):
        """The problem dimension.
        """
        return self._ndim

    @property
    def input_distribution(self):
        """The OpenTURNS input distribution.
        """
        return self._input_distribution

    @input_distribution.setter
    def input_distribution(self, dist):
        assert isinstance(dist, ot.DistributionImplementation), "The distribution should be an OpenTURNS implementation."
        self._input_distribution = dist
        self._ndim = self._input_distribution.getDimension()
        self._margins = [dist.getMarginal(i) for i in range(self._ndim)]
        self._copula = dist.getCopula()

    def get_input_sample(self, n_sample, sampling='lhs'):
        """Generate a sample of the input distribution.

        Parameters
        ----------
        n_sample : int,
            The number of observations.
        sampling : str,
            The sampling type.

        Returns
        -------
        input_sample : array,
            A sample of the input distribution.
        """
        if sampling =='lhs':
            lhs = ot.LHSExperiment(self._input_distribution, n_sample)
            input_sample = np.asarray(lhs.generate())
        elif sampling == 'monte-carlo':
            input_sample = np.asarray(self._input_distribution.getSample(n_sample))

        return input_sample

    @property
    def first_order_sobol_indices(self):
        """The true first order sobol indices.
        """
        if self._first_order_sobol_indices is None:
            print ('There is no true first order sobol indices')

        return self._first_order_sobol_indices