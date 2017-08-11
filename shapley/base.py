import openturns as ot
import numpy as np
import pandas as pd

VALUE_NAME = 'Indice values'

class Base(object):
    """Base class.

    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        And OpenTURNS distribution object.
    """
    def __init__(self, input_distribution):
        self.input_distribution = input_distribution
        self.indice_func = None

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
    def indice_func(self):
        """Function to estimate the indice.
        """
        return self._indice_func

    @indice_func.setter
    def indice_func(self, func):
        assert callable(func) or func is None, \
            "Indice function should be callable or None."

        self._indice_func = func


def panel_data(data, columns=None):
    """
    """
    dim, n_realization, n_boot = data.shape
    names = ('Variables', 'Kriging', 'Bootstrap')
    idx = [columns, range(n_realization), range(n_boot)]
    index = pd.MultiIndex.from_product(idx, names=names)
    df = pd.DataFrame(data.ravel(), columns=[VALUE_NAME], index=index)
    return df


class SensitivityResults(object):
    """
    """
    def __init__(self, first_indices=None, total_indices=None, calculation_method=None, true_indices=None):
        self.first_indices = first_indices
        self.total_indices = total_indices
        self.true_indices = true_indices
        self.calculation_method = calculation_method

    @property
    def calculation_method(self):
        """
        """
        return self._calculation_method

    @calculation_method.setter
    def calculation_method(self, method):
        """
        """
        self._calculation_method = method

    @property
    def true_indices(self):
        """
        """
        return self._true_indices

    @true_indices.setter
    def true_indices(self, indices):
        self._true_indices = indices

    @property
    def df_indices(self):
        """
        """
        dim = self.ndim
        n_boot = self.n_boot
        columns = ['$X_%d$' % (i+1) for i in range(dim)]

        if self._calculation_method == 'monte-carlo':
            df_first = pd.DataFrame(self._first_indices.T, columns=columns)
            df_total = pd.DataFrame(self._total_indices.T, columns=columns)
            df_first['Indices'] = 'First'
            df_total['Indices'] = 'Total'

            df = pd.concat([df_first, df_total])
            df = pd.melt(df, id_vars=['Indices'], value_vars=columns, var_name='Variables', value_name=VALUE_NAME)

            return df
        elif self._calculation_method == 'kriging-mc':
            df_first = panel_data(self._first_indices, columns=columns)
            df_total = panel_data(self._total_indices, columns=columns)
            df_first_melt = pd.melt(df_first.T, value_name=VALUE_NAME)
            df_total_melt = pd.melt(df_total.T, value_name=VALUE_NAME)
            df_first_melt['Indices'] = 'First'
            df_total_melt['Indices'] = 'Total'

            df = pd.concat([df_first_melt, df_total_melt])
            return df
        else:
            raise('You should first specify the calculation method.')
    
    @property
    def full_df_indices(self):
        """
        """
        dim = self.ndim
        columns = ['$X_%d$' % (i+1) for i in range(dim)]
        if self._calculation_method == 'monte-carlo':
            pass
        elif self._calculation_method == 'kriging-mc':
            df_first = panel_data(self._first_indices, columns=columns)
            df_total = panel_data(self._total_indices, columns=columns)
            df_first['Indices'] = 'First'
            df_total['Indices'] = 'Total'

            df = pd.concat([df_first, df_total])
            return df

    @property
    def first_indices(self):
        """
        """
        return self._first_indices.reshape(self.ndim, -1).mean(axis=1)

    @first_indices.setter
    def first_indices(self, indices):
        if indices is not None:
            if indices.ndim == 1:
                self.ndim = indices.shape[0]
                self.n_boot = 1
                self.n_realization = 1
            elif indices.ndim == 2:
                self.ndim, self.n_boot = indices.shape
                self.n_realization = 1
            elif indices.ndim == 3:
                self.ndim, self.n_realization, self.n_boot = indices.shape

        self._first_indices = indices

    @property
    def full_first_indices(self):
        """
        """
        return self._first_indices

    
    @property
    def full_total_indices(self):
        """
        """
        if np.isnan(self._total_indices).all():
            raise ValueError('The value is not registered')
        return self._total_indices

    @property
    def total_indices(self):
        """
        """
        return self._total_indices.reshape(self.ndim, -1).mean(axis=1)

    @total_indices.setter
    def total_indices(self, indices):
        self._total_indices = indices

    @property
    def full_df_first_indices(self):
        """
        """
        dim = self.ndim
        columns = ['$X_%d$' % (i+1) for i in range(dim)]
        if self._calculation_method == 'monte-carlo':
            pass
        elif self._calculation_method == 'kriging-mc':
            df = panel_data(self._first_indices, columns=columns)
        return df

    @property
    def full_df_total_indices(self):
        """
        """
        dim = self.ndim
        columns = ['$X_%d$' % (i+1) for i in range(dim)]
        if self._calculation_method == 'monte-carlo':
            pass
        elif self._calculation_method == 'kriging-mc':
            df = panel_data(self._total_indices, columns=columns)
        return df

    @property
    def df_first_indices(self):
        """
        """
        df = melt_kriging(self.full_df_first_indices)
        return df

    @property
    def df_total_indices(self):
        """
        """
        df = melt_kriging(self.full_df_total_indices)
        return df


def melt_kriging(df):
    """
    """
    df_boot = df.mean(level=['Variables', 'Kriging'])
    df_boot_melt = pd.melt(df_boot.T, value_name=VALUE_NAME)
    df_boot_melt['Error'] = 'Kriging'

    df_kriging = df.mean(level=['Variables', 'Bootstrap'])
    df_kriging_melt = pd.melt(df_kriging.T, value_name=VALUE_NAME)
    df_kriging_melt['Error'] = 'Bootstrap'

    df = pd.concat([df_boot_melt.drop('Kriging', axis=1), df_kriging_melt.drop('Bootstrap', axis=1)])
    return df

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
        if func is not None:
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
        self._total_sobol_indices = None

    @property
    def copula(self):
        """The problem copula.
        """
        return self._copula
    
    @copula.setter
    def copula(self, copula):
        assert isinstance(copula, (ot.CopulaImplementation, ot.DistributionImplementationPointer)), \
            "The copula should be an OpenTURNS implementation: {0}".format(type(copula))
        self._input_distribution = ot.ComposedDistribution(self._margins, copula)
        self._copula = copula

    @property
    def copula_parameters(self):
        """
        """
        return self._copula_parameters

    @copula_parameters.setter
    def copula_parameters(self, params):
        copula = self._copula
        copula.setParameter(params)
        self.copula = copula

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

    @first_order_sobol_indices.setter
    def first_order_sobol_indices(self, indices):
        """
        """
        self._first_order_sobol_indices = indices

    @property
    def total_sobol_indices(self):
        """The true total sobol indices.
        """
        if self._total_sobol_indices is None:
            print ('There is no true first order sobol indices')

        return self._total_sobol_indices

    @total_sobol_indices.setter
    def total_sobol_indices(self, indices):
        """
        """
        self._total_sobol_indices = indices

    @property
    def shapley_indices(self):
        """The true shapley effects.
        """
        if self._shapley_indice is None:
            print ('There is no true first order shapley effect.')

        return self._shapley_indice

    @property
    def sobol_indices(self):
        """
        """
        df = pd.DataFrame({'True first': self.first_order_sobol_indices,
                  'True total': self.total_sobol_indices,
                  'Variables': ['$X_%d$' % (i+1) for i in range(self._ndim)]})
        true_indices = pd.melt(df, id_vars=['Variables'], var_name='Indices', value_name=VALUE_NAME)
        
        return true_indices