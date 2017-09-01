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
    dim, n_boot, n_realization = data.shape
    names = ('Variables', 'Bootstrap', 'Kriging')
    idx = [columns, range(n_boot), range(n_realization)]
    index = pd.MultiIndex.from_product(idx, names=names)
    df = pd.DataFrame(data.ravel(), columns=[VALUE_NAME], index=index)
    return df

def get_shape(indices):
    """
    """
    if indices.ndim == 1:
        dim = indices.shape[0]
        n_boot = 1
        n_realization = 1
    elif indices.ndim == 2:
        dim, n_boot = indices.shape
        n_realization = 1
    elif indices.ndim == 3:
        dim, n_boot, n_realization = indices.shape

    return dim, n_boot, n_realization

class SensitivityResults(object):
    """							## add comment for each function of this part
    """
    def __init__(self, first_indices=None, total_indices=None, shapley_indices=None, calculation_method=None, true_first_indices=None,
                 true_total_indices=None, true_shapley_indices=None):
        self.dim = None
        self.n_boot = None
        self.n_realization = None
        self._var_names = None
        self.first_indices = first_indices
        self.total_indices = total_indices
        self.shapley_indices = shapley_indices
        self.true_first_indices = true_first_indices
        self.true_total_indices = true_total_indices
        self.true_shapley_indices = true_shapley_indices
        self.calculation_method = calculation_method

    @property
    def var_names(self):
        """
        """
        if self._var_names is None:
            dim = self.dim
            columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
            return columns
        else:
            return self._var_names

    @var_names.setter
    def var_names(self, names):
        self._var_names = names

    @property
    def true_indices(self):
        """The true sensitivity results.
        """
        data = {}
        if self.true_first_indices is not None:
            data['True first'] = self.true_first_indices
        if self.true_total_indices is not None:
            data['True total'] = self.true_total_indices
        if self.true_shapley_indices is not None:
            data['True shapley'] = self.true_shapley_indices
            
        if data != {}:
            data['Variables'] = ['$X_{%d}$' % (i+1) for i in range(self.dim)]
            df = pd.DataFrame(data)
            indices = pd.melt(df, id_vars=['Variables'], var_name='Indices', value_name=VALUE_NAME)
            return indices

    @property
    def first_indices(self):
        """The first sobol sensitivity estimation.
        """
        if self._first_indices is not None:
            return self._first_indices.reshape(self.dim, -1).mean(axis=1)

    @first_indices.setter
    def first_indices(self, indices):
        if indices is not None:
            indices = np.asarray(indices)
            self.dim, self.n_boot, self.n_realization = self._check_indices(indices)
        self._first_indices = indices

    @property
    def total_indices(self):
        """The total Sobol sensitivity indicies estimations.
        """
        if self._total_indices is not None:
            return self._total_indices.reshape(self.dim, -1).mean(axis=1)

    @total_indices.setter
    def total_indices(self, indices):
        if indices is not None:
            indices = np.asarray(indices)
            self.dim, self.n_boot, self.n_realization = self._check_indices(indices)
        self._total_indices = indices

    @property
    def shapley_indices(self):
        """The Shapley indices estimations.
        """
        if self._shapley_indices is not None:
            return self._shapley_indices.reshape(self.dim, -1).mean(axis=1)

    @shapley_indices.setter
    def shapley_indices(self, indices):
        if indices is not None:
            indices = np.asarray(indices)
            self.dim, self.n_boot, self.n_realization = self._check_indices(indices)
        self._shapley_indices = indices

    def _check_indices(self, indices):
        """Get the shape of the indices result matrix and check if the shape is correct with history.
        """
        dim, n_boot, n_realization = get_shape(indices)
        if self.dim is not None:
            assert self.dim == dim, \
                "Dimension should be the same as for the other indices. %d ! %d" % (self.dim, dim)
        if self.n_boot is not None:
            assert self.n_boot == n_boot, \
                "Bootstrap size should be the same as for the other indices. %d ! %d" % (self.n_boot, n_boot)
        if self.n_realization is not None:
            assert self.n_realization == n_realization, \
                "Number of realizations should be the same as for the other indices. %d ! %d" % (self.n_realization, n_realization)
        return dim, n_boot, n_realization

    @property
    def df_indices(self):
        """The dataframe of the sensitivity results
        """
        dim = self.dim
        n_boot = self.n_boot
        n_realization = self.n_realization
        feat_indices = 'Indices'
        columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
        all_df = []
        if self._first_indices is not None:
            df_first = panel_data(self._first_indices, columns=columns)
            df_first_melt = pd.melt(df_first.T, value_name=VALUE_NAME)
            df_first_melt[feat_indices] = 'First'
            all_df.append(df_first_melt)
        if self._total_indices is not None:
            df_total = panel_data(self._total_indices, columns=columns)
            df_total_melt = pd.melt(df_total.T, value_name=VALUE_NAME)
            df_total_melt[feat_indices] = 'Total'
            all_df.append(df_total_melt)
        if self._shapley_indices is not None:
            df_shapley = panel_data(self._shapley_indices, columns=columns)
            df_shapley_melt = pd.melt(df_shapley.T, value_name=VALUE_NAME)
            df_shapley_melt[feat_indices] = 'Shapley'
            all_df.append(df_shapley_melt)

        df = pd.concat(all_df)

        return df
    
    @property
    def full_df_indices(self):
        """
        """
        dim = self.dim
        columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
        df_first = panel_data(self._first_indices, columns=columns)
        df_total = panel_data(self._total_indices, columns=columns)
        df_first['Indices'] = 'First'
        df_total['Indices'] = 'Total'

        df = pd.concat([df_first, df_total])
        return df

    @property
    def full_first_indices(self):
        """
        """
        if np.isnan(self._first_indices).all():
            raise ValueError('The value is not registered')
        if self.n_realization == 1:
            return self._first_indices[:, :, 0]
        else:
            return self._first_indices
    
    @property
    def full_total_indices(self):
        """
        """
        if np.isnan(self._total_indices).all():
            raise ValueError('The value is not registered')
        if self.n_realization == 1:
            return self._total_indices[:, :, 0]
        else:
            return self._total_indices

    @property
    def full_shapley_indices(self):
        """
        """
        if np.isnan(self._shapley_indices).all():
            raise ValueError('The value is not registered')
        if self.n_realization == 1:
            return self._shapley_indices[:, :, 0]
        else:
            return self._shapley_indices

    @property
    def full_df_first_indices(self):
        """
        """
        dim = self.dim
        columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
        df = panel_data(self._first_indices, columns=columns)
        return df

    @property
    def full_df_total_indices(self):
        """
        """
        dim = self.dim
        columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
        df = panel_data(self._total_indices, columns=columns)
        return df

    @property
    def full_df_shapley_indices(self):
        """
        """
        dim = self.dim
        columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
        df = panel_data(self._shapley_indices, columns=columns)
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

    @property
    def df_shapley_indices(self):
        """
        """
        df = melt_kriging(self.full_df_shapley_indices)
        return df

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


class ProbabilisticModel(Model):			## add some comments in this class
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
        self._shapley_indices = None

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
        return self._margins

    @margins.setter
    def margins(self, margins):
        assert isinstance(margins, list), "It should be a list"
        for marginal in margins:
            assert isinstance(marginal, ot.DistributionImplementation), "The marginal should be an OpenTURNS implementation."
        self._input_distribution = ot.ComposedDistribution(margins, self._copula)
        self._margins = margins

    @property
    def dim(self):
        """The problem dimension.
        """
        return self._dim

    @property
    def input_distribution(self):
        """The OpenTURNS input distribution.
        """
        return self._input_distribution

    @input_distribution.setter
    def input_distribution(self, dist):
        assert isinstance(dist, ot.DistributionImplementation), "The distribution should be an OpenTURNS implementation."
        self._input_distribution = dist
        self._dim = self._input_distribution.getDimension()
        self._margins = [dist.getMarginal(i) for i in range(self._dim)]
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
        if self._shapley_indices is None:
            print ('There is no true first order shapley effect.')

        return self._shapley_indices

    @shapley_indices.setter
    def shapley_indices(self, indices):
        """
        """
        self._shapley_indices = indices

    @property
    def sobol_indices(self):
        """
        """
        df = pd.DataFrame({'True first': self.first_order_sobol_indices,
                  'True total': self.total_sobol_indices,
                  'Variables': ['$X_{%d}$' % (i+1) for i in range(self._dim)]})
        true_indices = pd.melt(df, id_vars=['Variables'], var_name='Indices', value_name=VALUE_NAME)
        
        return true_indices

    @property
    def indices(self):
        """
        """
        df = pd.DataFrame({'True first': self.first_order_sobol_indices,
                  'True total': self.total_sobol_indices,
                  'True shapley': self.shapley_indices,
                  'Variables': ['$X_{%d}$' % (i+1) for i in range(self._dim)]})
        indices = pd.melt(df, id_vars=['Variables'], var_name='Indices', value_name=VALUE_NAME)
        
        return indices


class MetaModel(ProbabilisticModel):
    """
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

    @property
    def input_sample(self):
        """The input sample to build the model.
        """
        return self._input_sample
    
    @input_sample.setter
    def input_sample(self, sample):
        n_sample, dim = sample.shape
        assert dim == self._dim, "Dimension should be the same as the input_distribution: %d != %d" % (dim, self._dim)
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
        assert n_sample == self._n_sample, "Samples should be the same sizes: %d != %d" % (n_sample, self._n_samples)
        self._output_sample = sample