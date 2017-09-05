import openturns as ot
import numpy as np
import pandas as pd

from .utils import q2_cv

VALUE_NAME = 'Indice values'

class BaseIndices(object):
    """Base class for sensitivity indices.

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


class SensitivityResults(object):
    """
    """
    def __init__(self, first_indices=None, total_indices=None, shapley_indices=None, true_first_indices=None,
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