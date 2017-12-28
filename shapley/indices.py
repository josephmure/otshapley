import openturns as ot
import numpy as np
import pandas as pd
from scipy import stats
from collections import OrderedDict

from .utils import DF_NAMES

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
    """Class to gather the sensitivity analysis results
    """
    def __init__(self, first_indices=None, total_indices=None, shapley_indices=None, true_first_indices=None,
                 true_total_indices=None, true_shapley_indices=None, shapley_indices_SE=None, total_indices_SE=None, first_indices_SE=None, estimation_method=None):
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
        self.shapley_indices_SE = shapley_indices_SE
        self.total_indices_SE = total_indices_SE
        self.first_indices_SE = first_indices_SE
        self.estimation_method = estimation_method

    def get_indices_confidence_intervals(self, alpha_ci=0.05):
        """
        """
        if self.estimation_method == 'random':
            z_alpha = stats.norm.ppf(alpha_ci*0.5)
            
            dim = self.dim
            feat_indices = 'Indices'
            columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
            all_df = []
            names = (DF_NAMES['var'], 'Bounds')
            idx = [columns, ['Min', 'Max']]
            index = pd.MultiIndex.from_product(idx, names=names)
            all_df = []
            if self._first_indices is not None:
                ci_up = self._first_indices[:, 0] - z_alpha*self._first_indices_SE
                ci_down = self._first_indices[:, 0] + z_alpha*self._first_indices_SE
                ci = np.asarray([ci_down, ci_up]).T
                df = pd.DataFrame(ci.ravel(), columns=[DF_NAMES['val']], index=index)
                df = pd.melt(df.T, value_name=DF_NAMES['val'])
                df[feat_indices] = DF_NAMES['1st']
                all_df.append(df)
            if self._total_indices is not None:
                ci_up = self._total_indices[:, 0] - z_alpha*self._total_indices_SE
                ci_down = self._total_indices[:, 0] + z_alpha*self._total_indices_SE
                ci = np.asarray([ci_down, ci_up]).T
                df = pd.DataFrame(ci.ravel(), columns=[DF_NAMES['val']], index=index)
                df = pd.melt(df.T, value_name=DF_NAMES['val'])
                df[feat_indices] = DF_NAMES['tot']
                all_df.append(df)
            if self._shapley_indices is not None:
                ci_up = self._shapley_indices[:, 0] - z_alpha*self._shapley_indices_SE
                ci_down = self._shapley_indices[:, 0] + z_alpha*self._shapley_indices_SE
                ci = np.asarray([ci_down, ci_up]).T
                df = pd.DataFrame(ci.ravel(), columns=[DF_NAMES['val']], index=index)
                df = pd.melt(df.T, value_name=DF_NAMES['val'])
                df[feat_indices] = DF_NAMES['shap']
                all_df.append(df)
            return pd.concat(all_df)
        else:
            print("Cant't compute asymptotical confidence intervals for exact method.")

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
        data = OrderedDict()
        if self.true_first_indices is not None:
            data['True first'] = self.true_first_indices
        if self.true_total_indices is not None:
            data['True total'] = self.true_total_indices
        if self.true_shapley_indices is not None:
            data['True shapley'] = self.true_shapley_indices
            
        if data != {}:
            data[DF_NAMES['var']] = ['$X_{%d}$' % (i+1) for i in range(self.dim)]
            df = pd.DataFrame(data)
            indices = pd.melt(df, id_vars=[DF_NAMES['var']], var_name=DF_NAMES['ind'], value_name=DF_NAMES['val'])
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
    def first_indices_SE(self):
        """The first sobol sensitivity estimation for c.i.
        """
        if self._first_indices_SE is not None:
            return self._first_indices_SE.reshape(self.dim, -1).mean(axis=1)

    @first_indices_SE.setter
    def first_indices_SE(self, indices):
        if indices is not None:
            indices = np.asarray(indices)
        self._first_indices_SE = indices

    @property
    def total_indices_SE(self):
        """The total sobol sensitivity estimation for c.i.
        """
        if self._total_indices_SE is not None:
            return self._total_indices_SE.reshape(self.dim, -1).mean(axis=1)

    @total_indices_SE.setter
    def total_indices_SE(self, indices):
        if indices is not None:
            indices = np.asarray(indices)
        self._total_indices_SE = indices

    @property
    def shapley_indices_SE(self):
        """The shapley sensitivity estimation for c.i.
        """
        if self._shapley_indices_SE is not None:
            return self._shapley_indices_SE.reshape(self.dim, -1).mean(axis=1)

    @shapley_indices_SE.setter
    def shapley_indices_SE(self, indices):
        if indices is not None:
            indices = np.asarray(indices)
        self._shapley_indices_SE = indices

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
            #return self._shapley_indices.reshape(self.dim, -1).mean(axis=1)
            return self._shapley_indices.reshape(self.dim, -1)

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
        feat_indices = 'Indices'
        columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
        all_df = []
        if self._first_indices is not None:
            df_first = self.full_df_first_indices
            df_first_melt = pd.melt(df_first.T, value_name=DF_NAMES['val'])
            df_first_melt[feat_indices] = DF_NAMES['1st']
            all_df.append(df_first_melt)
        if self._total_indices is not None:
            df_total = self.full_df_total_indices
            df_total_melt = pd.melt(df_total.T, value_name=DF_NAMES['val'])
            df_total_melt[feat_indices] = DF_NAMES['tot']
            all_df.append(df_total_melt)
        if self._shapley_indices is not None:
            df_shapley = self.full_df_shapley_indices
            df_shapley_melt = pd.melt(df_shapley.T, value_name=DF_NAMES['val'])
            df_shapley_melt[feat_indices] = DF_NAMES['shap']
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
        df_first[DF_NAMES['ind']] = DF_NAMES['1st']
        df_total[DF_NAMES['ind']] = DF_NAMES['tot']

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
        if self.n_boot > 1:
            s = 1
        else:
            s = 0            
        df = panel_data(self._first_indices[:, s:], columns=columns)
        return df

    @property
    def full_df_total_indices(self):
        """
        """
        dim = self.dim
        columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
        if self.n_boot > 1:
            s = 1
        else:
            s = 0            
        df = panel_data(self._total_indices[:, s:], columns=columns)
        return df

    @property
    def full_df_shapley_indices(self):
        """
        """
        dim = self.dim
        columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
        if self.n_boot > 1:
            s = 1
        else:
            s = 0
        df = panel_data(self._shapley_indices[:, s:], columns=columns)
        return df

    @property
    def df_first_indices(self):
        """
        """
        df_all = []
        df = self.full_df_first_indices
        if self.n_realization > 1:
            df_kriging = melt_kriging(df).drop(DF_NAMES['gp'], axis=1)
            df_all.append(df_kriging)
        if self.n_boot > 1:
            df_boot = melt_boot(df).drop(DF_NAMES['mc'], axis=1)
            df_all.append(df_boot)
        return pd.concat(df_all)

    @property
    def df_total_indices(self):
        """
        """
        df_all = []
        df = self.full_df_total_indices
        if self.n_realization > 1:
            df_kriging = melt_kriging(df).drop(DF_NAMES['gp'], axis=1)
            df_all.append(df_kriging)
        if self.n_boot > 1:
            df_boot = melt_boot(df).drop(DF_NAMES['mc'], axis=1)
            df_all.append(df_boot)
        return pd.concat(df_all)

    @property
    def df_shapley_indices(self):
        """
        """
        df_all = []
        df = self.full_df_shapley_indices
        if self.n_realization > 1:
            df_kriging = melt_kriging(df).drop(DF_NAMES['gp'], axis=1)
            df_all.append(df_kriging)
        if self.n_boot > 1:
            df_boot = melt_boot(df).drop(DF_NAMES['mc'], axis=1)
            df_all.append(df_boot)
        return pd.concat(df_all)


def melt_kriging(df):
    """
    """
    df_kriging = df.mean(level=[DF_NAMES['var'], DF_NAMES['gp']])
    df_kriging_melt = df_kriging.reset_index()
    df_kriging_melt['Error'] = DF_NAMES['gp']
    
    return df_kriging_melt

def melt_boot(df):
    """
    """
    df_boot = df.mean(level=[DF_NAMES['var'], DF_NAMES['mc']])
    df_boot_melt = df_boot.reset_index()
    df_boot_melt['Error'] = DF_NAMES['mc']

    return df_boot_melt

def melt_all(df):
    
    df_boot_melt = melt_boot(df)
    df_kriging_melt = melt_kriging(df)
    
    df = pd.concat([df_kriging_melt.drop(DF_NAMES['gp'], axis=1), df_boot_melt.drop(DF_NAMES['mc'], axis=1)])
    return df

def panel_data(data, columns=None):
    """
    """
    dim, n_boot, n_realization = data.shape
    names = (DF_NAMES['var'], DF_NAMES['mc'], DF_NAMES['gp'])
    idx = [columns, range(n_boot), range(n_realization)]
    index = pd.MultiIndex.from_product(idx, names=names)
    df = pd.DataFrame(data.ravel(), columns=[DF_NAMES['val']], index=index)
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