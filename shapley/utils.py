import pandas as pd
import numpy as np


def create_df_from_gp_indices(first_indices, mean_method=True):
    """
    """
    dim, n_realization, n_boot = first_indices.shape
    columns = ['S_%d' % (i+1) for i in range(dim)]
    if mean_method:
        df1 = pd.DataFrame(first_indices.mean(axis=2).T, columns=columns)
        df2 = pd.DataFrame(first_indices.mean(axis=1).T, columns=columns)
    else:
        df1 = pd.DataFrame(first_indices[:, :, 0].T, columns=columns)
        df2 = pd.DataFrame(first_indices[:, 0, :].T, columns=columns)

    df = pd.concat([df1, df2])
    df['Error'] = pd.DataFrame(['Kriging error']*n_realization + ['MC error']*n_boot)
    df = pd.melt(df, id_vars=['Error'], value_vars=columns, var_name='Variables', value_name='Indice values')
    return df


def create_df_from_mc_indices(first_indices):
    """
    """
    dim, n_boot = first_indices.shape
    columns = ['S_%d' % (i+1) for i in range(dim)]
    df = pd.DataFrame(first_indices.T, columns=columns)
    df = pd.melt(df, value_vars=columns, var_name='Variables', value_name='Indice values')
    return df