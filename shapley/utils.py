import pandas as pd


def create_df_from_gp_indices(first_indices):
    """
    """
    dim, n_realization, n_boot = first_indices.shape
    columns = ['S_%d' % (i+1) for i in range(dim)]

    df_gp = pd.DataFrame(first_indices.mean(axis=2).T, columns=columns)
    df_mc = pd.DataFrame(first_indices.mean(axis=1).T, columns=columns)

    df = pd.concat([df_gp, df_mc])
    err_gp = pd.DataFrame(['Kriging error']*n_realization)
    err_mc = pd.DataFrame(['MC error']*n_boot)
    df['Error'] = pd.concat([err_gp, err_mc])

    df = pd.melt(df, id_vars=['Error'], value_vars=columns, var_name='Variables', value_name='Indice values')
    return df


def create_df_from_mc_indices(indices):
    """
    """
    dim, n_boot = indices.shape
    columns = ['$X_%d$' % (i+1) for i in range(dim)]
    df = pd.DataFrame(indices.T, columns=columns)
    df = pd.melt(df, value_vars=columns, var_name='Variables', value_name='Indice values')
    return df


def create_df_from_mc_indices(indices):
    """
    """
    dim, n_boot = indices.shape
    columns = ['$X_%d$' % (i+1) for i in range(dim)]
    df = pd.DataFrame(indices.T, columns=columns)
    df = pd.melt(df, value_vars=columns, var_name='Variables', value_name='Indice values')
    return df