import pandas as pd

DF_NAMES = {
        'val': 'Indice values',
        'var': 'Variables',
        'ind': 'Indices',
        'error': 'Error',
        'mc-error': 'MC error',
        'gp-error': 'Kriging error',
        'mc': 'Monte-Carlo',
        'gp': 'Kriging',
        '1st': 'First',
        'tot': 'Total',
        'shap': 'Shapley'
            }


def create_df_from_gp_indices(first_indices):
    """
    """
    dim, n_realization, n_boot = first_indices.shape
    columns = ['X_%d' % (i+1) for i in range(dim)]

    df_gp = pd.DataFrame(first_indices.mean(axis=2).T, columns=columns)
    df_mc = pd.DataFrame(first_indices.mean(axis=1).T, columns=columns)

    df = pd.concat([df_gp, df_mc])
    err_gp = pd.DataFrame([DF_NAMES['gp-error']]*n_realization)
    err_mc = pd.DataFrame([DF_NAMES['mc-error']]*n_boot)
    df[DF_NAMES['error']] = pd.concat([err_gp, err_mc])

    df = pd.melt(df, id_vars=[DF_NAMES['error']], value_vars=columns, var_name=DF_NAMES['var'], value_name=DF_NAMES['val'])
    return df


def create_df_from_mc_indices(indices):
    """
    """
    dim, n_boot = indices.shape
    columns = ['$X_%d$' % (i+1) for i in range(dim)]
    df = pd.DataFrame(indices.T, columns=columns)
    df = pd.melt(df, value_vars=columns, var_name=DF_NAMES['var'], value_name=DF_NAMES['val'])
    return df


def q2_cv(ytrue, ypred):
    """Cross validation Q2 test.

    Parameters
    ----------
    ytrue : array,
        The true values.
    """
       
    ytrue = ytrue.squeeze()
    ypred = ypred.squeeze()
    q2 = max(0., test_q2(ytrue, ypred))
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