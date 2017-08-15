import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def set_style_paper():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")
    
    # Set the font to be serif, rather than sans
    sns.set(font='serif')
    
    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })


def plot_sensitivity_results(results, kind='violin', indice='all', ax=None):
    """Plot the result of the sensitivy result class

    Parameters
    ----------
    results : SensitivityResults instance,
        The result from a sensitivity analysis.
    kind : str,
        The type of plot to show the results.
    indice : str,
        The indices to show
    Return
    ------
    ax : matplotlib.axes
        The ploted result.
    """
    if indice == 'all':
        df_indices = results.df_indices
        hue = 'Indices'
        split = False
    elif indice == 'first':
        df_indices = results.df_first_indices
        hue = 'Error'
        split = True
    elif indice == 'total':
        df_indices = results.df_total_indices
        hue = 'Error'
        split = True
    elif indice == 'shapley':
        df_indices = results.df_shapley_indices
        hue = 'Error'
        split = True
    else:
        raise ValueError('Unknow indice parameter {0}'.format(indice))

    if kind == 'violin':
        sns.violinplot(x='Variables', y='Indice values', data=df_indices, hue=hue, split=split, ax=ax)
    elif kind == 'box':
        # TODO: to correct
        sns.boxplot(x='Variables', y='Indice values', hue='Indices', data=df_indices, ax=ax)
    else:
        raise ValueError('Unknow kind {0}'.format(kind))

    if results.true_indices is not None:
        true_indices = results.true_indices
        dodge = True if indice == 'all' else False
        colors = {'True first': "y", 'True total': "m", 'True shapley': "c"}
        names = {'all': true_indices['Indices'].unique(), 
                 'first': 'True first', 
                 'total': 'True total', 
                 'shapley': 'True shapley'}

        palette = {k:colors[k] for k in names[indice] if k in colors}
        if indice != 'all':
            true_indices = results.true_indices[results.true_indices['Indices'] == names[indice]]
        sns.stripplot(x='Variables', y='Indice values', data=true_indices, hue='Indices', ax=ax, dodge=dodge, size=9, palette=palette);

def plot_violin(df, with_hue=False, true_indices=None, ax=None, figsize=(8, 4), ylim=None, savefig=''):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if with_hue:
        sns.violinplot(x='Variables', y='Indice values', data=df, hue='Error', ax=ax, split=True)
    else:
        sns.violinplot(x='Variables', y='Indice values', data=df, ax=ax)
    if true_indices is not None:
        ax.plot(true_indices, 'yo', markersize=7, label='True indices')
        ax.legend(loc=0)
    ax.set_ylim(ylim)
    if ax is None:
        fig.tight_layout()

    return ax


def plot_violin(df, with_hue=False, true_indices=None, ax=None, figsize=(8, 4), ylim=None, savefig=''):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if with_hue:
        sns.violinplot(x='Variables', y='Indice values', data=df, hue='Error', ax=ax, split=True)
    else:
        sns.violinplot(x='Variables', y='Indice values', data=df, ax=ax)
    if true_indices is not None:
        ax.plot(true_indices, 'yo', markersize=7, label='True indices')
        ax.legend(loc=0)
    ax.set_ylim(ylim)
    if ax is None:
        fig.tight_layout()

    return ax

def violin_plot_indices(first_indices, true_indices=None, title=None, figsize=(8, 4), xlabel=None, ylim=None, ax=None):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=first_indices, ax=ax, label='First order indices')
    if true_indices is not None:
        ax.plot(true_indices, 'yo', markersize=13, label='True indices')
    ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Sobol Indices')
    ax.legend(loc=0)
    ax.set_title(title)
    if ax is None:
        fig.tight_layout()


def matrix_plot(sample, kde=False, figsize=3., aspect=1.2):
    """
    """
    data = pd.DataFrame(sample)
    plot = sns.PairGrid(data, palette=["red"], size=figsize, aspect=aspect)
    if kde:
        plot.map_upper(plt.scatter, s=10)
        plot.map_lower(sns.kdeplot, cmap="Blues_d")
    else:
        plot.map_offdiag(plt.scatter, s=10)
        
    plot.map_diag(sns.distplot, kde=False)
    plot.map_lower(corrfunc_plot)
       
    return plot

def corrfunc_plot(x, y, **kws):
    """
    
    
    Source: https://stackoverflow.com/a/30942817/5224576
    """
    r, _ = stats.pearsonr(x, y)
    k, _ = stats.kendalltau(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}\nk = {:.2f}".format(r, k),
                xy=(.1, .8), xycoords=ax.transAxes, 
                weight='heavy', fontsize=12)