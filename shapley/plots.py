import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy import stats
from collections import OrderedDict

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

        if indice == 'all':
            indice_names = {'First': 'first',
                      'Total': 'total',
                      'Shapley': 'shapley'}
            df = pd.DataFrame(columns=true_indices.columns)
            for name in df_indices.Indices.unique():
                tmp = names[indice_names[name]]
                if tmp in true_indices['Indices'].unique():
                    df = pd.concat([df, true_indices[true_indices['Indices'] == tmp]])
            true_indices = df
            palette = {k: colors[k] for k in names[indice] if k in colors}
        else:
            palette = {names[indice]: colors[names[indice]]}
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


def plot_correlation_indices(result_indices, corrs, n_boot, to_plot=['Shapley'], linewidth=1, markersize=10, ax=None, figsize=(9, 5), alpha=[0.05, 0.95]):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    dim = 3
    columns = ['$X_%d$' % (i+1) for i in range(dim)]
    names = ('Correlation', 'Variables', 'Bootstrap')
    idx = [corrs, columns, range(n_boot)]
    index = pd.MultiIndex.from_product(idx, names=names)

    markers = {'Shapley': 'o',
               'First Sobol': '*',
               'Total Sobol': '.',
               'First full Sobol': 8,
               'Total full Sobol': 9,
               'First ind Sobol': 10,
               'Total ind Sobol': 11,
               }

    colors = {'$X_1$': 'b',
             '$X_2$': 'r',
             '$X_3$': 'g'}

    for name in result_indices:
        if name in to_plot:
            results = pd.DataFrame(index=index)
            results['Indice Values'] = np.concatenate(result_indices[name])
            results.reset_index(inplace=True)
            quantiles = results.groupby(['Correlation', 'Variables']).quantile(alpha).drop('Bootstrap', axis=1)
            means = results.groupby(['Correlation', 'Variables']).mean().drop('Bootstrap', axis=1)
            quantiles.reset_index(inplace=True)
            means.reset_index(inplace=True)

            for i, var in enumerate(columns):
                df_quant = quantiles[quantiles['Variables'] == var]['Indice Values']
                df_means = means[means['Variables'] == var]['Indice Values']
                quant_up = df_quant.values[1::2]
                quant_down = df_quant.values[::2]
                ax.plot(corrs, df_means.values, '--', marker=markers[name], color=colors[var], linewidth=linewidth, markersize=markersize)
                ax.fill_between(corrs, quant_down, quant_up, interpolate=True, alpha=.5, color=colors[var])

    ax.set_ylim(0., 1.)
    ax.set_xlim([-1., 1.])

    patches = []
    for var in colors: 
        patches.append(mpatches.Patch(color=colors[var], label=var))

    for name in markers:
        if name in to_plot:
            patches.append(mlines.Line2D([], [], color='k', marker=markers[name], label=name, linewidth=linewidth, markersize=markersize))

    ax.legend(loc=0, handles=patches, fontsize=11, ncol=2)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Indices')

    return ax