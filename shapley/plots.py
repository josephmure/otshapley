import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

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


def plot_sensitivity_results(results, kind='violin', indice='all', ax=None, alpha_ci=0.05):
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
        if results.n_boot > 1:
            sns.boxplot(x='Variables', y='Indice values', hue='Indices', data=df_indices, ax=ax)
        else:
            z_alpha = stats.norm.ppf(alpha_ci*0.5)
            ci_up = results.shapley_indices + z_alpha*results.shapley_indices_SE
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


def plot_correlation_indices(result_indices, corrs, n_boot, true_indices=None, to_plot=['Shapley'], linewidth=1, markersize=10, ax=None, figsize=(9, 5), alpha=[0.05, 0.95], ci='error'):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    dim = 3
    columns = ['$X_{%d}$' % (i+1) for i in range(dim)]
    names = ('Correlation', 'Variables', 'Bootstrap')
    idx = [corrs, columns, range(n_boot)]
    index = pd.MultiIndex.from_product(idx, names=names)

    markers = {'Shapley': 'o',
               'First Sobol': '*',
               'Total Sobol': '.',
               'First full Sobol': 8,
               'Total full Sobol': 11,
               'First ind Sobol': 10,
               'Total ind Sobol': 11,
               }

    colors = {'$X_{1}$': 'b',
             '$X_{2}$': 'r',
             '$X_{3}$': 'g'}

    for name in result_indices:
        if name in to_plot:
            results = pd.DataFrame(index=index)
            n_corr = len(result_indices[name])
            results['Indice Values'] = np.concatenate(result_indices[name])
            means_no_boot = results['Indice Values'].values.reshape(n_corr, dim, -1)[:, :, 0]
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
                if ci == 'quantile':
                    mean = df_means.values
                    ci_up = quant_up
                    ci_down = quant_down
                elif ci == 'error':
                    mean = means_no_boot[:, i]
                    ci_up = 2*mean - quant_up
                    ci_down = 2*mean - quant_down
                else:
                    raise ValueError('Unknow confidence interval')
                if true_indices is not None:
                    if name in true_indices:
                        mean = np.asarray(true_indices[name])[:, i]
                #ax.plot(corrs, mean, '--', marker=markers[name], color=colors[var], linewidth=linewidth, markersize=markersize)
                ax.plot(corrs, mean, '-', color=colors[var], linewidth=linewidth, markersize=markersize)
                ax.fill_between(corrs, ci_down, ci_up, interpolate=True, alpha=.15, color=colors[var])

    ax.set_ylim(0., 1.)
    ax.set_xlim([-1., 1.])

    patches = []
    for var in colors: 
        patches.append(mpatches.Patch(color=colors[var], label=var))

    for name in markers:
        if name in to_plot:
            if False:
                patches.append(mlines.Line2D([], [], color='k', marker=markers[name], label=name, linewidth=linewidth, markersize=markersize))

    #ax.legend(loc=0, handles=patches, fontsize=11, ncol=2)
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Indices')

    return ax


def plot_error(results, true_results, x, ax=None, 
               figsize=(7, 4), ylim=[0., None], loc=0, logscale=False, legend=False,
               error_type='absolute'):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    colors = {
        'Shapley': 'b',
        'First Sobol': 'r',
        'Total Sobol': 'g'
    }
    
    lns = []
    if logscale:
        ax.set_yscale('log')

    sorted_x = np.argsort(x)
    for i, name in enumerate(results):
        result = results[name]
        true_indices = true_results[name]        
        
        # Estimation without bootstrap
        no_boot_estimation = result[:, :, :, 0]
            
        # Shows the absolute error or relative
        norm = 1 if error_type == 'absolute' else true_indices
        error = (abs(no_boot_estimation - true_indices) / norm ).mean(axis=2)
        error_quants = np.percentile(error, [2.5, 97.5], axis=1)

        lns2 = ax.plot(x[sorted_x], error.mean(axis=1)[sorted_x], '--', label='Error %s' % (name), linewidth=2, color=colors[name])
        ax.fill_between(x[sorted_x], error.mean(axis=1)[sorted_x], error_quants[0][sorted_x], alpha=0.3, color=colors[name])
        ax.fill_between(x[sorted_x], error.mean(axis=1)[sorted_x], error_quants[1][sorted_x], alpha=0.3, color=colors[name])

        lns.extend(lns2)

    ax.set_xlim(x[sorted_x][0], x[sorted_x][-1])
    
    labs = [l.get_label() for l in lns]
    if legend:
        ax.legend(lns, labs, loc=loc)
        
    label = 'Absolute' if error_type else 'Relative'
    ax.set_ylabel('%s error' % label)

    return ax

def plot_cover(results, true_results, x, results_SE=None, ax=None, figsize=(7, 4), 
               ylim=[0., None], ci_prob=0.95, loc=0, legend=True,
               error_type='absolute'):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        
    colors = {
        'Shapley': 'b',
        'First Sobol': 'r',
        'Total Sobol': 'g'
    }
    
    lns = []

    sorted_x = np.argsort(x)
    for i, name in enumerate(results):
        result = results[name]
        if results_SE is not None:
            result_SE = results_SE[name]
        true_indices = true_results[name]
        dim = true_indices.shape[0]
        n_boot = result.shape[-1]
        z_alpha = stats.norm.ppf(ci_prob*0.5)
        # Estimation without bootstrap
        no_boot_estimation = result[:, :, :, 0]

        if n_boot > 1:
            boot_estimation = result[:, :, :, 1:]
            # Quantile of Gaussian of the empirical CDF at the no_boot estimation
            z_0 = stats.norm.ppf((boot_estimation <= no_boot_estimation[:, :, :, np.newaxis]).mean(axis=-1))

            # Quantile func of the empirical bootstrap distribution
            tmp_up = stats.norm.cdf(2*z_0 - z_alpha)
            tmp_down = stats.norm.cdf(2*z_0 + z_alpha)

            n_N = result.shape[0]
            n_test = result.shape[1]

            ci_up = np.zeros((n_N, n_test, dim))
            ci_down = np.zeros((n_N, n_test, dim))
            for i in range(n_N):
                for j in range(n_test):
                    for d in range(dim):
                        ci_up[i, j, d] = np.percentile(boot_estimation[i, j, d], tmp_up[i, j, d]*100.)
                        ci_down[i, j, d] = np.percentile(boot_estimation[i, j, d], tmp_down[i, j, d]*100.)

        else:
            ci_up = no_boot_estimation - z_alpha * result_SE
            ci_down = no_boot_estimation + z_alpha * result_SE
            
        # Cover with mean over the number of tests
        cover = ((ci_down < true_indices.reshape(1, 1, dim)) & (ci_up > true_indices.reshape(1, 1, dim))).mean(axis=1)

        lns1 = ax.plot(x[sorted_x], cover.mean(axis=1)[sorted_x], '-', label='Coverage of %s' % (name), linewidth=2, color=colors[name])
        lns.extend(lns1)

    xmin, xmax = x[sorted_x][0], x[sorted_x][-1]
    ax.plot([xmin, xmax], [1. - ci_prob]*2, 'k-.', label='%d%% c.i.' % ((1.- ci_prob)*100))
    ax.set_ylabel('Coverage Probability')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ylim)

    return ax

def plot_var(results, x, ax=None, figsize=(7, 4), ylim=None, alpha=[2.5, 97.5], loc=0, logscale=False, legend=True):
    """
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    colors = {
        'Shapley': 'b',
        'First Sobol': 'r',
        'Total Sobol': 'g'
    }

    sorted_x = np.argsort(x)
    if logscale:
        ax.set_yscale('log')
    for i, name in enumerate(results):
        result = results[name]
        no_boot_estimation = result[:, :, :, 0]
        mean = result[:, :, :, 1:].mean(axis=3).mean(axis=2)
        std = result[:, :, :, 1:].std(axis=3).mean(axis=2)
        #mean = no_boot_estimation.mean(axis=2)
        #std = no_boot_estimation.std(axis=2)
        cov = abs(std/mean)
        cov_quants = np.percentile(cov, [2.5, 97.5], axis=1)
        lns1 = ax.plot(x[sorted_x], cov.mean(axis=1)[sorted_x], '-', label='COV %s' % (name), linewidth=2, color=colors[name])
        ax.fill_between(x[sorted_x], cov.mean(axis=1)[sorted_x], cov_quants[0][sorted_x], alpha=0.3, color=colors[name])
        ax.fill_between(x[sorted_x], cov.mean(axis=1)[sorted_x], cov_quants[1][sorted_x], alpha=0.3, color=colors[name])

    ax.set_xlabel('$N_i$')
    ax.set_ylabel('Ceofficient of Variation')
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(ylim)
    if legend:
        ax.legend(loc=loc)

    return ax