import seaborn as sns
import matplotlib.pyplot as plt

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


def violin_plot_indices(first_indices, true_indices=None, figsize=(8, 4), xlim=None):
    """
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=first_indices, ax=ax, label='First order indices')
    if true_indices is not None:
        ax.plot(true_indices, 'yo', markersize=13, label='True indices')
    ax.set_ylim(xlim)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Sobol Indices')
    ax.legend(loc=0)
    fig.tight_layout()