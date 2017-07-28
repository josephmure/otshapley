import numpy as np


def ishigami(x):
    """
    """
    x = np.asarray(x)
    dim = x.ndim
    if dim == 1:
        return np.sin(x[0]) + 7*np.sin(x[1])**2 + 0.1*x[2]**4 * np.sin(x[0])
    else:
        return np.sin(x[:, 0]) + 7*np.sin(x[:, 1])**2 + 0.1*x[:, 2]**4 * np.sin(x[:, 0])

def linear(x, beta=None):
    """
    """
    x = np.asarray(x)
    dim = x.ndim
    if beta is None:
        beta = np.ones((dim, ))
    else:
        beta = np.asarray(beta)
    return np.dot(x, beta)

ishigami_true_indices = [0.314, 0.442, 0.]