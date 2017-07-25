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

def ot_ishigami(x):
    """Ishigami function for an OpenTURNS use.
    """
    return [ishigami(x)]

ishigami_true_indices = [0.314, 0.442, 0.]