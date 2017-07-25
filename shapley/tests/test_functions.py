import numpy as np


def ishigami(x):
    """Ishigami function for an OpenTURNS use.
    """
    dim = len(x)
    x = np.asarray(x)
    return [np.sin(x[0]) + 7*np.sin(x[1])**2 + 0.1*x[2]**4 * np.sin(x[0])]

ishigami_true_indices = [0.314, 0.442, 0.]