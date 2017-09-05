import numpy as np
import openturns as ot

from shapley.model import ProbabilisticModel

class Ishigami(ProbabilisticModel):
    """This class collect all the information about the Ishigami test function for sensitivity analysis.
    """
    def __init__(self):
        dim = 3
        margins = [ot.Uniform(-np.pi, np.pi)]*dim
        copula = ot.IndependentCopula(dim)
        input_distribution = ot.ComposedDistribution(margins, copula)
        ProbabilisticModel.__init__(
            self,            model_func=ishigami_func, 
            input_distribution=input_distribution,
            first_sobol_indices=[0.314, 0.442, 0.],
            total_sobol_indices=[0.56, 0.44, 0.24],
            shapley_indices=[0.437, 0.441, 0.12])
        self.name = 'Ishigami'

class AdditiveGaussian(ProbabilisticModel):
    """This class collect all the information about the Ishigami test function for sensitivity analysis.
    """
    def __init__(self, dim, beta=None):
        margins = [ot.Normal()]*dim
        copula = ot.NormalCopula(dim)
        input_distribution = ot.ComposedDistribution(margins, copula)
        ProbabilisticModel.__init__(self, model_func=additive_func, input_distribution=input_distribution)
        self.beta = beta
        self.name = 'Additive Gaussian'

    @property
    def beta(self):
        """
        """
        return self._beta

    @beta.setter
    def beta(self, beta):
        if beta is None:
            beta = np.ones((self.dim, ))
        else:
            beta = np.asarray(beta)

        self._beta = beta

    @property
    def first_order_sobol_indices(self):
        """
        """
        #beta = self.beta
        #dim = self.dim
        #sigma = np.asarray(self.input_distribution.getCovariance())
        #print(sigma)
        #inv_sigma = np.linalg.inv(sigma)
        #var_y = (beta.dot(sigma)).dot(beta)
        #indices = np.zeros((dim,))
        #for j in range(dim):
        #    c_j = np.asarray([i for i in range(dim) if i != j])
        #    inv_j = np.linalg.inv(sigma[c_j, :][:, c_j])
        #    #inv_j = inv_sigma[:, c_j][c_j, :]
        #    var_j = (beta.dot(sigma - (sigma[:, c_j].dot(inv_j)).dot(sigma[c_j, :]))).dot(beta)
        #    var_j = np.sum(sigma - (sigma[:, c_j].dot(inv_j)).dot(sigma[c_j, :]))
        #    print(var_j)
        #    indices[j] = var_j / var_y
    
        #return indices

        dim = self.dim
        beta = self.beta
        sigma = np.asarray(self.input_distribution.getCovariance())
        var_y = (beta.dot(sigma)).dot(beta)
        sigma_x = np.sqrt(sigma.diagonal())

        # Effects without correlation
        s_uncorr = (beta * sigma_x)**2

        # Effects with correlation
        theta = np.asarray(self.copula.getParameter())
        dep_pair = theta != 0
        rho = theta[dep_pair]
        s_corr = np.zeros((dim, ))
        if dim == 3 and len(rho) <= 1:
            rho = rho.item() if len(rho) == 1 else 0
            s_corr[0] = 1
            s_corr[1] = (1 + rho * sigma_x[2])**2
            s_corr[2] = (rho + sigma_x[2])**2
    
            indices = (s_corr)/ var_y
            return indices 
        elif dim == 2:
            indices = np.asarray([(1 + 2*theta[0] + theta[0]**2)/var_y]*dim)
            return indices
        else:
            return self._first_order_sobol_indices

    @first_order_sobol_indices.setter
    def first_order_sobol_indices(self, indices):
        self._first_order_sobol_indices = indices

    @property
    def total_sobol_indices(self):
        """
        """
        #beta = self.beta
        #dim = self.dim
        #sigma = np.asarray(self.input_distribution.getCovariance())
        #inv_sigma = np.linalg.inv(sigma)
        #var_y = (beta.dot(sigma)).dot(beta)
        #indices = np.zeros((dim,))
        #for j in range(dim):
        #    c_j = np.asarray([i for i in range(dim) if i != j])
        #    inv_j = np.linalg.inv(sigma[c_j, :][:, c_j])
        #    #inv_j = inv_sigma[:, c_j][c_j, :]
        #    var_j = (beta.dot((sigma[:, c_j].dot(inv_j)).dot(sigma[c_j, :]))).dot(beta)
        #    indices[j] = var_j / var_y
    
        #return indices

        dim = self.dim
        beta = self.beta
        sigma = np.asarray(self.input_distribution.getCovariance())
        var_y = (beta.dot(sigma)).dot(beta)
        sigma_x = np.sqrt(sigma.diagonal())

        # Effects without correlation
        s_uncorr = (beta * sigma_x)**2

        # Effects with correlation
        theta = np.asarray(self.copula.getParameter())
        dep_pair = theta != 0
        rho = theta[dep_pair]
        s_corr = np.zeros((dim, ))
        if dim == 3 and len(rho) <= 1:
            rho = rho.item() if len(rho) == 1 else 0
            s_corr[0] = 1
            s_corr[1] = 1. - rho**2
            s_corr[2] = sigma_x[2]**2 * ( 1 - rho **2)
    
            indices = (s_corr)/ var_y
            return indices
        elif dim == 2:
            indices = np.asarray([(1.- theta[0]**2)/var_y]*dim)
            return indices
        else:
            return self._total_sobol_indices

    @total_sobol_indices.setter
    def total_sobol_indices(self, indices):
        self._total_sobol_indices = indices

    @property
    def shapley_indices(self):
        """
        """
        dim = self.dim
        beta = self.beta
        sigma = np.asarray(self.input_distribution.getCovariance())
        var_y = (beta.dot(sigma)).dot(beta)
        sigma_x = np.sqrt(sigma.diagonal())

        # Effects without correlation
        s_uncorr = (beta * sigma_x)**2
        
        # Effects with correlation
        theta = np.asarray(self.copula.getParameter())
        dep_pair = theta != 0
        rho = theta[dep_pair]
        if dim == 3 and len(rho) <= 1:
            rho = rho.item() if len(rho) == 1 else 0
            s_corr = np.zeros((dim, ))
            for i, j in [[1, 2], [2, 1]]:
                s_corr[i] = rho * beta[i] * beta[j] * sigma_x[i] * sigma_x[j] + 0.5 * rho**2 * (s_uncorr[j] - s_uncorr[i])
    
            indices = (s_uncorr + s_corr)/ var_y
            return indices 
        elif dim == 2:
            indices = np.asarray([1./dim]*dim)
            return indices
        else:
            return self._shapley_indices

    @shapley_indices.setter
    def shapley_indices(self, indices):
        self._shapley_indices = indices

def additive_func(x, beta=None):
    """
    """
    x = np.asarray(x)
    if x.ndim == 1:
        dim = x.shape[0]
    else:
        n_sample, dim = x.shape

    if beta is None:
        beta = np.ones((dim, ))
    else:
        beta = np.asarray(beta)
    y = np.dot(x, beta)
    return y

def ishigami_func(x, a=7, b=0.1):
    """Ishigami function.

    Parameters
    ----------
    x : array,
        The input variables. The shape should be 3 x n.

    Returns
    -------
    y : int or array,
        The function output.
    """
    x = np.asarray(x).squeeze()
    if x.ndim == 1:
        ndim = x.shape[0]
    else:
        ndim = x.shape[1]

    assert ndim == 3, "Dimension problem %d != %d " % (3, ndim)

    if ndim == 1:
        y = np.sin(x[0]) + a*np.sin(x[1])**2 + b*x[2]**4 * np.sin(x[0])
    else:
        y = np.sin(x[:, 0]) + a*np.sin(x[:, 1])**2 + b*x[:, 2]**4 * np.sin(x[:, 0])

    return y
