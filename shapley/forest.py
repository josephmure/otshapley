import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .base import Base, ProbabilisticModel, SensitivityResults, MetaModel

class RandomForestModel(MetaModel):
    """Class to build a kriging model.
    
    Parameters
    ----------
    model : callable,
        The true function.
    input_distribution : ot.DistributionImplementation,
        The input distribution for the sampling of the observations.
    """
    def __init__(self, model, input_distribution):
        self.true_model = model
        ProbabilisticModel.__init__(self, model_func=None, input_distribution=input_distribution)

    def build(self):
        """
        """
        RandomForestRegressor.fit(self.input)

    def __call__(self, X, n_realization=1):
        n_sample = X.shape[0]
        y = np.zeros((n_sample, n_realization))
        forest = RandomForestRegressor(n_estimators=n_realization)
        forest.fit(self.input_sample, self.output_sample)
        for i, tree in enumerate(forest.estimators_):
            y[:, i] = tree.predict(X, n_realization)
        return y