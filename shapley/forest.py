import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

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
        self.reg_rf = None

    def build(self, n_estimators=10, method='random-forest'):
        """
        """
        if method == 'random-forest':
            regressor = RandomForestRegressor(n_estimators=n_estimators, max_depth=None)
        elif method == 'extra-tree':
            regressor = ExtraTreesRegressor(n_estimators=n_estimators)
            
        self.reg_rf = regressor.fit(self.input_sample, self.output_sample)

        def meta_model(X, n_estimators):
            if self.reg_rf is None or self.reg_rf.n_estimators != n_estimators:
                self.reg_rf = RandomForestRegressor(n_estimators=n_estimators).fit(self.input_sample, self.output_sample)
        
            n_sample = X.shape[0]
            y = np.zeros((n_sample, n_estimators))
            for i, tree in enumerate(self.reg_rf.estimators_):
                y[:, i] = tree.predict(X)

            return y

        self.predict = self.reg_rf.predict
        self.model_func = meta_model