import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from .indices import BaseIndices, SensitivityResults
from .model import ProbabilisticModel, MetaModel


class RandomForestModel(MetaModel):
    """Class to build a random forest model.
    
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

    def build(self, n_estimators=10, method='random-forest', n_iter_search=None, n_fold=3):
        """
        """
        if method == 'random-forest':
            regressor = RandomForestRegressor(n_estimators=n_estimators)
        elif method == 'extra-tree':
            regressor = ExtraTreesRegressor(n_estimators=n_estimators)

        if n_iter_search not in [0, None]:
            search_spaces = {
                "max_features": Integer(1, self.dim),
                "min_samples_split": Integer(2, 20),
                "min_samples_leaf": Integer(1, 20)
            }
            bayes_search = BayesSearchCV(regressor, search_spaces=search_spaces,
                                               n_iter=n_iter_search, cv=n_fold, n_jobs=7)

            bayes_search.fit(self.input_sample, self.output_sample)
            self.reg_rf = bayes_search.best_estimator_
        else:
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