import numpy as np
import openturns as ot
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from skgarden import RandomForestQuantileRegressor
from sklearn.ensemble.forest import _generate_unsampled_indices
from skopt.space import Integer

from .model import MetaModel, change_distribution, sample_dist
from .utils import q2_cv
from .indices import SensitivityResults


class RandomForestModel(MetaModel):
    """Class to build a random forest model.

    Parameters
    ----------
    model : callable
        The true function.

    input_distribution : ot.DistributionImplementation
        The input distribution for the sampling of the observations.
    """

    def __init__(self, model=None, input_distribution=None):
        MetaModel.__init__(self, model=model,
                           input_distribution=input_distribution)
        self.reg_rf = None

    def build(self, n_estimators=10, method='random-forest', n_iter_search=None, n_fold=3, min_samples_leaf=1, n_jobs=-1, 
        max_features='auto', oob_score=False, random_state=None):
        """
        """
        if method == 'random-forest':
            regressor = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                              n_jobs=n_jobs, max_features=max_features, oob_score=oob_score, random_state=random_state)
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
                self.reg_rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True).fit(
                    self.input_sample, self.output_sample)

            n_sample = X.shape[0]
            y = np.zeros((n_sample, n_estimators))
            for i, tree in enumerate(self.reg_rf.estimators_):
                y[:, i] = tree.predict(X)
            return y

        self.predict = self.reg_rf.predict
        self.model_func = meta_model

    def compute_score_q2_cv(self, n_sample=100, sampling='lhs', sampling_type='classic', alpha=0.99, by_tree=False):
        """Cross Validation estimation of Q2.
        """
        dist = change_distribution(
            self._input_distribution, sampling_type, alpha)
        x = sample_dist(dist, n_sample, sampling)

        ytrue = self.true_model(x)
        if by_tree:
            ntree = self.reg_rf.n_estimators
            q2 = np.zeros((ntree, ))
            for i, tree in enumerate(self.reg_rf.estimators_):
                ypred = tree.predict(x)
                q2[i] = q2_cv(ytrue, ypred)
        else:
            ypred = self.predict(x)
            q2 = q2_cv(ytrue, ypred)
        self.score_q2_cv = q2
        return q2


def get_pos(dim, j1, j2):
    """
    """
    k = 0
    for i in range(1, dim):
        for j in range(i):
            if i == j1 and j == j2:
                return k
            elif i == j2 and j == j1:
                return k
            k += 1
