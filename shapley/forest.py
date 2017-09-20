import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from skopt import BayesSearchCV
from skopt.space import Integer

from .model import MetaModel


class RandomForestModel(MetaModel):
    """Class to build a random forest model.
    
    Parameters
    ----------
    model : callable
        The true function.
        
    input_distribution : ot.DistributionImplementation
        The input distribution for the sampling of the observations.
    """
    def __init__(self, model, input_distribution):
        MetaModel.__init__(self, model=model, input_distribution=input_distribution)
        self.reg_rf = None

    def build(self, n_estimators=10, method='random-forest', n_iter_search=None, n_fold=3):
        """
        """
        if method == 'random-forest':
            regressor = RandomForestRegressor(n_estimators=n_estimators, oob_score=True)
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
                self.reg_rf = RandomForestRegressor(n_estimators=n_estimators, oob_score=True).fit(self.input_sample, self.output_sample)

            n_sample = X.shape[0]
            y = np.zeros((n_sample, n_estimators))
            for i, tree in enumerate(self.reg_rf.estimators_):
                y[:, i] = tree.predict(X)

            return y

        self.predict = self.reg_rf.predict
        self.model_func = meta_model
        
def compute_perm_indices(rfq, X, y):
    dim = rfq.n_features_
    n_tree = rfq.n_estimators
    oob_idx = np.invert(rfq.y_weights_.astype(bool))
    perm_indices = np.zeros((dim, n_tree))
    var_y = y.var()
    for t, tree in enumerate(rfq.estimators_):
        X_tree = X[oob_idx[t]]
        y_tree = y[oob_idx[t]]
        var_y_tree = y_tree.var()
        y_pred_tree = tree.predict(X_tree)
        r2 = ((y_tree - y_pred_tree)**2).mean()
        # permutation
        for i in range(dim):
            X_tree_i = X_tree.copy()
            np.random.shuffle(X_tree_i[:, i])
            y_pred_tree_i = tree.predict(X_tree_i)
            r2_i = ((y_tree - y_pred_tree_i)**2).mean()
            perm_indices[i, t] = (r2_i - r2) / (2*var_y_tree)
            perm_indices[i, t] = (r2_i - r2) / (2*var_y)
#             perm_indices[i, t] = (r2_i - r2)
            
    return perm_indices