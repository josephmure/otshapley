import numpy as np
import openturns as ot
import random
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
        n_sample = len(y_tree)
        # permutation
        for i in range(dim):
            X_tree_i = X_tree.copy()
            np.random.shuffle(X_tree_i[:, i])
            #X_tree_i[:, i] = X_tree_i[np.random.randint(0, n_sample), i]
            y_pred_tree_i = tree.predict(X_tree_i)
            r2_i = ((y_tree - y_pred_tree_i)**2).mean()
            perm_indices[i, t] = (r2_i - r2) / (2*var_y_tree)
            #perm_indices[i, t] = (r2_i - r2) / (2*var_y)
#             perm_indices[i, t] = (r2_i - r2)
            
    return perm_indices


def perm_ind_tree(tree, X_tree, y_tree):
    dim = X_tree.shape[1]
    var_y_tree = y_tree.var()
    y_pred_tree = tree.predict(X_tree)
    r2 = ((y_tree - y_pred_tree)**2).mean()
    # permutation
    perm_indices = np.zeros((dim, ))
    for i in range(dim):
        X_tree_i = X_tree.copy()
        np.random.shuffle(X_tree_i[:, i])
        y_pred_tree_i = tree.predict(X_tree_i)
        r2_i = ((y_tree - y_pred_tree_i)**2).mean()
        perm_indices[i] = (r2_i - r2) / (2*var_y_tree)
    return perm_indices
        
def compute_shap_indices(rfq, X, y):
    dim = rfq.n_features_
    n_tree = rfq.n_estimators
    oob_idx = np.invert(rfq.y_weights_.astype(bool))
    var_y = y.var()

    perms = list(ot.KPermutations(dim, dim).generate())
    n_perms = len(perms)
    perm_indices = np.zeros((dim, n_tree))
    c_hat = np.zeros((n_perms, dim, n_tree))
    variance = np.zeros((n_tree, ))

    for t, tree in enumerate(rfq.estimators_):
        X_tree = X[oob_idx[t]]
        y_tree = y[oob_idx[t]]
        var_y_tree = y_tree.var()
        variance[t] = var_y_tree
        y_pred_tree = tree.predict(X_tree)
        r2 = ((y_tree - y_pred_tree)**2).mean()
        for i_p, perm in enumerate(perms):
            # permutation
            for j in range(dim - 1):
                idx = perm[:j + 1]
                X_tree_i = X_tree.copy()
                X_tree_i[:, idx] = np.random.permutation(X_tree_i[:, idx])
                y_pred_tree_i = tree.predict(X_tree_i)
                r2_i = ((y_tree - y_pred_tree_i)**2).mean()
                perm_indice = (r2_i - r2)
                c_mean_var = perm_indice/2.
                c_hat[i_p, j, t] = c_mean_var
            c_hat[i_p, -1, t] = var_y_tree

    # Cost variation
    delta_c = c_hat.copy()
    delta_c[:, 1:] = c_hat[:, 1:] - c_hat[:, :-1]
        
    shapley_indices = np.zeros((dim, n_tree))
    # Estimate Shapley, main and total Sobol effects
    for i_p, perm in enumerate(perms):
        # Shapley effect
        shapley_indices[perm] += delta_c[i_p]

    shapley_indices = shapley_indices / n_perms / variance.reshape(1, n_tree)
            
    return shapley_indices