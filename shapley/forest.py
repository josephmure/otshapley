import numpy as np
import openturns as ot
import random
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from skopt import BayesSearchCV
from skopt.space import Integer

from .model import MetaModel
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
        
def compute_perm_indices(rfq, X, y, dist, indice_type='full'):
    """
    """
    dim = rfq.n_features_
    n_tree = rfq.n_estimators
    trees = rfq.estimators_
    n_pairs = int(dim * (dim-1)/2)
    oob_idx = np.invert(rfq.y_weights_.astype(bool))

    if indice_type == 'full':
        dev = 0
    elif indice_type == 'ind':
        dev = 1
    else:
        raise(ValueError('Unknow indice_type: {0}').format(indice_type))

    margins = [ot.Distribution(dist.getMarginal(j)) for j in range(dim)]
    order = []
    order_inv = []
    transform = []
    inv_transform = []
    for i in range(dim):
        # We consider the rotations for the Rosenblatt Transformation (RT)
        order.append(np.roll(range(dim), -i))
        order_inv.append(np.roll(range(dim), i))
        order_cop = np.roll(range(n_pairs), i)

        # Rotation of the margins and the copula
        # TODO: check the rotation for the copula
        margins_i = [margins[j] for j in order[i]]
        copula_i = ot.Copula(dist.getCopula())
        params_i = np.asarray(copula_i.getParameter())[order_cop]
        copula_i.setParameter(params_i)

        # Create the distribution and build the RTs
        dist_i = ot.ComposedDistribution(margins_i, copula_i)
        transform.append(lambda u: np.asarray(dist_i.getIsoProbabilisticTransformation()(u)))
        inv_transform.append(lambda u: np.asarray(dist_i.getInverseIsoProbabilisticTransformation()(u)))

    first_indices = np.zeros((dim, n_tree))
    total_indices = np.zeros((dim, n_tree))

    for t, tree in enumerate(trees):
        X_tree = X[oob_idx[t]]
        y_tree = y[oob_idx[t]]
        var_y_tree = y_tree.var()
        y_pred_tree = tree.predict(X_tree)
        error = ((y_tree - y_pred_tree)**2).mean()
        #TODO: it can be better by taking out of the loop the permutations
        # which are computed for each tree
        for i in range(dim):
            # The following lines transform the sample to be in an iso
            # probabilistic space. Then doing the permutation in the
            # uncorrelated space and going back in the normal space.
            # This steps makes the permutation possible without changing the 
            # input law

            # Iso transformation
            U_tree = transform[i](X_tree[:, order[i]])
            U_tree_i_total = U_tree.copy()
            U_tree_i_first = U_tree.copy()

            # Permutation of the 1st column (due to rearangement) (total indices)
            U_tree_i_total[:, 0-dev] = np.random.permutation(U_tree[:, 0-dev])
            U_tree_i_first[:, (1-dev):(dim-dev)] = np.random.permutation(U_tree[:, (1-dev):(dim-dev)])
            
            # Inverse Iso transformation
            X_tree_i_total = inv_transform[i](U_tree_i_total)
            X_tree_i_first = inv_transform[i](U_tree_i_first)
            
            # Reordering of the features
            X_tree_i_total = X_tree_i_total[:, order_inv[i]]
            X_tree_i_first = X_tree_i_first[:, order_inv[i]]

            # Computes the error with the permuted variable
            y_pred_tree_i_total = tree.predict(X_tree_i_total)
            y_pred_tree_i_first = tree.predict(X_tree_i_first)
            error_i_total = ((y_tree - y_pred_tree_i_total)**2).mean()
            error_i_first = ((y_tree - y_pred_tree_i_first)**2).mean()

            # The total sobol indices
            total_indices[i-dev, t] = (error_i_total - error) / (2*var_y_tree)
            first_indices[i-dev, t] = 1. - (error_i_first - error) / (2*var_y_tree)

    results_permutation = SensitivityResults(first_indices=first_indices.reshape(dim, n_tree, 1),
                                             total_indices=total_indices.reshape(dim, n_tree, 1))
    return results_permutation


def tree_indice(tree, X_tree, y_tree):
    """
    """
    var_y_tree = y_tree.var()
    y_pred_tree = tree.predict(X_tree)
    error = ((y_tree - y_pred_tree)**2).mean()
    #TODO: it can be better by taking out of the loop the permutations
    # which are computed for each tree
    for i in range(dim):
        # The following lines transform the sample to be in an iso
        # probabilistic space. Then doing the permutation in the
        # uncorrelated space and going back in the normal space.
        # This steps makes the permutation possible without changing the 
        # input law

        # Iso transformation
        U_tree = transform[i](X_tree[:, order[i]])
        U_tree_i_total = U_tree.copy()
        U_tree_i_first = U_tree.copy()

        # Permutation of the 1st column (due to rearangement) (total indices)
        U_tree_i_total[:, 0-dev] = np.random.permutation(U_tree[:, 0-dev])
        U_tree_i_first[:, (1-dev):(dim-dev)] = np.random.permutation(U_tree[:, (1-dev):(dim-dev)])
            
        # Inverse Iso transformation
        X_tree_i_total = inv_transform[i](U_tree_i_total)
        X_tree_i_first = inv_transform[i](U_tree_i_first)
            
        # Reordering of the features
        X_tree_i_total = X_tree_i_total[:, order_inv[i]]
        X_tree_i_first = X_tree_i_first[:, order_inv[i]]

        # Computes the error with the permuted variable
        y_pred_tree_i_total = tree.predict(X_tree_i_total)
        y_pred_tree_i_first = tree.predict(X_tree_i_first)
        error_i_total = ((y_tree - y_pred_tree_i_total)**2).mean()
        error_i_first = ((y_tree - y_pred_tree_i_first)**2).mean()

        # The total sobol indices
        total_indices[i-dev, t] = (error_i_total - error) / (2*var_y_tree)
        first_indices[i-dev, t] = 1. - (error_i_first - error) / (2*var_y_tree)


def compute_shap_indices(rfq, X, y, dist):
    dim = rfq.n_features_
    n_tree = rfq.n_estimators
    oob_idx = np.invert(rfq.y_weights_.astype(bool))
    var_y = y.var()

    perms = list(ot.KPermutations(dim, dim).generate())
    n_perms = len(perms)
    perm_indices = np.zeros((dim, n_tree))
    c_hat = np.zeros((n_perms, dim, n_tree))
    variance = np.zeros((n_tree, ))
    n_pairs = int(dim * (dim-1) / 2)

    for t, tree in enumerate(rfq.estimators_):
        X_tree = X[oob_idx[t]]
        y_tree = y[oob_idx[t]]
        var_y_tree = y_tree.var()
        variance[t] = var_y_tree
        y_pred_tree = tree.predict(X_tree)
        r2 = ((y_tree - y_pred_tree)**2).mean()
        for i_p, perm in enumerate(perms):
            # permutation

            order_i = perm
            order_i_inv = [list(order_i).index(j) for j in range(dim)]
            order_cop = order_i_inv
            for i in range(dim - 1):
                idx = perm[:i + 1]

                margins = [ot.Distribution(dist.getMarginal(j)) for j in range(dim)]
                copula = ot.Copula(dist.getCopula())

                margins_i = [margins[j] for j in order_i]
                params_i = np.asarray(copula.getParameter())[order_cop]

                copula.setParameter(params_i)
                dist_i = ot.ComposedDistribution(margins_i, copula)
                trans_i = dist_i.getIsoProbabilisticTransformation()
                U_tree = np.asarray(trans_i(X_tree[:, order_i]))

                # 3) Inverse Transformation
                tmp = dist_i.getInverseIsoProbabilisticTransformation()
                inv_rosenblatt_transform_i = lambda u: np.asarray(tmp(u))

                U_tree_i = U_tree.copy()
                U_tree_i[:, :i + 1] = np.random.permutation(U_tree_i[:, :i + 1])
                X_tree_i = inv_rosenblatt_transform_i(U_tree_i)
            
                X_tree_i = X_tree_i[:, order_i_inv]


                y_pred_tree_i = tree.predict(X_tree_i)
                r2_i = ((y_tree - y_pred_tree_i)**2).mean()
                perm_indice = (r2_i - r2)
                c_mean_var = perm_indice/2.
                c_hat[i_p, i, t] = c_mean_var
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