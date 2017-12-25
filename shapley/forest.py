import numpy as np
import openturns as ot
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from skgarden import RandomForestQuantileRegressor
from sklearn.ensemble.forest import _generate_unsampled_indices  
#from skopt import BayesSearchCV
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


def compute_perm_indices(rfq, X, y, dist, indice_type='full', error='mse'):
    """
    """
    dim = rfq.n_features_
    n_tree = rfq.n_estimators
    trees = rfq.estimators_
    if isinstance(rfq, RandomForestQuantileRegressor):
        oob_idx = np.invert(rfq.y_weights_.astype(bool))
    elif isinstance(rfq, RandomForestRegressor):
        n_samples = X.shape[0]
        oob_idx = []
        for tree in rfq.estimators_:
            # Here at each iteration we obtain out of bag samples for every tree.
            oob_idx.append(_generate_unsampled_indices(tree.random_state, n_samples))
    else:
        raise('Unknow rfq type: {0}'.format(type(rfq)))
        
    margins = [ot.Distribution(dist.getMarginal(j)) for j in range(dim)]
    copula = dist.getCopula()
    
    first_indices = np.zeros((dim, n_tree))
    total_indices = np.zeros((dim, n_tree))
    for t, tree in enumerate(trees):
        X_tree = X[oob_idx[t]]
        y_tree = y[oob_idx[t]]
        total, first = perm_tree_sobol(tree, X_tree, y_tree, margins, copula, indice_type, error=error)
        total_indices[:, t] = total
        first_indices[:, t] = first
            
    results_permutation = SensitivityResults(first_indices=first_indices.reshape(dim, n_tree, 1),
                                             total_indices=total_indices.reshape(dim, n_tree, 1))
    return results_permutation


def perm_tree_sobol(tree, X, y, margins, copula, indice_type, error='mse'):
    """
    """
    if indice_type == 'full':
        dev = 0
    elif indice_type == 'ind':
        dev = 1
    else:
        raise(ValueError('Unknow indice_type: {0}').format(indice_type))
        
    alpha = 0.25
    
    if error == 'mse':
        cost_func = lambda y, q: (y - q)**2
    elif error == 'quantile':
        cost_func = lambda y, q: (y - q) * ((y <= q)*1. - alpha)
        
    dim = X.shape[1]
    var_y_tree = y.var()
    y_pred_tree = tree.predict(X, quantile=alpha*100.)

    error_y = cost_func(y, np.percentile(y, alpha*100)).mean()

    error_tree = cost_func(y, y_pred_tree)
    error_tree_mean = error_tree.mean()
    error_tree_var = error_tree.var()
    total_indices = np.zeros((dim, ))
    first_indices = np.zeros((dim, ))
    for i in range(dim):
        # We consider the rotations for the Rosenblatt Transformation (RT)
        order_i = np.roll(range(dim), -i)
        order_i_inv = [list(order_i).index(j) for j in range(dim)]
        
        # Get the transformations
        transform_i, inv_transform_i = get_transformations(margins, copula, order_i)
        # The following lines transform the sample to be in an iso
        # probabilistic space. Then doing the permutation in the
        # uncorrelated space and going back in the normal space.
        # This steps makes the permutation possible without changing the 
        # input law

        # Iso transformation
        U_tree = transform_i(X[:, order_i])
        U_tree_i_total = U_tree.copy()
        U_tree_i_first = U_tree.copy()
        
        # Permutation of the 1st column (due to rearangement) (total indices)
        U_tree_i_total[:, -dev] = np.random.permutation(U_tree[:, -dev])
        U_tree_i_first[:, (1-dev):(dim-dev)] = np.random.permutation(U_tree[:, (1-dev):(dim-dev)])
        
        # Inverse Iso transformation
        X_tree_i_total = inv_transform_i(U_tree_i_total)
        X_tree_i_first = inv_transform_i(U_tree_i_first)
        
        # Reordering of the features
        X_tree_i_total = X_tree_i_total[:, order_i_inv]
        X_tree_i_first = X_tree_i_first[:, order_i_inv]

        # Computes the error with the permuted variable
        y_pred_tree_i_total = tree.predict(X_tree_i_total, quantile=alpha*100.)
        y_pred_tree_i_first = tree.predict(X_tree_i_first, quantile=alpha*100.)

        error_i_total = cost_func(y, y_pred_tree_i_total).mean()
        error_i_first = cost_func(y, y_pred_tree_i_first).mean()
        
        perm_indice_i_total = error_i_total - error_tree_mean
        perm_indice_i_first = error_i_first - error_tree_mean

        # The total sobol indices
        if error == 'mse':
            total_indices[i-dev] = 0.5 * perm_indice_i_total / var_y_tree
            first_indices[i-dev] = 1. - 0.5 * perm_indice_i_first / var_y_tree
        elif error == 'quantile':
            total_indices[i-dev] = 1. - error_i_total / error_y
            first_indices[i-dev] = 1. - error_i_first / error_y 
        
    return total_indices, first_indices

def get_transformations(margins, copula, order):
    """
    """
    dim = len(margins)
    # Rotation of the margins and the copula
    order_cop = []
    for j1 in range(1, dim):
        for j2 in range(j1):
            order_cop.append(get_pos(dim, order[j1], order[j2]))
    
    margins = [margins[j] for j in order]
    copula = ot.Copula(copula)
    params = np.asarray(copula.getParameter())
#    print('Params: ', params)
#    print('order:', order)
#    print('order_cop: ', order_cop)
    params = params[order_cop]
#    print('params_cop:', params)
#    print()
    copula.setParameter(params)

    # Create the distribution and build the RTs
    dist = ot.ComposedDistribution(margins, copula)
    transform = lambda u: np.asarray(dist.getIsoProbabilisticTransformation()(u))
    inv_transform = lambda u: np.asarray(dist.getInverseIsoProbabilisticTransformation()(u))
    
    return transform, inv_transform


def compute_shap_indices(rfq, X, y, dist):
    """
    """
    dim = rfq.n_features_
    n_tree = rfq.n_estimators
    trees = rfq.estimators_
    if isinstance(rfq, RandomForestQuantileRegressor):
        oob_idx = np.invert(rfq.y_weights_.astype(bool))
    elif isinstance(rfq, RandomForestRegressor):
        n_samples = X.shape[0]
        oob_idx = []
        for tree in rfq.estimators_:
            # Here at each iteration we obtain out of bag samples for every tree.
            oob_idx.append(_generate_unsampled_indices(tree.random_state, n_samples))
    else:
        raise('Unknow rfq type: {0}'.format(type(rfq)))

    margins = [ot.Distribution(dist.getMarginal(j)) for j in range(dim)]
    copula = dist.getCopula()
    
    perms = list(ot.KPermutations(dim, dim).generate())
    n_perms = len(perms)
    c_hat = np.zeros((n_perms, dim, n_tree))
    variance = np.zeros((n_tree, ))
    
    for t, tree in enumerate(trees):
        X_tree = X[oob_idx[t]]
        y_tree = y[oob_idx[t]]
        var_y_tree = y_tree.var(ddof=1)
        variance[t] = var_y_tree
        y_pred_tree = tree.predict(X_tree)
        error = ((y_tree - y_pred_tree)**2).mean()
        c_hat[:, -1, t] = var_y_tree
        for i_p, perm in enumerate(perms):
            # We consider the rotations for the Rosenblatt Transformation (RT)
            order_i = perm
            order_i_inv = [list(order_i).index(j) for j in range(dim)]
            # Get the transformations
            transform_i, inv_transform_i = get_transformations(margins, copula, order_i)
            # Iso transformation
            U_tree = np.asarray(transform_i(X_tree[:, order_i]))
            for i in range(dim - 1):
                U_tree_i = U_tree.copy()
                # Permutation of the i-st column (due to rearangement)
                U_tree_i[:, :i+1] = np.random.permutation(U_tree_i[:, :i+1])
                
                # Inverse Iso transformation
                X_tree_i = inv_transform_i(U_tree_i)
            
                # Reordering of the features
                X_tree_i = X_tree_i[:, order_i_inv]

                y_pred_tree_i = tree.predict(X_tree_i)
                error_i = ((y_tree - y_pred_tree_i)**2).mean()
                c_hat[i_p, i, t] = (error_i - error)/2.

    # Cost variation
    delta_c = c_hat.copy()
    delta_c[:, 1:] = c_hat[:, 1:] - c_hat[:, :-1]
        
    shapley_indices = np.zeros((dim, n_tree))
    # Estimate Shapley, main and total Sobol effects
    for i_p, perm in enumerate(perms):
        # Shapley effect
        shapley_indices[perm] += delta_c[i_p]

    shapley_indices = shapley_indices / n_perms / variance
    shapley_indices_permutation = SensitivityResults(shapley_indices=shapley_indices.reshape(dim, n_tree, 1))
    return shapley_indices_permutation