import pandas
from sklearn.linear_model import ElasticNet, Lars
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import make_scorer, explained_variance_score

class Optimizer:
    def __init__(self, optimize_groupwise):
        self.optimize_groupwise = optimize_groupwise
        if self.optimize_groupwise == False:
            self.optimizer_cv = 5
        else:
            self.optimizer_cv = GroupKFold(n_splits=5).split(X, y, X["block_id"])
    def optimize_classifier(self, classifier, X, y, param_grid):
        lr_cv = GridSearchCV(classifier, param_grid, cv=self.optimizer_cv, verbose=2, scoring=make_scorer(get_error))
        model = lr_cv.fit(X, y)
        cv_results = lr_cv.cv_results_
        cv_table = pandas.DataFrame({"algo":classifier.__class__.__name__,"param":cv_results['params'], "error":cv_results['mean_test_score'], "error_std":cv_results['std_test_score']}).sort_values(by="error", ascending=False)
        return (cv_table)

# ----------------------------------------------------------------------------------------------------------------------

opt = Optimizer(optimize_groupwise=False)

# List Param Grids
lr_param_grid = {'normalize':[True], 'alpha':[1.0, 1.5, 2.0], 'l1_ratio':[1.0]}
lars_param_grid = {'n_nonzero_coefs': [10, 50, 100, 150, 200], 'eps': [0.1, 0.5, 1.0, 1.5, 2.0]}

optimization_X = X[polynomial_features]
optimization_y = y

# Optimize
lr_cv_table = opt.optimize_classifier(param_grid=lr_param_grid, classifier=ElasticNet(), X=optimization_X, y=optimization_y)
lars_cv_table = opt.optimize_classifier(param_grid=lars_param_grid, classifier=Lars(), X=optimization_X, y=optimization_y)

cv_table = pandas.concat([lr_cv_table, lars_cv_table])
cv_table.to_csv("re/gridsearch/cv_table.csv", sep=";", index=False)