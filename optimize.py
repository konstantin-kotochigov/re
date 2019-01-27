import pandas
from sklearn.linear_model import ElasticNet, Lars, HuberRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import make_scorer, explained_variance_score
from sklearn.pipeline import Pipeline

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
        cv_table = pandas.DataFrame({"algo":classifier.__class__.__name__,"param":cv_results['params'], "quality":cv_results['mean_test_score'], "error_std":cv_results['std_test_score']}).sort_values(by="error", ascending=False)
        return (cv_table)

# ----------------------------------------------------------------------------------------------------------------------



# Define three parameter Grids
param_grid = dict()
param_grid['elastic'] = {'normalize':[True], 'alpha':[1.0, 1.5, 2.0], 'l1_ratio':[1.0]}
param_grid['lars'] = {'lars__n_nonzero_coefs': [10, 50, 100, 150, 200], 'lars__eps': [0.1, 0.5, 1.0, 1.5, 2.0]}
param_grid['huber'] = {'epsilon':[1.0, 1.35, 2.0], 'alpha':[0.0001, 0.001, 0.01, 0.1]}

param_grid_with_selection = dict()
param_grid_with_selection['elastic'] = {'elastic__normalize':[True], 'elastic__alpha':[1.0, 1.5, 2.0], 'elastic__l1_ratio':[1.0], 'kbest__k':[10,50,100,150,200]}
param_grid_with_selection['lars'] = {'lars__n_nonzero_coefs': [10, 50, 100, 150, 200], 'lars__eps': [0.1, 0.5, 1.0, 1.5, 2.0], 'kbest__k':[10,50,100,150,200]}
param_grid_with_selection['huber'] = {'epsilon':[1.0, 1.35, 2.0], 'alpha':[0.0001, 0.001, 0.01, 0.1]}

param_grid_with_pca = dict()
param_grid_with_pca['elastic'] = {'elastic__normalize':[True], 'elastic__alpha':[1.0, 1.5, 2.0], 'elastic__l1_ratio':[1.0], 'kbest__k':[10,50,100,150,200]}
param_grid_with_pca['lars'] = {'lars__n_nonzero_coefs': [10, 50, 100, 150, 200], 'lars__eps': [0.1, 0.5, 1.0, 1.5, 2.0], 'kbest__k':[10,50,100,150,200]}
param_grid_with_pca['huber'] = {'epsilon':[1.0, 1.35, 2.0], 'alpha':[0.0001, 0.001, 0.01, 0.1]}

# Models
models = dict()
models['elastic'] = ElasticNet()
models['lars'] = Lars()
models['huber'] = HuberRegressor()

# Results
optimization_result = dict()

optimization_X = X[polynomial_features]
optimization_y = y

# Optimize Models
models_to_fit = ['huber']
for optimize_groupwise in [False, True]:
    opt = Optimizer(optimize_groupwise)
    cv_table = pandas.DataFrame()
    for model in models_to_fit:
        cv_table.append(opt.optimize_classifier(param_grid=param_grid[model], classifier=models[model], X=optimization_X, y=optimization_y))
    cv_table.to_csv("re/gridsearch/cv_table_pointwise_"+str(optimize_groupwise)+".csv", sep=";", index=False)

# Optimize Models with SelectKBest
models_to_fit = ['huber']
for optimize_groupwise in [False, True]:
    opt = Optimizer(optimize_groupwise)
    cv_table = pandas.DataFrame()
    for model in models_to_fit:
        cv_table.append(opt.optimize_classifier(param_grid=param_grid[model], classifier=models[model], X=optimization_X, y=optimization_y))
    cv_table.to_csv("re/gridsearch/cv_table_pointwise_"+str(optimize_groupwise)+".csv", sep=";", index=False)

# Optimize Models with PCA
for optimize_groupwise in [False, True]:
    opt = Optimizer(optimize_groupwise)
    cv_table = pandas.DataFrame()
    models_to_fit = ['huber']
    for model in models_to_fit:
        cv_table.append(opt.optimize_classifier(param_grid=param_grid[model], classifier=models[model], X=optimization_X, y=optimization_y))
    cv_table.to_csv("re/gridsearch/cv_table_pointwise_"+str(optimize_groupwise)+".csv", sep=";", index=False)




