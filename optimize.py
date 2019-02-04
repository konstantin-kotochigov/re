import pandas
from sklearn.linear_model import ElasticNet, Lars, HuberRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
        cv_table = pandas.DataFrame({"algo":classifier.__class__.__name__,"param":cv_results['params'], "quality":cv_results['mean_test_score'], "error_std":cv_results['std_test_score']}).sort_values(by="quality", ascending=False)
        return (cv_table)

# ----------------------------------------------------------------------------------------------------------------------


optimization_features = polynomial_features + ['geo_ads_mean','geo_ring','building_buildYear','building_floors','geo_underground_dist','geo_underground_new','building_parking','building_totalArea','building_material_block', 'building_material_brick', 'building_material_monolith', 'building_material_monolithBrick', 'building_material_old', 'building_material_panel', 'building_material_stalin', 'building_material_wood']

X = df[df.city=="Санкт-Петербург"].copy()
y = df[df.city=="Санкт-Петербург"]['target']

optimization_X = X[optimization_features]
optimization_y = y

# Define three parameter Grids
param_grid = dict()
param_grid['elastic'] = {'normalize':[True], 'alpha':[1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0], 'l1_ratio':[1.0]}
param_grid['lars'] = {'n_nonzero_coefs': [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210], 'eps': [0.1, 0.5, 1.0, 1.5, 2.0]}
param_grid['huber'] = {'epsilon':[1.0, 1.35, 2.0], 'alpha':[0.0001, 0.001, 0.01, 0.1]}
param_grid['randomforest'] = {'max_depth':(5,10,15), 'n_estimators':[100,1000]}
param_grid['boosting'] = {'max_depth':(5,10,15), 'n_estimators':[100,1000]}

# Models
models = dict()
models['elastic'] = ElasticNet()
models['lars'] = Lars()
models['huber'] = HuberRegressor()
models['randomforest'] = RandomForestRegressor(n_jobs=-1)
models['boosting'] = GradientBoostingRegressor()

# Optimize Models
models_to_fit = ['elastic','lars','huber']
for optimize_groupwise in [False]:
    opt = Optimizer(optimize_groupwise)
    cv_table = pandas.DataFrame()
    for model in models_to_fit:
        cv_table = cv_table.append(opt.optimize_classifier(param_grid=param_grid[model], classifier=models[model], X=optimization_X, y=optimization_y))
    cv_table.sort_values(by="quality").to_csv("re/gridsearch/cv_table_pointwise_"+str(optimize_groupwise)+".csv", sep=";", index=False)



# ----------------------------------------------------------------------------------------------------------------------


param_grid_with_selection = dict()
param_grid_with_selection['elastic'] = {'elastic__normalize':[True], 'elastic__alpha':[1.0, 1.5, 2.0], 'elastic__l1_ratio':[1.0], 'kbest__k':[10,50,100,150,200]}
param_grid_with_selection['lars'] = {'lars__n_nonzero_coefs': [10, 50, 100, 150, 200], 'lars__eps': [0.1, 0.5, 1.0, 1.5, 2.0], 'kbest__k':[10,50,100,150,200]}
param_grid_with_selection['huber'] = {'epsilon':[1.0, 1.35, 2.0], 'alpha':[0.0001, 0.001, 0.01, 0.1]}

models_with_selection = dict()
models_with_selection['elastic'] = Pipeline([("kbest",SelectKBest()), ("elastic",ElasticNet())])
models_with_selection['lars']    = Pipeline([("kbest",SelectKBest()), ("lars",Lars())])
models_with_selection['huber']   = Pipeline([("kbest",SelectKBest()), ("huber",HuberRegressor())])

# Optimize Models with SelectKBest
models_to_fit = ['huber']
for optimize_groupwise in [False, True]:
    opt = Optimizer(optimize_groupwise)
    cv_table = pandas.DataFrame()
    for model in models_to_fit:
        cv_table.append(opt.optimize_classifier(param_grid=param_grid_with_selection[model], classifier=models_with_selection[model], X=optimization_X, y=optimization_y))
    cv_table.to_csv("re/gridsearch/cv_table_pointwise_"+str(optimize_groupwise)+"_with_selection.csv", sep=";", index=False)


# ----------------------------------------------------------------------------------------------------------------------



param_grid_with_pca = dict()
param_grid_with_pca['elastic'] = {'elastic__normalize':[True], 'elastic__alpha':[1.0, 1.5, 2.0], 'elastic__l1_ratio':[1.0], 'kbest__k':[10,50,100,150,200]}
param_grid_with_pca['lars'] = {'lars__n_nonzero_coefs': [10, 50, 100, 150, 200], 'lars__eps': [0.1, 0.5, 1.0, 1.5, 2.0], 'kbest__k':[10,50,100,150,200]}
param_grid_with_pca['huber'] = {'epsilon':[1.0, 1.35, 2.0], 'alpha':[0.0001, 0.001, 0.01, 0.1]}

models_with_pca = dict()
models_with_pca['elastic'] = Pipeline([("kbest",PCA()), ("elastic",ElasticNet())])
models_with_pca['lars'] = Pipeline([("kbest",PCA()), ("elastic",Lars())])
models_with_pca['huber'] = Pipeline([("kbest",PCA()), ("elastic",HuberRegressor())])

# Optimize Models with PCA
for optimize_groupwise in [False, True]:
    opt = Optimizer(optimize_groupwise)
    cv_table = pandas.DataFrame()
    models_to_fit = ['huber']
    for model in models_to_fit:
        cv_table.append(opt.optimize_classifier(param_grid=param_grid_with_pca[model], classifier=models_with_pca[model], X=optimization_X, y=optimization_y))
    cv_table.to_csv("re/gridsearch/cv_table_pointwise_"+str(optimize_groupwise)+"_with_pca.csv", sep=";", index=False)




