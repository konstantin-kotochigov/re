X = df[df.city=="Санкт-Петербург"].copy()
y = df[df.city=="Санкт-Петербург"]['target']

# Fit models with Optimal Parameters
models_to_fit = ['elastic','lars']
models = dict()
models['elastic']       =  ElasticNet(normalize=True, alpha=1.0, l1_ratio=1.0)
models['lars']          =  Lars(n_nonzero_coefs=50, eps=1.0)
models['huber']         =  HuberRegressor(epsilon=1.0, alpha=0.0001)
models['randomforest']  =  RandomForestRegressor(max_depth=10, n_estimators=1000)
models['boosting']      =  GradientBoostingRegressor(max_depth=10, n_estimators=1000)

model_features = optimization_features

for model in models_to_fit:
    models[model].fit(X[model_features],y)
    X[model+'_pred'] = models[model].predict(X[model_features])
    X_test[model+'_pred'] = models[model].predict(X_test[model_features])
    print(get_error(X[model + "_pred"], y))

model = models['elastic']