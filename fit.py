X = df[df.city=="Санкт-Петербург"].copy()
y = df[df.city=="Санкт-Петербург"]['target']

# Fit models with Optimal Parameters
models_to_fit = ['elastic','lars','huber','randomforest','boosting']
models = dict()
models['elastic']       =  ElasticNet(normalize=True, alpha=, l1_ratio=)
models['lars']          =  Lars(n_nonzero_coefs=, eps=)
models['huber']         =  HuberRegressor(epsilon, alpha=)
models['randomforest']  =  RandomForestRegressor(max_depth, n_estimators=)
models['boosting']      =  GradientBoostingRegressor(max_depth, n_estimators=)

for model in models_to_fit:
    model.fit(X[polynomial_features],y)
    X[model+'_pred'] = model.predict(X[polynomial_features])
    X_test[model+'_pred'] = model.predict(X_test.to_frame().T[polynomial_features])
    print(get_error(X.linear_pred, y))

