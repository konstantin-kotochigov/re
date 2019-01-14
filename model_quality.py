# Grid Search with CV (Gradient Boosting)
clf = GradientBoostingRegressor(loss='ls')
parameters = {'max_depth':(4,6,8), 'n_estimators':[250], 'loss':['ls']}

group_kfold = GroupKFold(n_splits=5)
cvs = group_kfold.split(X, y, df["block_id"])

cv = GridSearchCV(clf, parameters, cv=cvs, verbose=2, scoring=make_scorer(get_error))
cv.fit(X,y)
cv_results = cv.cv_results_
cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)

# Grid Search with CV (RandomForest)
rf = RandomForestRegressor(criterion='mse')
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':[8], 'n_estimators':[100,1000]}
cv = GridSearchCV(rf, parameters, cv=5, verbose=2, scoring=make_scorer(get_error))
cv.fit(X,y)
cv_results = cv.cv_results_
cv_table = pandas.DataFrame("param":cv_results['params'], "error":['mean_test_error']).sort_values(by="mean_test_error", False)

# Grid Search with CV (Logistic Refression)  
rf = RandomForestRegressor(criterion='mse')
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth':[8], 'n_estimators':[100,1000]}
cv = GridSearchCV(rf, parameters, cv=5, verbose=2, scoring=make_scorer(get_error))
cv.fit(X,y)
cv_results = cv.cv_results_
cv_table = pandas.DataFrame("param":cv_results['params'], "error":['mean_test_error']).sort_values(by="mean_test_error", False)

# Show Feature Importance for best model
fi = cv.best_estimator_.feature_importances_
pandas.DataFrame(list(zip(features, fi)),columns=['feature','imp']).sort_values(by="imp", ascending=False).reset_index(drop=True)


# Check accuracy per city using manual Cross-Validation

# Model "Cities"
X = df[df.city!="Москва"][features+["city"]]
y = df[df.city!="Москва"]['target']

def get_error(y, y_pred, error_rate=0.20):
    t = abs(y_pred - y) / y
    return (len(t[t<error_rate])/len(t))

gbr = GradientBoostingRegressor(loss='ls', n_estimators=1000, max_depth=8, verbose=0)
results = []
for cv in range(5):
    print("cv="+str(cv))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=X.city)
    model = gbr.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])
    for city in list(X_test.city.unique()):
        y_test_city = y_test[X_test.city==city]
        y_pred_city = y_pred[X_test.city==city]
        results.append((cv, 
                       city, 
                       get_error(y_test_city,y_pred_city, 0.1),
                       get_error(y_test_city,y_pred_city, 0.2),
                       get_error(y_test_city,y_pred_city, 0.3), 
                       y_test_city.shape[0]
                      )
                     )
result_df = pandas.DataFrame(results, columns=['cv','city','error_10','error_20','error_30','city_count'])
result_df = result_df_cv.groupby("city", as_index=False)[['error_10','error_20','error_30','city_count']].mean()

# Model "Moscow"
X = df[df.city=="Москва"][features]
y = df[df.city=="Москва"]['target']

def get_error(y, y_pred, error_rate=0.20):
    t = abs(y_pred - y) / y
    return (len(t[t<error_rate])/len(t))

gbr = GradientBoostingRegressor(loss='ls', n_estimators=1000, max_depth=8, verbose=0)
results = []
for cv in range(5):
    print("cv="+str(cv))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    model = gbr.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])
    y_test_city = y_test
    y_pred_city = y_pred
    results.append((cv, 
                       get_error(y_test_city,y_pred_city, 0.1),
                       get_error(y_test_city,y_pred_city, 0.2),
                       get_error(y_test_city,y_pred_city, 0.3), 
                       y_test_city.shape[0]
                      )
                     )
result_df_cv = pandas.DataFrame(results, columns=['cv','city','error_20','error_30','error_50','city_count'])


# Model "All"
X = df[features+["city"]]
y = df['target']

def get_error(y, y_pred, error_rate=0.20):
    t = abs(y_pred - y) / y
    return (len(t[t<error_rate])/len(t))

gbr = GradientBoostingRegressor(loss='ls', n_estimators=1000, max_depth=8, verbose=0)
results = []
for cv in range(5):
    print("cv="+str(cv))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=X.city)
    model = gbr.fit(X_train[features], y_train)
    y_pred = model.predict(X_test[features])
    for city in list(X_test.city.unique()):
        y_test_city = y_test[X_test.city==city]
        y_pred_city = y_pred[X_test.city==city]
        results.append((cv, 
                       city, 
                       get_error(y_test_city,y_pred_city, 0.1),
                       get_error(y_test_city,y_pred_city, 0.2),
                       get_error(y_test_city,y_pred_city, 0.3), 
                       y_test_city.shape[0]
                      )
                     )
result_df_cv = pandas.DataFrame(results, columns=['cv','city','error_20','error_30','error_50','city_count'])
result_df = result_df_cv.groupby("city", as_index=False)[['error_20','error_30','error_50','city_count']].mean()


# Feature Importance using Gradient Boosting "All" model
X = df[features]
y = df['target']

gbr = GradientBoostingRegressor(loss='ls', n_estimators=1000, max_depth=8, verbose=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model = gbr.fit(X_train, y_train)

fi = model.feature_importances_
pandas.DataFrame(list(zip(features, fi)),columns=['feature','imp']).sort_values(by="imp", ascending=False).reset_index(drop=True)


# Create 10% examples
X = df[features]
y = df['target']
gbr = GradientBoostingRegressor(loss='ls', n_estimators=1000, max_depth=8, verbose=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
model = gbr.fit(X_train, y_train)
y_pred = model.predict(X_test)
X_test['Price_prediction'] = [round(x,2) for x in list(y_pred)]
X_test['Price'] = [round(x,2) for x in list(y_test)]
X_test['address'] = list(X_test.address)
X_test['error'] = round(100 * abs(X_test['Price'] - X_test['Price_prediction']) / X_test['Price'], 2)
X_test[['Price_prediction','Price','error','address','link','cnt']].to_csv("results.csv", index=False)