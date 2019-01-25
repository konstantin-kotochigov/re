# Feature Importance

# Input:
#   DF

# Output
#   Dataframe of feature importance

# Compute manual feature Impotance
col_type = {"nominal":[], "numeric":[]}
for x in features:
    if df[x].nunique() < 5:
        col_type["nominal"].append(x)
    else:
        col_type["numeric"].append(x)

col_f_importance = pandas.DataFrame({"feature": col_type["numeric"], "f_measure": f_regression(X[col_type["numeric"]], y)[0]}).sort_values("f_measure", ascending=False)
col_f_importance['feature_name'] = col_f_importance.feature.map(places_inverse)

# Simple regression
regression = LinearRegression()
col_importance = {"coeff": {}, "intercept": {}}
for x in col_type["numeric"]:
    col_importance['coeff'][x] = '{:f}'.format(regression.fit(X[[x]], y).coef_[0])
    # col_importance['intercept'][x] = regression.fit(X[[x]], y).intercept_
col_f_importance['regr1_coeff'] = col_f_importance.feature.map(col_importance['coeff'])
# col_f_importance['intercept'] = col_f_importance.feature.map(col_importance['intercept'])

# Two-way regression
regression = LinearRegression()
col_importance = {"coeff": {}, "intercept": {}}
for x in col_type["numeric"]:
    result = regression.fit(X[[x, "geo_ads_mean"]], y)
    col_importance['coeff'][x] = r'{:f}'.format(result.coef_[0])
    # col_importance['intercept'][x] = result.intercept_
col_f_importance['regr2_coef'] = col_f_importance.feature.map(col_importance['coeff'])
# col_f_importance['regr2_intercept'] = col_f_importance.feature.map(col_importance['intercept'])