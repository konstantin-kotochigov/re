import json
import pandas
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression,  LassoLarsCV, ElasticNet
from sklearn.metrics import mean_squared_error
from pandas.io.json import json_normalize
from sklearn.metrics import make_scorer, explained_variance_score
from sklearn.feature_selection import f_regression
import math
import numpy
from tqdm import tqdm

# Load Data
cian_data = json.load(open("re/cian_piter.json"))
df = pandas.read_csv("re/piter_df.csv", sep=";")

# Check nulls!
# df = df[df.place17_nearest.isna()==False]

# Create Inverse Map
line = cian_data[0]
data = line['data']
places = data['places']
places_dict = dict()
places_inverse = dict()
for (i,x) in enumerate(places.keys()):
    places_dict["place"+str(i)+"_100"] = []
    places_dict["place"+str(i)+"_500"] = []
    places_dict["place"+str(i)+"_1000"] = []
    places_dict["place"+str(i)+"_nearest"] = []
    places_inverse["place"+str(i)+"_100"] = x
    places_inverse["place"+str(i)+"_500"] = x
    places_inverse["place"+str(i)+"_1000"] = x
    places_inverse["place"+str(i)+"_nearest"] = x

places_converter = {v:k[0:k.index("_")] for k,v in places_inverse.items()}



# Create List of Model Features
features = list(df.columns)
features.remove('address')
features.remove('city')
features.remove('geo_lat')
features.remove('geo_lon')
features.remove('geo_lat_2')
features.remove('geo_lon_2')
features.remove('target')
# features.remove('building_buildYear') # we converted them
features.remove('building_material')
features.remove('link')
features.remove('block_id')







# Custom quality measure
def get_error(y, y_pred):
    t = abs(y_pred - y) / y
    return (len(t[t<0.2])/len(t))

X = df[df.city=="Санкт-Петербург"].copy()
y = df[df.city=="Санкт-Петербург"]['target']



# Model only on average price
get_error(y, X.geo_ads_mean)








# Delete rare features
col_objects = {}
min_instances = 10
for x in features:
    col_objects[x]=sum(df[x]>0)

col_objects_df = pandas.DataFrame.from_dict(col_objects, orient="index", columns=["cnt"]).sort_values("cnt")
features = list(col_objects_df.index[col_objects_df.cnt>min_instances])

# Compute manual feature Impotance
col_type = {"nominal":[], "numeric":[]}
for x in features:
    if df[x].nunique() < 5:
        col_type["nominal"].append(x)
    else:
        col_type["numeric"].append(x)




# Check dependencies for School atribute
test = X.groupby("place15_500", as_index=False)['target'].agg({"mean","count","std"})
import matplotlib
matplotlib.use('agg')
plt.switch_backend('agg')
import matplotlib.pyplot as plt
plt.bar(list(test.index), test["mean"].values, yerr=test['std'].values)
plt.savefig("re/plots/school_dict_500.png", dpi=300)


col_f_importance = pandas.DataFrame({"feature":col_type["numeric"], "f_measure":f_regression(X[col_type["numeric"]], y)[0]}).sort_values("f_measure", ascending=False)
col_f_importance['feature_name'] = col_f_importance.feature.map(places_inverse)

# Simple regression
regression = LinearRegression()
col_importance = {"coeff":{}, "intercept":{}}
for x in col_type["numeric"]:
    col_importance['coeff'][x] = '{:f}'.format(regression.fit(X[[x]], y).coef_[0])
    # col_importance['intercept'][x] = regression.fit(X[[x]], y).intercept_

col_f_importance['regr1_coeff'] = col_f_importance.feature.map(col_importance['coeff'])
# col_f_importance['intercept'] = col_f_importance.feature.map(col_importance['intercept'])

# Two-way regression
regression = LinearRegression()
col_importance = {"coeff":{}, "intercept":{}}
for x in col_type["numeric"]:
    result = regression.fit(X[[x,"geo_ads_mean"]], y)
    col_importance['coeff'][x] = r'{:f}'.format(result.coef_[0])
    # col_importance['intercept'][x] = result.intercept_

col_f_importance['regr2_coef'] = col_f_importance.feature.map(col_importance['coeff'])
# col_f_importance['regr2_intercept'] = col_f_importance.feature.map(col_importance['intercept'])








# Set Features for Models

places_features = [x for x in X.columns if x.startswith("place") and not x.endswith("_nearest")]

# Binarize place features
binary_places_features = []
for x in tqdm(places_features, total=len(places_features), unit="features"):
    X[x+"_bin"] = numpy.where(X[x] > 0, 1, 0)
    binary_places_features.append(x+"_bin")

coordinate_features = ['geo_lat_1','geo_lon_1','geo_ring']
linear_features = [x for x in features if x not in coordinate_features]
polynomial_features = linear_features + list(map(lambda x: x+"_square", linear_features)) + list(map(lambda x: x+"_sqrt", linear_features)) 
#   + binary_places_features



# Random Forest Model
group_kfold = GroupKFold(n_splits=5)
cvs = group_kfold.split(X, y, X["block_id"])

rf = RandomForestRegressor()
parameters = {'max_depth':(5,10,15), 'n_estimators':[100,1000]}
cv = GridSearchCV(rf, parameters, cv=cvs, verbose=2, scoring=make_scorer(get_error))
cv.fit(X[features],y)
cv_results = cv.cv_results_
cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
cv_table.to_csv("re/gridsearch/coordinates_randomforest.csv", index=False)


# Linear Model



# Generate features
for linear_feature in tqdm(linear_features, total=len(linear_features), unit="features"):
    # print(linear_feature)
    X[linear_feature+"_square"] = X[linear_feature]**2
    X[linear_feature+"_sqrt"] = X[linear_feature].apply(math.sqrt)
    X_test[linear_feature+"_sqrt"] = math.sqrt(X_test[linear_feature]) 
    X_test[linear_feature+"_square"] = X_test[linear_feature]**2
    # X[linear_feature+"_log"] = X[linear_feature].apply(lambda x: math.log(x+0.01))

# Optimize params for linear model
parameters = {'normalize':[True], 'alpha':[1.0, 1.5, 2.0], 'l1_ratio':[1.0]}
lr_cv = GridSearchCV(ElasticNet(), parameters, cv=5, verbose=2, scoring=make_scorer(get_error))
model = lr_cv.fit(X[polynomial_features], y)
nonzero_features = sorted([x for x in list(zip(polynomial_features, lr_cv.best_estimator_.coef_)) if x[1]!=0], key=lambda tup: tup[1])
for x in nonzero_features:
    print(x[0],x[1])
cv_results = lr_cv.cv_results_
cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
cv_table.to_csv("re/gridsearch/elasticnet.csv", index=False)

# model linear estimate
lr = ElasticNet(alpha=2.0, l1_ratio=1.0, normalize=True)
lr.fit(X[polynomial_features], y)
X['linear_pred'] = lr.predict(X[polynomial_features])
X_test['linear_pred'] = lr.predict(X_test.to_frame().T[polynomial_features])

get_error(X.linear_pred, y)

rf = RandomForestRegressor(max_depth=10, n_estimators=1000, n_jobs=-1)
rf.fit(X[linear_features],y)
X_test['rf_pred'] = rf.predict(X_test.to_frame().T[linear_features])[0]


