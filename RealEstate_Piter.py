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

# Load Saved Data
df = pandas.read_csv("re/piter_df.csv", sep=";")
places_inverse = pandas.read_csv("re/feature_encoding.csv", sep=";", index_col=0).to_dict(orient="dict")['feature_name']

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
# group_kfold = GroupKFold(n_splits=5)
# cvs = group_kfold.split(X, y, X["block_id"])
# rf = RandomForestRegressor()
# parameters = {'max_depth':(5,10,15), 'n_estimators':[100,1000]}
# cv = GridSearchCV(rf, parameters, cv=cvs, verbose=2, scoring=make_scorer(get_error))
# cv.fit(X[features],y)
# cv_results = cv.cv_results_
# cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
# cv_table.to_csv("re/gridsearch/coordinates_randomforest.csv", index=False)


# Linear Model



# Generate features
for linear_feature in tqdm(linear_features, total=len(linear_features), unit="features"):
    # print(linear_feature)
    X[linear_feature+"_square"] = X[linear_feature]**2
    X[linear_feature+"_sqrt"] = X[linear_feature].apply(math.sqrt)
    # X_test[linear_feature+"_sqrt"] = math.sqrt(X_test[linear_feature])
    # X_test[linear_feature+"_square"] = X_test[linear_feature]**2
    # X[linear_feature+"_log"] = X[linear_feature].apply(lambda x: math.log(x+0.01))





# model linear estimate
lr = ElasticNet(alpha=2.0, l1_ratio=1.0, normalize=True)
lr.fit(X[polynomial_features], y)
X['linear_pred'] = lr.predict(X[polynomial_features])
X_test['linear_pred'] = lr.predict(X_test.to_frame().T[polynomial_features])

get_error(X.linear_pred, y)

rf = RandomForestRegressor(max_depth=10, n_estimators=1000, n_jobs=-1)
rf.fit(X[linear_features],y)
X_test['rf_pred'] = rf.predict(X_test.to_frame().T[linear_features])[0]


