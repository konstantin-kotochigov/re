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

# Dummy Model only on average price
# get_error(y, X.geo_ads_mean)

# Delete rare features
col_objects = {}
min_instances = 10
for x in features:
    col_objects[x]=sum(df[x]>0)

col_objects_df = pandas.DataFrame.from_dict(col_objects, orient="index", columns=["cnt"]).sort_values("cnt")
features = list(col_objects_df.index[col_objects_df.cnt>min_instances])

# delete features with no variance



# Set Features for Models

places_features = [x for x in df.columns if x.startswith("place") and not x.endswith("_nearest")]

# Binarize place features
binary_features = []
for x in tqdm(places_features, total=len(places_features), unit="features"):
    df[x+"_bin"] = numpy.where(df[x] > 0, 1, 0)
    X_test[x+"_bin"] = numpy.where(X_test[x] > 0, 1, 0)
    binary_features.append(x+"_bin")

coordinate_features = ['geo_lat_1','geo_lon_1','geo_ring']
linear_features = [x for x in features if x not in coordinate_features]
polynomial_features = linear_features + list(map(lambda x: x+"_square", linear_features)) + list(map(lambda x: x+"_sqrt", linear_features))

# Generate features

for linear_feature in tqdm(linear_features, total=len(linear_features), unit="features"):
    # print(linear_feature)
    df[linear_feature+"_square"] = df[linear_feature]**2
    df[linear_feature+"_sqrt"] = df[linear_feature].apply(math.sqrt)
    # df[] = df[] / df[].std()
    X_test[linear_feature+"_sqrt"] = math.sqrt(X_test[linear_feature])
    X_test[linear_feature+"_square"] = X_test[linear_feature]**2
    # X[linear_feature+"_log"] = X[linear_feature].apply(lambda x: math.log(x+0.01))

from sklearn import preprocessing
df1 = pandas.DataFrame(preprocessing.scale(df[polynomial_features], columns=polynomial_features))

for x in places_features:
    if df[x].max() < 5:
        print(places_inverse[x])






