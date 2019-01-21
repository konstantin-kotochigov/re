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
features.remove('building_buildYear') # we converted them
features.remove('building_material')
features.remove('link')
features.remove('block_id')

``





# Custom quality measure
def get_error(y, y_pred):
    t = abs(y_pred - y) / y
    return (len(t[t<0.2])/len(t))

X = df[df.city=="Санкт-Петербург"]
y = df[df.city=="Санкт-Петербург"]['target']



# Model only on average price
get_error(y, X.geo_ads_mean)



# Generate point of interest
# X_test = df[df.address=="Россия, Санкт-Петербург , Октябрьская набережная, д 80 к 1"]
X_test = df[df.address=="Санкт-Петербург, Невский район, Октябрьская наб. 98 к1"].iloc[0]

# Загрузить тестовую строчку для Питера
test_places_file = open("re/places.json")
test_places = json.loads(test_places_file.readline())[0]
test_places_file.close()

test_string = {}
for place_attribute, place_data in test_places.items():
    test_string[places_converter[place_attribute]+"_100"] = place_data['cntInRadius']['100']
    test_string[places_converter[place_attribute]+"_500"] = place_data['cntInRadius']['500']
    test_string[places_converter[place_attribute]+"_1000"] = place_data['cntInRadius']['1000']
    test_string[places_converter[place_attribute]+"_nearest"] = place_data['minDistance']['value']

for k,v in test_string.items():
    X_test.loc[k] = v

X_test['building_buildYear'] = 0.0 # New
X_test['building_totalArea'] = 35 # Как считать?
X_test['geo_ring'] = 2000
X_test['geo_lat'] = 59.872067
X_test['geo_lon'] = 30.474787
X_test['geo_lat_1'] = 59.9
X_test['geo_lon_1'] = 30.5
X_test['building_parking'] = 1
X_test['building_material_monolith'] = 1
X_test['building_material_brick'] = 0
# X_test['geo_ads_mean'] = 
# X_test['geo_ads_count'] =
X_test['building_passengerLiftsCount'] = 1
X_test['building_floors'] = 10
X_test['building_cargoLiftsCount'] = 0
X_test['block_id'] = "59.930.5"


coordinate_features = ['geo_lat_1','geo_lon_1','geo_ring']


# Get coordinate model params
# clf = GradientBoostingRegressor(loss='ls')
# parameters = {'max_depth':(10, 12, 14, 16), 'n_estimators':[100,250]}
# cv = GridSearchCV(clf, parameters, cv=5, verbose=2, scoring=make_scorer(get_error))


# coordinate_features = ['geo_lat_1','geo_lon_1']
# cv.fit(X[coordinate_features],y)
# cv_results = cv.cv_results_
# cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
# cv_table.to_csv("re/gridsearch/coordinates_boosting.csv", index=False)

# rf = RandomForestRegressor()
# parameters = {'max_depth':(12,14), 'n_estimators':[100]}
# cv = GridSearchCV(rf, parameters, cv=5, verbose=2, scoring=make_scorer(get_error))
# cv.fit(X[coordinate_features],y)
# cv_results = cv.cv_results_
# cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
# cv_table.to_csv("re/gridsearch/coordinates_randomforest.csv", index=False)

# Model coordinate price estimate
# clf = GradientBoostingRegressor(loss='ls', max_depth=8, n_estimators=1000)
# rf = RandomForestRegressor(max_depth=12, n_estimators=100)
# rf.fit(X[coordinate_features], y)
# X['coord_pred'] = rf.predict(X[coordinate_features])
# X_test['coord_pred'] = rf.predict(X_test.to_frame().T[coordinate_features])


group_kfold = GroupKFold(n_splits=5)
cvs = group_kfold.split(X, y, X["block_id"])

rf = RandomForestRegressor()
parameters = {'max_depth':(5,10,15), 'n_estimators':[100,1000]}
cv = GridSearchCV(rf, parameters, cv=cvs, verbose=2, scoring=make_scorer(get_error))
cv.fit(X[features],y)
cv_results = cv.cv_results_
cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
cv_table.to_csv("re/gridsearch/coordinates_randomforest.csv", index=False)

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

linear_features = [x for x in features if x not in coordinate_features] + ['coord_pred']
polynomial_features = linear_features + list(map(lambda x: x+"_square", linear_features)) + list(map(lambda x: x+"_sqrt", linear_features))

# Get coordinate model params
for linear_feature in linear_features:
    print(linear_feature)
    # X[linear_feature+"_square"] = X[linear_feature]**2
    # X[linear_feature+"_sqrt"] = X[linear_feature].apply(math.sqrt)
    X_test[linear_feature+"_sqrt"] = math.sqrt(X_test[linear_feature]) 
    X_test[linear_feature+"_square"] = X_test[linear_feature]**2
    # X[linear_feature+"_log"] = X[linear_feature].apply(lambda x: math.log(x+0.01))

# Optimize params for linear model
parameters = {'normalize':[False,True], 'alpha':[0.0, 0.5, 1.0, 1.5], 'l1_ratio':[0.0, 0.25, 0.5, 0.75, 1.0]}
lr_cv = GridSearchCV(ElasticNet(), parameters, cv=5, verbose=2, scoring=make_scorer(get_error))
model = lr_cv.fit(X[polynomial_features], y)
nonzero_features = sorted([x for x in list(zip(polynomial_features, lr_cv.best_estimator_.coef_)) if x[1]!=0], key=lambda tup: tup[1])
for x in nonzero_features:
    print(x[0],x[1])
cv_results = lr_cv.cv_results_
cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
cv_table.to_csv("re/gridsearch/elasticnet.csv", index=False)

# model linear estimate
lr = ElasticNet(alpha=1.0, l1_ratio=1.0, normalize=True)
lr.fit(X[polynomial_features], y)
X['linear_pred'] = lr.predict(X[polynomial_features])
X_test['linear_pred'] = lr.predict(X_test.to_frame().T[polynomial_features])

get_error(X.linear_pred, y)


nonzero_features = sorted(list(set([polynomial_features[n].replace("_square","").replace("_sqrt","") for n,x in enumerate(lr.coef_) if x!=0.0])))

# Sort features by coefficient absolute value
from functools import reduce

# get features with nonzero coefficients
nonzero_features_with_coef = [(polynomial_features[n].replace("_square","").replace("_sqrt",""),x) for n,x in enumerate(lr.coef_) if x!=0.0]

# sort by coefficient descending
nonzero_features_sorted = [x[0] for x in sorted(nonzero_features_with_coef, key=lambda x: abs(x[1]), reverse=True)]

# remove duplicates preserving order
nonzero_features_sorted_unique = reduce(lambda l, x: l.append(x) or l if x not in l else l, nonzero_features_sorted, [])
feature_rank = {x:n for n,x in enumerate(nonzero_features_sorted_unique)}


# test_samples = X.sample(10)
# modifiable_features = [x for x in features if x not in ['geo_ring_1','geo_ring_2','geo_ring_3','geo_lat_1','goe_lon_1']]
modifiable_features = [x for x in nonzero_features if x not in ['coord_pred','geo_ads_count','geo_ads_mean']]
modifiable_integer_features = [x for x in modifiable_features if x[0:5] in "place"] + ['building_parking','building_cargoLiftsCount','building_passengerLiftsCount']
modifiable_real_features = ['building_totalArea','building_buildYear']
modifiable_cat_features = ['building_material']
# modifiable_numeric_features = modifiable_integer_features + modifiable_real_features

import matplotlib.pyplot as plt

# Get test sample for checking plots
# test_samples = X[polynomial_features].sample(10)
# test_samples = X[X.address=="Россия, Санкт-Петербург , Октябрьская набережная, д 80 к 1"]
# test_samples = test_samples[polynomial_features]

feature_predictions = dict()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

# Dict for setting limits
features_dict = dict()
for feature in modifiable_integer_features:
    features_dict[feature] = dict()
    features_dict[feature]['min'] = min(X[feature])
    features_dict[feature]['max'] = max(X[feature])
    features_dict[feature]['mean'] = round(X[feature].mean())

# Plot integer features
for feature in modifiable_integer_features:
    
    # zero_indexes = [n for n,x in enumerate(lr.coef_) if x == 0.0]
    print(feature)
    # if feature in [polynomial_features[x] for x in zero_indexes]:
    #     continue
    feature_predictions[feature] = []
    feature_range = range(int(features_dict[feature]['min']),int(features_dict[feature]['max'] + 1))
    if len(feature_range) == 1:
        continue
    for feature_value in list(feature_range):
        test_sample = X_test.copy()
        test_sample[feature] = feature_value
        test_sample[feature+"_square"] = feature_value**2
        test_sample[feature+"_sqrt"] = math.sqrt(feature_value)
        feature_predictions[feature].append(round(lr.predict(test_sample.to_frame().T[polynomial_features])[0],2))
    plt.switch_backend('agg')
    # fig, ax = plt.subplots()
    xnew = numpy.linspace(0,10,300)
    if len(feature_range) <= 2: 
        power = 1
    else:
        power = 2
    spl = make_interp_spline(list(feature_range),feature_predictions[feature], k=power)
    power_smooth = spl(xnew)
    # plt.plot(feature_predictions[feature])
    # plt.savefig("re/plots/raw" + feature + ".png")
    # plt.switch_backend('agg')
    plt.plot(xnew, power_smooth)
    if feature.startswith("place"):
        plt.title(places_inverse[feature])
    plt.savefig("re/plots/" + str(feature_rank[feature])+"_"+ feature + ".png")


# Plot categorical features
for feature in modifiable_cat_features:
    feature_predictions[feature] = []
    feature_values = [x[len(feature)+1:] for x in features if x.startswith(feature)]
    ind = numpy.arange(0, len(feature_values))
    for value in feature_values:
        test_sample = X_test.copy()
        test_sample[[feature + "_" + x for x in feature_values]] = 0
        test_sample[feature+"_"+value] = 1
        feature_predictions[feature].append(round(lr.predict(test_sample.to_frame().T[polynomial_features])[0],2))
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    plt.barh(ind, feature_predictions[feature])
    ax.set_yticks(ind)
    ax.set_yticklabels(feature_values)
    plt.savefig("re/plots/" + str(min([feature_rank.get(feature+"_"+x, 1000) for x in feature_values]))+"_" + feature + ".png")


# Check dependencies for School atribute
test = X.groupby("place15_500", as_index=False)['target'].agg({"mean","count","std"})
import matplotlib
matplotlib.use('agg')
plt.switch_backend('agg')
import matplotlib.pyplot as plt
plt.bar(list(test.index), test["mean"].values, yerr=test['std'].values)
plt.savefig("re/plots/school_dict_500.png", dpi=300)