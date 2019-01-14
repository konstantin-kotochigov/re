import json
import pandas
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression,  LassoLarsCV, ElasticNet
from sklearn.metrics import mean_squared_error
from pandas.io.json import json_normalize
from sklearn.metrics import make_scorer
import math
import numpy

# Extract from Postgres
from sqlalchemy import create_engine
engine = create_engine('postgresql://cian:5Dsy4LcXQVPSfKeRzEB@46.183.221.135:5432/cian')
x = pandas.read_sql_query('select * from public.cian where cian."taskId"=1 limit 10',con=engine)
x.to_json("cian_postgres.json", orient="records")

x = pandas.read_sql_query('select * from public.cian',con=engine)
x.to_json("cian_piter.json", orient="records")

# Load Data
cian_data = json.load(open("re/c.json"))

# Read Initial Data
line = cian_data[0]
data = line['data']
places = data['places']

# Create Attribute Dict for GeoMetrics
places_dict = dict()
places_inverse = dict()
for (i,x) in enumerate(places.keys()):
    places_dict["place"+str(i)+"_100"] = []
    places_dict["place"+str(i)+"_500"] = []
    places_dict["place"+str(i)+"_1000"] = []
    places_inverse["place"+str(i)+"_100"] = x
    places_inverse["place"+str(i)+"_500"] = x
    places_inverse["place"+str(i)+"_1000"] = x


# ID attributes
address = []
target = []
name = []
city = []
link = []

# Building attributes
building_floors = []
building_buildYear = []
building_materialType = []
building_cargoLiftsCount = []
building_passengerLiftsCount =[]
building_parking = []
building_totalArea = []

# Geo Attributes
geo_lat = []
geo_lon = []
geo_lat_1 = []
geo_lon_1 = []
geo_lat_2 = []
geo_lon_2 = []
geo_ring_dist_1 = []
geo_ring_dist_2 = []
geo_ring_dist_3 = []

# flatType = []
# floorNumber = []
# roomsCount = []

line_num = 0
for line in cian_data:
    
    line_num = line_num + 1
    if line_num % 1000 == 0:
        print("records read: "+str(line_num))
    
    # Inner Data Structures
    data = line['data']
    rings = line['data']['rings']
    places = line['data']['places']
    
    # ID attributes
    address.append(line['data']['geo']['userInput'])   
    geo_lat.append(line['data']['geo']['coordinates']['lat'])
    geo_lon.append(line['data']['geo']['coordinates']['lng'])
    city.append(
        ([x for x in data['geo']['address'] if x['type']=="location" and x['locationTypeId']==1] + 
        [x for x in data['geo']['address'] if x['type']=="location"])
        [0]['fullName']
    )
    link.append(line['data']['siteUrl'])
    # address_street = [x for x in data['geo']['address'] if x['type']=="street"][0]['fullName']
    # address_house = [x for x in data['geo']['address'] if x['type']=="house"][0]['fullName']
    # address.append(address_city+","+address_street+","+address_house)
    
    # Geo Attributes
    geo_lat_2.append(round(float(line['data']['geo']['coordinates']['lat']),2))
    geo_lon_2.append(round(float(line['data']['geo']['coordinates']['lng']),2))    
    geo_lat_1.append(round(float(line['data']['geo']['coordinates']['lat']),1))
    geo_lon_1.append(round(float(line['data']['geo']['coordinates']['lng']),1))
    geo_ring_dist_1.append(rings['МКАД'])
    geo_ring_dist_2.append(rings['ТТК'])
    geo_ring_dist_3.append(rings['Садовое'])
    
    # Geometrics
    for (i,x) in enumerate(places.keys()):
        places_dict["place"+str(i)+"_100"].append(places[x]['cntInRadius']['100'])
        places_dict["place"+str(i)+"_500"].append(places[x]['cntInRadius']['500'])
        places_dict["place"+str(i)+"_1000"].append(places[x]['cntInRadius']['1000'])
    
    # Target Variable
    target.append(round(float(line['price'])/float(data['totalArea']),2))
    
    # Building attributes
    building_floors.append(data['building'].get('floorsCount',0))
    building_cargoLiftsCount.append(data['building'].get('cargoLiftsCount',numpy.NaN))
    building_passengerLiftsCount.append(data['building'].get('passengerLiftsCount',numpy.NaN))
    building_totalArea.append(data['building'].get('totalArea',-1))
    building_materialType.append(data['building'].get('materialType','NA'))
    building_buildYear.append(2018 - data['building'].get("buildYear",2018))
    
    if "parking" in data['building']:
        parking = len(data['building']['parking']) 
    else: 
        parking = 0.0
    building_parking.append(parking)
    
    # flatType.append(data['flatType'])
    # floorNumber.append(data['floorNumber'])
    # roomsCount.append(data.get('roomsCount',-1))
        
    

# DataFrame
df = pandas.DataFrame({"address":address, "city":city, "target":target, "building_floors":building_floors, "building_material":building_materialType, 
    "building_buildYear":building_buildYear,"geo_ring_1":geo_ring_dist_1, "geo_ring_2":geo_ring_dist_2, 
    "geo_ring_3":geo_ring_dist_3,"geo_lat":geo_lat, "geo_lon":geo_lon, 
    "geo_lat_1":geo_lat_1, "geo_lon_1":geo_lon_1, "geo_lat_2":geo_lat_2, "geo_lon_2":geo_lon_2,
    "building_cargoLiftsCount":building_cargoLiftsCount, "building_passengerLiftsCount":building_passengerLiftsCount, 
    "building_parking":building_parking, "building_totalArea":building_totalArea, "building_floors":building_floors,"link":link})
for (i,x) in enumerate(places.keys()):
    df["place"+str(i)+"_100"] = places_dict["place"+str(i)+"_100"]
    df["place"+str(i)+"_500"] = places_dict["place"+str(i)+"_500"]
    df["place"+str(i)+"_1000"] = places_dict["place"+str(i)+"_1000"]

# Save Dataset
df.to_csv("re/cian_df.csv", sep=";", header=True, index=False)


# Load dataset
df = pandas.read_csv("re/cian_df.csv", sep=";")

# Aggregate Data
aggregation_dict = dict((x,"first") for x in df.columns if x not in ["target", "address", "building_totalArea"])
aggregation_dict["target"]="mean"
aggregation_dict["building_totalArea"]="mean"
df = df.groupby(by=["address"], as_index=False).agg(aggregation_dict)

# Vectorize Categorical Data
df = pandas.concat([df, pandas.get_dummies(df[['building_material']])], axis=1)
cat_features = [x for x in df.columns if x[0:18]=="building_material_" or x[0:19]=="building_buildYear_"]



# Remove cities with < 10 points
cities_series = df.city.value_counts()[df.city.value_counts() > 10]
df = df.join(cities_series, how="inner", on="city", rsuffix="huy").drop(["cityhuy"], axis=1)

df['block_id'] = df['geo_lat_1'].map(str) + df['geo_lon_1'].map(str)

# DQ (ToDO: fill missing values using coordinates average map)
df.loc[(df['building_floors']>100) | (df['building_floors']==0), 'building_floors'] = round(df['building_floors'].mean())

df.loc[(df.building_floors<=5) & (df.building_cargoLiftsCount.isnull()), 'building_cargoLiftsCount'] = 0.15
df.loc[(df.building_floors>5) & (df.building_cargoLiftsCount.isnull()), 'building_cargoLiftsCount'] = 0.93
df.loc[(df.building_floors<=5) & (df.building_passengerLiftsCount.isnull()), 'building_passengerLiftsCount'] = 0.4
df.loc[(df.building_floors>5) & (df.building_passengerLiftsCount.isnull()), 'building_passengerLiftsCount'] = 1.12

df.loc[(df['building_buildYear']==0) | (df['building_buildYear']>200), 'building_buildYear'] = round(df['building_buildYear'].mean(), 2)
df['building_buildYear'] = df.building_buildYear / 100

df.loc[df.building_totalArea < 0, 'building_totalArea'] = df.building_totalArea.mean()

# Angle to Center of Moscow
moscow_gps = {"lat":55.755826, "lon":37.617300}
moscow_gps['lat'] - df['geo_lat'];
dy = (moscow_gps['lat'] - df['geo_lat'])
dx = (10/6)*(moscow_gps['lon'] - df['geo_lon'])
df['geo_angle'] = 90 + ((dy) / (dx)).apply(math.atan) * 57.29

lat_var = "geo_lat_1"
lon_var = "geo_lon_1"
average_price = df.groupby(by=[lat_var,lon_var], as_index=False)['target'].agg({"target":["mean","count"]})
average_price.columns = ['geo_lat_1','geo_lon_1','geo_ads_mean','geo_ads_count']
df = df.merge(average_price, on=['geo_lat_1','geo_lon_1'])

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

for x in features:
  print(x)


# Prepare Training Data
X = df[features]
y = df['target']

# Model only on average price
get_error(y, X.geo_ads_mean)





# Custom quality measure
def get_error(y, y_pred):
    t = abs(y_pred - y) / y
    return (len(t[t<0.2])/len(t))

# Grid Search with CV (Gradient Boosting)ß
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





# Attrbiute Plot

# x_range = []
# y_range = []
# for i in [5,10,15,20,25,30]:
#     test['building_floors'] = i
#     x_range.append(i)
#     y_range.append(model.predict(test)[0])

# fig, ax = plt.subplots()
# ax.plot(x=x_range,y=y_range)

# plt.subplot(2, 1, 1)
# plt.plot(x_range, y_range, '-', lw=2)




# import numpy
# heatmap = numpy.zeros((heatmap_data[lat_var].nunique(),heatmap_data[lon_var].nunique()))




# min_lat = heatmap_data[lat_var].min()
# min_lon = heatmap_data[lon_var].min()
# max_lat = heatmap_data[lat_var].max()
# max_lon = heatmap_data[lon_var].max()

# for i in range(0,heatmap_data.shape[0]):
#     line = heatmap_data.loc[i]
#     # print(line)
#     heatmap[int(10*(line[lat_var]-min_lat)), int(10*(line[lon_var]-min_lon))] = line['target']

# fig, ax = plt.subplots()
# im = ax.imshow(heatmap,interpolation="none")
# ax.set_yticks(np.arange(heatmap.shape[0]))
# ax.set_xticks(np.arange(heatmap.shape[1]))
# ax.set_xticklabels(sorted(heatmap_data[lon_var].unique()))
# ax.set_yticklabels(sorted(heatmap_data[lat_var].unique()))
# ax.invert_yaxis()

# for i in range(heatmap.shape[0]):
#     for j in range(heatmap.shape[1]):
#         text = ax.text(j, i, "" if heatmap[i, j]==0.0 else str(int(round(heatmap[i, j]/1000))), ha="center", va="center", color="w")

# cbar = ax.figure.colorbar(im, ax=ax)
# cbar.ax.set_ylabel("", rotation=-90, va="bottom")
# fig.set_size_inches(10, 10)
# plt.show()



# Lookup average of block

# xx = pandas.concat([df, heatmap_data], keys=['geo_lat_1', 'geo_lon_1'])
# get_error(df.target, df.avg_price)


X = df[df.city=="Москва"][features]
y = df[df.city=="Москва"]['target']

def get_error(y, y_pred, error_rate=0.20):
    t = abs(y_pred - y) / y
    return (len(t[t<error_rate])/len(t))









# Get coordinate model params
clf = GradientBoostingRegressor(loss='ls')
parameters = {'max_depth':(2, 4, 6, 8, 10, 12), 'n_estimators':[100,250,1000]}
cv = GridSearchCV(clf, parameters, cv=5, verbose=2, scoring=make_scorer(get_error))

# coordinate_features = ['geo_lat_1','geo_lon_1']
cv.fit(X[coordinate_features],y)
cv_results = cv.cv_results_
cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
cv_table.to_csv("re/gridsearch/coordinates_boosting.csv", index=False)

rf = RandomForestRegressor()
parameters = {'max_depth':(2, 4, 6, 8, 10, 12), 'n_estimators':[100,250,1000]}
cv = GridSearchCV(rf, parameters, cv=5, verbose=2, scoring=make_scorer(get_error))
coordinate_features = ['geo_lat_1','geo_lon_1','geo_ring_1','geo_ring_2','geo_ring_3']
# coordinate_features = ['geo_lat_1','geo_lon_1']
cv.fit(X[coordinate_features],y)
cv_results = cv.cv_results_
cv_table = pandas.DataFrame({"param":cv_results['params'], "error":cv_results['mean_test_score']}).sort_values(by="error", ascending=False)
cv_table.to_csv("re/gridsearch/coordinates_randomforest.csv", index=False)

# Model coordinate price estimate
clf = GradientBoostingRegressor(loss='ls', max_depth=10, n_estimators=1000)
clf.fit(X[coordinate_features], y)
X['coord_pred'] = clf.predict(X[coordinate_features])






coordinate_features = ['geo_lat_1','geo_lon_1','geo_ring_1','geo_ring_2','geo_ring_3']
linear_features = [x for x in features if x not in coordinate_features] + ['coord_pred']
polynomial_features = linear_features + list(map(lambda x: x+"_square", linear_features)) + list(map(lambda x: x+"_sqrt", linear_features))

# Get coordinate model params
for linear_feature in linear_features:
    print(linear_feature)
    X[linear_feature+"_square"] = X[linear_feature]**2
    X[linear_feature+"_sqrt"] = X[linear_feature].apply(math.sqrt)
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

# model linear estiamte
model = lr_cv.fit(X[polynomial_features], y)
X['linear_pred'] = lr_cv.predict(X[polynomial_features])




test_samples = X.sample(10)
# modifiable_features = [x for x in features if x not in ['geo_ring_1','geo_ring_2','geo_ring_3','geo_lat_1','goe_lon_1']]
modifiable_features = [x for x in linear_features if x not in ['cood_pred']]
modifiable_integer_features = [x for x in modifiable_features if x[0:5] in "place"] + ['building_parking','building_floors','building_cargoLiftsCount','building_passengerLiftsCount'] 
modifiable_real_features = ['building_totalArea','building_buildYear']
modifiable_cat_features = ['building_material']
modifiable_numeric_features = modifiable_integer_features + modifiable_real_features

def create_valuelist(x):
    return list(range(10))

        model.predict()

import matplotlib.pyplot as plt

# Get test sample for checking plots
test_samples = X[polynomial_features].sample(10)
test_sample = test_samples.iloc[0]
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
    feature_predictions[feature] = []
    print(feature)
    for feature_value in list(range(1,10)):
        test_sample = test_samples.iloc[0].copy()
        test_sample[feature] = feature_value
        test_sample[feature+"_square"] = feature_value**2
        test_sample[feature+"_sqrt"] = math.sqrt(feature_value)
        feature_predictions[feature].append(round(model.predict(test_sample.to_frame().T)[0],2))
    plt.switch_backend('agg')
    # fig, ax = plt.subplots()
    xnew = numpy.linspace(0,10,300)
    spl = make_interp_spline(list(range(1,10)),feature_predictions[feature], k=2)
    power_smooth = spl(xnew)
    # plt.plot(feature_predictions[feature])
    # plt.savefig("re/plots/raw" + feature + ".png")
    # plt.switch_backend('agg')
    plt.plot(xnew, power_smooth)
    if feature.startswith("place"):
        plt.title(places_inverse[feature])
    plt.savefig("re/plots/" + feature + ".png")


# Plot categorical features
for feature in modifiable_cat_features:
    feature_predictions[feature] = []
    feature_values = [x[len(feature)+1:] for x in features if x.startswith(feature)]
    ind = numpy.arange(0, len(feature_values))
    for value in feature_values:
        test_sample = test_samples.iloc[0].copy()
        test_sample[[feature + "_" + x for x in feature_values]] = 0
        test_sample[feature+"_"+value] = 1
        feature_predictions[feature].append(round(model.predict(test_sample.to_frame().T)[0],2))
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    plt.barh(ind, feature_predictions[feature])
    ax.set_yticks(ind)
    ax.set_yticklabels(feature_values)
    plt.savefig("re/plots/" + feature + ".png")


