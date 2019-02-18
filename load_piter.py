import json
import pandas
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression,  LassoLarsCV, ElasticNet
from sklearn.metrics import mean_squared_error
from pandas.io.json import json_normalize
from sklearn.metrics import make_scorer
from sklearn.neighbors import KernelDensity
import math
import numpy
from tqdm import tqdm

# Compute NULL statistics and do some EDA
assess_quality = False

# Extract from Postgres
from sqlalchemy import create_engine
engine = create_engine('postgresql://cian:5Dsy4LcXQVPSfKeRzEB@46.183.221.135:5432/cian')
x = pandas.read_sql_query('select * from public.cian',con=engine)
x.to_json("re/cian_piter.json", orient="records")

# Load Data
cian_data = json.load(open("re/cian_piter.json"))

# Read Initial Data
line = cian_data[0]
data = line['data']
places = data['places']

# Dictionary to collect data while reading JSON
places_dict = dict()

# A map from feature index to name
places_inverse = dict()

# Fill in the maps
for (i,x) in enumerate(places.keys()):
    places_dict["place"+str(i)+"_100"] = []
    places_dict["place"+str(i)+"_500"] = []
    places_dict["place"+str(i)+"_1000"] = []
    places_dict["place"+str(i)+"_nearest"] = []
    places_inverse["place"+str(i)+"_100"] = x
    places_inverse["place"+str(i)+"_500"] = x
    places_inverse["place"+str(i)+"_1000"] = x
    places_inverse["place"+str(i)+"_nearest"] = x





places_inverse_df = pandas.DataFrame.from_dict(places_inverse, orient="index", columns=["feature_name"])
places_inverse_df.to_csv("re/feature_encoding.csv", sep=";", index=True)

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
geo_ring_dist = []

geo_underground_dist = []
geo_underground_new = []

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
    geo_ring_dist.append(rings['КАД'])
    underground_dist = None
    underground_new = None
    if 'undergrounds' in line['data']['geo']:
        if len(line['data']['geo']['undergrounds']) > 0:
            if 'time' in line['data']['geo']['undergrounds'][0]:
                underground_dist = line['data']['geo']['undergrounds'][0]['time']
            if 'underConstruction' in line['data']['geo']['undergrounds'][0]:
                underground_new = line['data']['geo']['undergrounds'][0]['underConstruction']
    geo_underground_dist.append(underground_dist)
    geo_underground_new.append(underground_new)
    # Geometrics
    for (i,x) in enumerate(places.keys()):
        places_dict["place"+str(i)+"_100"].append(places[x]['cntInRadius']['100'])
        places_dict["place"+str(i)+"_500"].append(places[x]['cntInRadius']['500'])
        places_dict["place"+str(i)+"_1000"].append(places[x]['cntInRadius']['1000'])
        if places[x]['minDistance'] != []:
            places_dict["place"+str(i)+"_nearest"].append(places[x]['minDistance']['value'])
        else:
            places_dict["place"+str(i)+"_nearest"].append(None)
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
    "building_buildYear":building_buildYear,"geo_ring":geo_ring_dist, "geo_lat":geo_lat, "geo_lon":geo_lon,
    "geo_lat_1":geo_lat_1, "geo_lon_1":geo_lon_1, "geo_lat_2":geo_lat_2, "geo_lon_2":geo_lon_2, "geo_underground_dist":geo_underground_dist, "geo_underground_new":geo_underground_new,
    "building_cargoLiftsCount":building_cargoLiftsCount, "building_passengerLiftsCount":building_passengerLiftsCount,
    "building_parking":building_parking, "building_totalArea":building_totalArea, "building_floors":building_floors,"link":link})
for (i,x) in enumerate(places.keys()):
    df["place"+str(i)+"_100"] = places_dict["place"+str(i)+"_100"]
    df["place"+str(i)+"_500"] = places_dict["place"+str(i)+"_500"]
    df["place"+str(i)+"_1000"] = places_dict["place"+str(i)+"_1000"]
    df["place"+str(i)+"_nearest"] = places_dict["place"+str(i)+"_nearest"]

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

# df['geo_underground_dist'] = round((df.geo_underground_dist - df.geo_underground_dist.mean()) / df.geo_underground_dist.std(), 2)
df.geo_underground_dist.fillna(df.geo_underground_dist.mean(), inplace=True)
df.geo_underground_new.fillna(False, inplace=True)

# Angle to Center of Moscow
# moscow_gps = {"lat":55.755826, "lon":37.617300}
# moscow_gps['lat'] - df['geo_lat'];
# dy = (moscow_gps['lat'] - df['geo_lat'])
# dx = (10/6)*(moscow_gps['lon'] - df['geo_lon'])
# df['geo_angle'] = 90 + ((dy) / (dx)).apply(math.atan) * 57.29

lat_var = "geo_lat_1"
lon_var = "geo_lon_1"
average_price = df.groupby(by=[lat_var,lon_var], as_index=False)['target'].agg({"target":["mean","count"]})
average_price.columns = ['geo_lat_1','geo_lon_1','geo_ads_mean','geo_ads_count']
df = df.merge(average_price, on=['geo_lat_1','geo_lon_1'])

df['geo_ring'] = round(df['geo_ring'], -3)

# Check NULLs
df[list(places_inverse.keys())].isna().sum(axis=0).sort_index().reset_index().to_csv("re/nulls.csv", sep=";", index=False)

nearest_features = [x for x in df.columns if x.endswith("_nearest")]
for x in nearest_features:
    df[x] = df[x].fillna(value=10000)





df.to_csv("re/piter_df.csv", sep=";", header=True, index=False)



# Load Test Point

# Generate point of interest
# X_test = df[df.address=="Россия, Санкт-Петербург , Октябрьская набережная, д 80 к 1"]
# X_test1f. = df[df.address=="Санкт-Петербург, Невский район, Октябрьская наб. 98 к4"].iloc[0]
X_test = df.iloc[0].copy()

# Загрузить тестовую строчку для Питера
test_places_file = open("re/target.json")
test_places = json.loads(test_places_file.read())
test_places_file.close()

places_converter = {v: k[0:k.index("_")] for k, v in places_inverse.items()}

if (assess_quality):
    counts = {}
    for x in list(places_inverse.keys()):
        counts[x] = sum(df[x] > 0)
    counts_df = pandas.DataFrame.from_dict(counts, orient="index", columns=["cnt"])
    counts_df['feature'] = counts_df.index.map(places_inverse)
    counts_df['feature'] = counts_df.feature + counts_df.index.str.replace("place[0-9]*", "")
    counts_df = counts_df.sort_values(by="feature")
    counts_df['total'] = df.shape[0]
    counts_df = counts_df.reset_index()[['feature', 'cnt', 'total']]
    counts_df.to_csv("re/feature_density.csv", sep=";", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    for x in tqdm(nearest_features, total=len(nearest_features), unit="features"):
        kde = KernelDensity(kernel="gaussian", bandwidth=100)
        dummy = kde.fit(df[x][:, numpy.newaxis])
        X_plot = numpy.linspace(0, 10000, 250)[:, numpy.newaxis]
        d = numpy.exp(kde.score_samples(X_plot))
        dummy = plt.clf()
        dummy = plt.xlabel("km")
        dummy = plt.xlim(0, 2000)
        dummy = plt.ylabel("Density")
        dummy = plt.title(places_inverse[x])
        dummy = plt.axvline(x=df[x][df[x] < 10000].mean(), color='red', alpha=0.25, linestyle='--')
        # plt.xticks([])
        dummy = plt.plot(X_plot, d, 'o-')
        # plt.hist(y_pred, bins=20)
        dummy = plt.savefig("re/plots/nearest_densities/" + x + "_density.png", dpi=300)

places_attributes = [x for x in df.columns if x.startswith("place") and not x.endswith("_nearest")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.switch_backend("agg")
for place_attr in tqdm(places_attributes, total=len(places_attributes), unit="features"):
    y = []
    y_err = []
    y_color = []
    x_cnt = []
    x_width = []
    x_name = []
    plt.clf()
    from_tick = df[place_attr].min()
    to_tick = df[place_attr].max()
    if to_tick - from_tick > 10:
        num_ticks = 10
    else:
        num_ticks = to_tick - from_tick
    # print(place_attr, " ", num_ticks)
    x = list(numpy.linspace(from_tick, to_tick, num_ticks+1))
    x.append(max(x)+1)
    for tick_num in range(1,len(x)):
        i = (df[place_attr] < x[tick_num]) & (df[place_attr] >= x[tick_num - 1])
        val = (df.target[i]-df.geo_ads_mean[i])
        y.append(val.mean())
        y_err.append(val.std())
        x_cnt.append(sum(i))
        x_name.append(str(math.ceil(x[tick_num-1])) + "-" + str(math.floor(x[tick_num])))
    y_color = ["blue" if x >0 else "red" for x in y]
    dummy = plt.bar(x=x[0:-1], height=y, width=max(0.75, (to_tick-from_tick)/11), color=y_color, alpha =0.75, yerr=y_err, edgecolor="black", capsize=10, ecolor='black', error_kw={"alpha":0.25})
    # plt.errorbar(elinewidth=1)
    # plt.text(1, 100000, "qwerty")
    for i, v in enumerate(y):
        dummy = plt.text(x[i], v + y_err[i] + 1000, x_cnt[i], color='blue',  alpha=0.5, fontsize=10, horizontalalignment="center")
    # if to_tick - from_tick <= 10:
    #     dummy = plt.set_xticklabels(x)
    # else:
    #     dummy = plt.set_xticklabels(x_name)
    dummy = plt.title(places_inverse[place_attr]+" ("+place_attr+")")
    dummy = plt.savefig("re/plots/attributes/"+place_attr+".png", dpi=300)

    # Check dependencies for School atribute
    # test = X.groupby("place15_500", as_index=False)['target'].agg({"mean","count","std"})
    # import matplotlib
    # matplotlib.use('agg')
    # plt.switch_backend('agg')
    # import matplotlib.pyplot as plt
    # plt.bar(list(test.index), test["mean"].values, yerr=test['std'].values)
    # plt.savefig("re/plots/school_dict_500.png", dpi=300)



test_string = {}
for place_attribute, place_data in test_places.items():
    # print(place_attribute, place_data)
    test_string[places_converter[place_attribute]+"_100"] = place_data['cntInRadius']['100']
    test_string[places_converter[place_attribute]+"_500"] = place_data['cntInRadius']['500']
    test_string[places_converter[place_attribute]+"_1000"] = place_data['cntInRadius']['1000']
    if len(place_data['minDistance'])==0:
        test_string[places_converter[place_attribute]+"_nearest"] = 10000
    else:
        test_string[places_converter[place_attribute] + "_nearest"] = place_data['minDistance']['value']


# for k,v in test_string.items():
#     X_test.loc[k] = v

X_test['building_buildYear'] = 0.0 # New
X_test['building_totalArea'] = 50 # Как считать?
X_test['geo_ring'] = 2000
X_test['geo_lat'] = 59.872067
X_test['geo_lon'] = 30.474787
X_test['geo_lat_1'] = 59.9
X_test['geo_lon_1'] = 30.5
X_test['building_parking'] = 1
X_test['building_material_monolith'] = 1
X_test['building_material_brick'] = 0
X_test['geo_ads_mean'] = 102000
X_test['geo_ads_count'] = 903
X_test['building_passengerLiftsCount'] = 1
X_test['building_floors'] = 10
X_test['building_cargoLiftsCount'] = 0
X_test['block_id'] = "59.930.5"
X_test['geo_underground_dist'] = 5

X_test.to_frame().T.to_csv("re/test.csv", sep=";", index=False, header=True)