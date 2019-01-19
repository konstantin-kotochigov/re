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
    "geo_lat_1":geo_lat_1, "geo_lon_1":geo_lon_1, "geo_lat_2":geo_lat_2, "geo_lon_2":geo_lon_2,
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

counts = {}
for x in list(places_inverse.keys()):
    counts[x] = sum(df[x] > 0)
counts_df = pandas.DataFrame.from_dict(counts, orient="index", columns=["cnt"])
counts_df['feature'] = counts_df.index.map(places_inverse)
counts_df['feature'] = counts_df.feature + counts_df.index.str.replace("place[0-9]*","")
counts_df = counts_df.sort_values(by="feature")
counts_df['total'] = df.shape[0]
counts_df = counts_df.reset_index()[['feature','cnt','total']]

counts_df.to_csv("re/feature_density.csv", sep=";", index=False)

df.to_csv("re/piter_df.csv", sep=";", header=True, index=False)