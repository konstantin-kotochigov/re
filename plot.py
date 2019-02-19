nonzero_features = sorted(list(set([model_features[n].replace("_square","").replace("_sqrt","") for n,x in enumerate(model.coef_) if x!=0.0])))

# Sort features by coefficient absolute value
from functools import reduce

# get features with nonzero coefficients
nonzero_features_with_coef = [(model_features[n].replace("_square","").replace("_sqrt",""),x) for n,x in enumerate(model.coef_) if x!=0.0]

# sort by coefficient descending
nonzero_features_sorted = [x[0] for x in sorted(nonzero_features_with_coef, key=lambda x: abs(x[1]), reverse=True)]

# remove duplicates preserving order
nonzero_features_sorted_unique = reduce(lambda l, x: l.append(x) or l if x not in l else l, nonzero_features_sorted, [])
feature_rank = {x:n for n,x in enumerate(nonzero_features_sorted_unique)}


# test_samples = X.sample(10)
# modifiable_features = [x for x in features if x not in ['geo_ring_1','geo_ring_2','geo_ring_3','geo_lat_1','goe_lon_1']]
modifiable_features = [x for x in nonzero_features if x not in ['coord_pred','geo_ads_count','geo_ads_mean', 'geo_ring']]
# modifiable_integer_features = [x for x in modifiable_features if x[0:5] in "place"] + ['building_parking','building_cargoLiftsCount','building_passengerLiftsCount']
# modifiable_numeric_features = modifiable_integer_features + modifiable_real_features

import matplotlib.pyplot as plt

# Get test sample for checking plots
# test_samples = X[polynomial_features].sample(10)
# test_samples = X[X.address=="Россия, Санкт-Петербург , Октябрьская набережная, д 80 к 1"]
# test_samples = test_samples[polynomial_features]

# Create two lists to gather predictions for an attribute range

import matplotlib
matplotlib.use('Agg')
plt.switch_backend("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

# Dict for setting limits
features_dict = dict()
for feature in modifiable_features:
    features_dict[feature] = dict()
    # features_dict[feature]['min'] = min(X[feature])
    # features_dict[feature]['max'] = max(X[feature])
    # features_dict[feature]['mean'] = round(X[feature].mean())
    num_points =  min(len(X[feature].value_counts()), 20)
    features_dict[feature]['range'] = numpy.linspace(min(X[feature]), max(X[feature]), num_points)

        lr_predictions = dict()
        # rf_predictions = dict()
        result_json = []

        # Plot integer features
        for feature in modifiable_features:
            print(feature)
            lr_predictions[feature] = []
            feature_range = features_dict[feature]['range']
            if len(feature_range) == 1:
                continue
            for feature_value in feature_range:
                test_sample = X_test.copy()
                test_sample[feature] = feature_value
                test_sample[feature+"_square"] = feature_value**2
                test_sample[feature+"_sqrt"] = math.sqrt(feature_value)
                lr_predictions[feature].append(round(model.predict(test_sample[model_features])[0],2))
            plt.clf()
            plt.plot(feature_range, lr_predictions[feature], 'o-')
            if feature.startswith("place"):
                plt.title(places_inverse[feature.replace("_bin_2","").replace("_bin_1","")] + "(" + feature + ")")
            if len(feature_range) < 10:
                 plt.xticks(feature_range)
            plt.savefig("re/plots/" + str(feature_rank.get(feature,""))+"_"+ feature + ".png", dpi=300)
            result_json.append({"feature_id":feature,"feature_name":places_inverse.get(feature,""), "feature_rank":feature_rank[feature], "x":list(feature_range), "y":list(lr_predictions[feature])})


# Plot categorical features
for feature in modifiable_cat_features:
    lr_predictions[feature] = []
    feature_values = [x[len(feature)+1:] for x in features if x.startswith(feature)]
    ind = numpy.arange(0, len(feature_values))
    for value in feature_values:
        test_sample = X_test.copy()
        test_sample[[feature + "_" + x for x in feature_values]] = 0
        test_sample[feature+"_"+value] = 1
        lr_predictions[feature].append(round(lr.predict(test_sample.to_frame().T[polynomial_features])[0],2))
    plt.switch_backend('agg')
    fig, ax = plt.subplots()
    plt.barh(ind, lr_predictions[feature])
    ax.set_yticks(ind)
    ax.set_yticklabels(feature_values)
    plt.savefig("re/plots/" + str(min([feature_rank.get(feature+"_"+x, 1000) for x in feature_values]))+"_" + feature + ".png")

import json
outfile = open('re/data.json', 'w')
json.dump(result_json, outfile)
outfile.close()



features_rank_df = pandas.DataFrame.from_dict(feature_rank, orient="index",columns=["rank"])
features_rank_df['feature'] = features_rank_df.index
features_rank_df['feature_name'] = features_rank_df.feature.map(places_inverse)
features_rank_df.sort_values(by="rank", inplace=True)

features_rank_df.to_csv("re/plots/feature_list.csv", sep=";", index=False)









from sqlalchemy import create_engine
engine = create_engine('postgresql://cian:5Dsy4LcXQVPSfKeRzEB@46.183.221.135:5432/cian')

x = pandas.read_sql_query('select * from public.cian ',con=engine)



cian_data = json.load(open("re/cian_piter.json"))
res = []
for line in cian_data:
    places = line['data']['places']
    res.append(places['Ипподром']['cntInRadius']['1000'])
