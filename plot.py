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

# Create two lists to gather predictions for an attribute range
lr_predictions = dict()
rf_predictions = {}

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
    lr_predictions[feature] = []
    rf_predictions[feature] = []
    feature_range = range(int(features_dict[feature]['min']),int(features_dict[feature]['max'] + 1))
    if len(feature_range) == 1:
        continue
    for feature_value in list(feature_range):
        test_sample = X_test.copy()
        test_sample[feature] = feature_value
        test_sample[feature+"_square"] = feature_value**2
        test_sample[feature+"_sqrt"] = math.sqrt(feature_value)
        lr_predictions[feature].append(round(lr.predict(test_sample.to_frame().T[polynomial_features])[0],2))
        rf_predictions[feature].append(round(rf.predict(test_sample.to_frame().T[features])[0],2))
    # plt.switch_backend('agg')
    plt.clf()
    # fig, ax = plt.subplots()
    xnew = numpy.linspace(min(list(feature_range)), max(list(feature_range))+1, len(feature_range)+1)
    if len(feature_range) <= 2: 
        power = 1
    else:
        power = 2
    # spl = make_interp_spline(list(feature_range),lr_predictions[feature], k=power)
    # power_smooth = spl(xnew)
    # plt.plot(feature_predictions[feature])
    # plt.savefig("re/plots/raw" + feature + ".png")
    # plt.switch_backend('agg')
    
    # plt.plot(xnew, power_smooth)
    plt.plot(feature_range, lr_predictions[feature], 'o-', feature_range, rf_predictions[feature], 'r--')
    
    if feature.startswith("place"):
        plt.title(places_inverse[feature] + "(" + feature + ")")
    if len(feature_range) < 10:
         plt.xticks(feature_range)
    plt.savefig("re/plots/" + str(feature_rank[feature])+"_"+ feature + ".png", dpi=300)


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