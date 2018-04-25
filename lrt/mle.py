import numpy as np
from scipy.stats import chi2

from matplotlib import pyplot as plt

from scipy import stats
import statsmodels.api as sm

import pandas as pd
import patsy
import operator

def likelihood_ratio(global_ll, region_ll, outside_ll):
    return(-2.0*(region_ll + outside_ll - global_ll))

def lrt_spatial_pre_process(train_data):
    lrt_cols = ['total_distance', 'total_travel_time', 'label_pick', 'distance_haversine',
                'trip_duration', 'passenger_count']
    train_data = train_data[lrt_cols]

    return train_data

def lrt_temporal_pre_process(train_data):
    lrt_cols = ['total_distance', 'total_travel_time', 'day_of_year', 'distance_haversine',
                'trip_duration', 'passenger_count']
    train_data = train_data[lrt_cols]

    return train_data

def lrt_spatial_temporal_pre_process(train_data):
    lrt_cols = ['total_distance', 'total_travel_time', 'label_pick', 'distance_haversine',
                'trip_duration', 'passenger_count', 'day_of_year']
    train_data = train_data[lrt_cols]
    train_data['day_of_year'] = train_data['day_of_year'].astype(str)
    train_data['label_pick'] = train_data['label_pick'].astype(str)
    train_data['label_pick_day_of_year'] = train_data[['label_pick', 'day_of_year']].apply(lambda x: ''.join(x), axis=1)

    return train_data

def lrt_taxi_data(train_data, type = "spatial"):

    # Grab segments in the default case of spatial analysis
    segments = sorted(train_data.label_pick.unique())
    segment_label = 'label_pick'
    if type == "spatial":
        train_data = lrt_spatial_pre_process(train_data)

    elif type == "temporal":
        train_data = lrt_temporal_pre_process(train_data)
        # Else grab segments in the case of temporal analysis
        segments = sorted(train_data.day_of_year.unique())
        segment_label = 'day_of_year'

    else:
        # Prototype for spatial temporal analysis
        train_data = lrt_spatial_temporal_pre_process(train_data)
        segments = sorted(train_data.label_pick_day_of_year.unique())
        segment_label = 'label_pick_day_of_year'

    p_values_all_regions = dict()
    lrt_values = dict()

    # Use patsy to retrieve data frames specific for what we are trying to model (similar to R formula)
    y, X = patsy.dmatrices('trip_duration ~ total_distance + total_travel_time + distance_haversine', data=train_data,
                           return_type='dataframe')

    # Convert data frames to numpy arrays
    y = y.values
    X = X.values

    # Use stats models GLM to model null hypothesis and get average log-likelihood
    global_model = sm.GLM(y, X, family=sm.families.Gaussian())
    global_results = global_model.fit()
    global_ll = global_results.llf/len(y)

    for segment in segments:

        # Grab indices for s and s not
        segment_indices = train_data[segment_label] == segment
        outside_indices = train_data[segment_label] != segment

        # Get data frames for two data subsets for alternative and convert to numpy array
        y_R, X_R = patsy.dmatrices('trip_duration ~ total_distance + total_travel_time + distance_haversine',
                               data=train_data.loc[segment_indices, :], return_type='dataframe')

        y_R = y_R.values
        X_R = X_R.values
        y_O, X_O = patsy.dmatrices('trip_duration ~ total_distance + total_travel_time + distance_haversine',
                               data=train_data.loc[outside_indices, :], return_type='dataframe')
        y_O = y_O.values
        X_O = X_O.values

        # Model both using stats models GLM and get average log-likelihoods (can replace Gaussian with Gamma/log link)
        region_model = sm.GLM(y_R, X_R, family=sm.families.Gaussian())
        outside_model = sm.GLM(y_O, X_O, family=sm.families.Gaussian())

        region_results = region_model.fit()
        outside_results = outside_model.fit()

        region_ll = region_results.llf/sum(segment_indices)
        outside_ll = outside_results.llf/sum(outside_indices)

        # Calculate likelihood ratio
        region_lrt = likelihood_ratio(global_ll, region_ll, outside_ll)

        lrt_values[segment] = [region_lrt, global_ll, region_ll, outside_ll]
        # df = 3
        # p = 1 - chi2.cdf(region_lrt, df)
        # p_values_all_regions[segment] = p
        # print(segment, 'num of samples:', sum(segment_indices))

    lrt_sorted = sorted(lrt_values.items(), key=operator.itemgetter(1),reverse=True)
    iter = 0
    print("The top 3 anomalous points are:")
    for element in lrt_sorted:
        #print (element[0], '-->', element[1])
        print(element[0])
        iter+=1
        if(iter>=3):
            break
    plt.scatter(segments, [i[0] for i in lrt_values.values()])
    plt.xticks(range(0, max(segments),5),rotation = 'vertical')
    #plt.xticks(np.arange(0, max(segments)))
    plt.xlabel('Cluster Labels')
    plt.ylabel('Likelihood Ratio')
    plt.title('Likelihood Ratio of Data points')
    plt.show()
    return p_values_all_regions


