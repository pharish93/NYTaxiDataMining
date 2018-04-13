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
    lrt_cols = ['total_distance', 'total_travel_time', 'label_pick', 'label_drop', 'distance_haversine',
                'trip_duration', 'passenger_count']
    train_data = train_data[lrt_cols]

    return train_data

def lrt_temporal_pre_process(train_data):
    lrt_cols = ['total_distance', 'total_travel_time', 'day_of_year', 'distance_haversine',
                'trip_duration', 'passenger_count']
    train_data = train_data[lrt_cols]

    return train_data

def lrt_taxi_data(train_data, type = "spatial"):

    segments = sorted(train_data.label_pick.unique())
    segment_label = 'label_pick'
    if type == "spatial":
        train_data = lrt_spatial_pre_process(train_data)

    else:
        train_data = lrt_temporal_pre_process(train_data)
        segments = sorted(train_data.day_of_year.unique())
        segment_label = 'day_of_year'


    p_values_all_regions = dict()
    lrt_values = dict()


    y, X = patsy.dmatrices('trip_duration ~ total_distance + total_travel_time + distance_haversine', data=train_data,
                           return_type='dataframe')

    y = y.values
    X = X.values

    global_model = sm.GLM(y, X, family=sm.families.Gaussian())
    global_results = global_model.fit()
    global_ll = global_results.llf/len(y)

    for segment in segments:
        # region_indices = (train_data['label_pick'] == region) | (train_data['label_drop'] == region)
        # outside_indices = (train_data['label_pick'] != region) & (train_data['label_drop'] != region)

        segment_indices = train_data[segment_label] == segment
        outside_indices = train_data[segment_label] != segment

        y_R, X_R = patsy.dmatrices('trip_duration ~ total_distance + total_travel_time + distance_haversine',
                               data=train_data.loc[segment_indices, :], return_type='dataframe')
        y_R = y_R.values
        X_R = X_R.values
        y_O, X_O = patsy.dmatrices('trip_duration ~ total_distance + total_travel_time + distance_haversine',
                               data=train_data.loc[outside_indices, :], return_type='dataframe')
        y_O = y_O.values
        X_O = X_O.values

        region_model = sm.GLM(y_R, X_R, family=sm.families.Gaussian())
        outside_model = sm.GLM(y_O, X_O, family=sm.families.Gaussian())

        region_results = region_model.fit()
        outside_results = outside_model.fit()



        region_ll = region_results.llf/sum(segment_indices)
        outside_ll = outside_results.llf/sum(outside_indices)

        region_lrt = likelihood_ratio(global_ll, region_ll, outside_ll)

        lrt_values[segment] = [region_lrt, global_ll, region_ll, outside_ll]
        df = 7
        p = 1 - chi2.cdf(region_lrt, df)
        p_values_all_regions[segment] = p

    lrt_sorted = sorted(lrt_values.items(), key=operator.itemgetter(1))
    for element in lrt_sorted:
        print (element[0], '-->', element[1])
    return p_values_all_regions
