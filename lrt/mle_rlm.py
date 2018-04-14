import numpy as np
from scipy.stats import chi2

from matplotlib import pyplot as plt
import statsmodels.api as sm
from scipy import stats

import pandas as pd
import patsy
import operator

def likelihood_ratio(global_ll, region_ll, outside_ll):
    return(-2.0*(region_ll + outside_ll - global_ll))

def lrt_pre_process(train_data):
    lrt_cols = ['total_distance', 'total_travel_time', 'label_pick', 'label_drop', 'distance_haversine',
                'trip_duration']
    train_data = train_data[lrt_cols]

    return train_data

def lrt_taxi_data(train_data):
    train_data = lrt_pre_process(train_data)

    train_data = train_data.sample(n=5000, replace=False)

    regions = sorted(train_data.label_pick.unique())
    p_values_all_regions = dict()
    lrt_values = dict()


    y, X = patsy.dmatrices('trip_duration ~ total_distance + total_travel_time + distance_haversine', data=train_data,
                           return_type='dataframe')

    global_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
    global_results = global_model.fit()
    print(global_results)
    exit()
    global_ll = global_results.llf()


    for region in regions:
        #region_indices = (train_data['label_pick'] == region) | (train_data['label_drop'] == region)
        #outside_indices = (train_data['label_pick'] != region) & (train_data['label_drop'] != region)

        region_indices = train_data['label_pick'] == region
        outside_indices = train_data['label_pick'] != region
        print(region, sum(region_indices), sum(outside_indices))

        y_R, X_R = patsy.dmatrices('trip_duration ~ total_distance + total_travel_time + distance_haversine',
                               data=train_data.loc[region_indices, :], return_type='dataframe')
        y_R = y_R.values
        X_R = X_R.values
        y_O, X_O = patsy.dmatrices('trip_duration ~ total_distance + total_travel_time + distance_haversine',
                               data=train_data.loc[outside_indices, :], return_type='dataframe')
        y_O = y_O.values
        X_O = X_O.values

        region_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
        outside_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())

        region_results = region_model.fit()
        outside_results = outside_model.fit()

        region_ll = region_results.llf()/sum(region_indices)
        outside_ll = outside_results.llf()/sum(outside_indices)

        region_lrt = likelihood_ratio(global_ll, region_ll, outside_ll)

        #print(region, region_lrt)
        lrt_values[region] = [region_lrt, global_ll, region_ll, outside_ll]
        # print(region_lrt)
        df = 7
        p = 1 - chi2.cdf(region_lrt, df)
        p_values_all_regions[region] = p

    lrt_sorted = sorted(lrt_values.items(), key=operator.itemgetter(1))
    for element in lrt_sorted:
        print (element[0], '-->', element[1])
    return p_values_all_regions
