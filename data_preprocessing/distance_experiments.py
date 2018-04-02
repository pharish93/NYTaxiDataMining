import numpy as np
import pandas as pd
import time

def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def add_distance_measures(train_df):

    start = time.time()
    train_df.loc[:, 'distance_haversine'] = haversine_array(train_df['pickup_latitude'].values,
                                                            train_df['pickup_longitude'].values,
                                                            train_df['dropoff_latitude'].values,
                                                            train_df['dropoff_longitude'].values)
    train_df.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train_df['pickup_latitude'].values,
                                                                           train_df['pickup_longitude'].values,
                                                                           train_df['dropoff_latitude'].values,
                                                                           train_df['dropoff_longitude'].values)
    train_df.loc[:, 'direction'] = bearing_array(train_df['pickup_latitude'].values,
                                                 train_df['pickup_longitude'].values,
                                                 train_df['dropoff_latitude'].values,
                                                 train_df['dropoff_longitude'].values)

    end = time.time()
    print("Time taken to add distance values is {}.".format(end - start))

    return train_df
