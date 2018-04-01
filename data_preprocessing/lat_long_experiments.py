import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def lat_long_bounds(train_df):
    start = time.time()
    xlim = [-74.03, -73.77]
    ylim = [40.63, 40.85]
    # xlim = [-74.25, -73.77]
    # ylim = [40.55, 40.95]
    train_df = train_df[(train_df.pickup_longitude > xlim[0]) & (train_df.pickup_longitude < xlim[1])]
    train_df = train_df[(train_df.dropoff_longitude > xlim[0]) & (train_df.dropoff_longitude < xlim[1])]
    train_df = train_df[(train_df.pickup_latitude > ylim[0]) & (train_df.pickup_latitude < ylim[1])]
    train_df = train_df[(train_df.dropoff_latitude > ylim[0]) & (train_df.dropoff_latitude < ylim[1])]
    end = time.time()
    print("Time taken in lat_long_bounds is {}.".format(end - start))
    return train_df


def lat_long_labeling(train_df):
    start = time.time()
    df_pick = train_df[['pickup_longitude', 'pickup_latitude']]
    df_drop = train_df[['dropoff_longitude', 'dropoff_latitude']]

    init = np.array([[-73.98737616, 40.72981533],
                     [-121.93328857, 37.38933945],
                     [-73.78423222, 40.64711269],
                     [-73.9546417, 40.77377538],
                     [-66.84140269, 36.64537175],
                     [-73.87040541, 40.77016484],
                     [-73.97316185, 40.75814346],
                     [-73.98861094, 40.7527791],
                     [-72.80966949, 51.88108444],
                     [-76.99779701, 38.47370625],
                     [-73.96975298, 40.69089596],
                     [-74.00816622, 40.71414939],
                     [-66.97216034, 44.37194443],
                     [-61.33552933, 37.85105133],
                     [-73.98001393, 40.7783577],
                     [-72.00626526, 43.20296402],
                     [-73.07618713, 35.03469086],
                     [-73.95759366, 40.80316361],
                     [-79.20167796, 41.04752096],
                     [-74.00106031, 40.73867723]])

    k = 20
    k_means_pick = KMeans(n_clusters=k, init=init, n_init=1)
    k_means_pick.fit(df_pick)
    clust_pick = k_means_pick.labels_
    train_df['label_pick'] = clust_pick.tolist()
    train_df['label_drop'] = k_means_pick.predict(df_drop)

    end = time.time()
    print("Time taken in k_means_computation is {}.".format(end - start))

    return train_df
