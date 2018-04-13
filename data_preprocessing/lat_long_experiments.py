import time
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from visualizations.exploratory_analysis import vis_kmean_lat_long


def lat_long_modifications(train_df):
    train_df = lat_long_bounds(train_df)
    train_df = lat_long_labeling(train_df)
    train_df = lat_long_remove_small_clusters(train_df)
    train_df = lat_long_reform_clusters(train_df)

    return train_df


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
    vis_kmean_lat_long(train_df, k_means_pick)
    import matplotlib.pyplot as plt
    plt.savefig('before_removal_kmeans.png')
    plt.show()

    return train_df


def lat_long_remove_small_clusters(train_df):
    regions = sorted(train_df.label_pick.unique())
    max_elem_count = max(train_df.label_pick.value_counts())
    for region in regions:
        if train_df[train_df['label_pick'] == region].shape[0] < 0.1 * max_elem_count:
            train_df = train_df[train_df['label_pick'] != region]

    return train_df


def lat_long_reform_clusters(train_df):
    start = time.time()
    df_pick = train_df[['pickup_longitude', 'pickup_latitude']]
    df_drop = train_df[['dropoff_longitude', 'dropoff_latitude']]
    k = 20
    init = np.array([[-73.99164461 , 40.75943564],
                    [-73.78450561  ,40.64556703],
                    [-73.99144962,    40.73762034],
                    [-73.86897935 , 40.77203725],
                    [-73.95216681,    40.7809176],
                    [-74.01021634 , 40.71170423],
                    [-73.96786457,    40.76113638],
                    [-73.98161075 , 40.74365185],
                    [-73.94496886,    40.81278371],
                    [-73.99319492 , 40.72199218],
                    [-74.00338907,    40.74307957],
                    [-73.96664798 , 40.79800915],
                    [-73.99149785,    40.74948639],
                    [-73.95879924 , 40.77089706],
                    [-73.97609084,    40.78506861],
                    [-73.97560068 , 40.75266796],
                    [-73.98331581,    40.77275706],
                    [-74.00313903 , 40.72949336],
                    [-73.98431508,    40.72914079],
                    [-73.98040424 , 40.7617903]])

    k_means_pick = KMeans(n_clusters=k,  init=init, n_init=1)
    k_means_pick.fit(df_pick)
    clust_pick = k_means_pick.labels_
    train_df['label_pick'] = clust_pick.tolist()
    train_df['label_drop'] = k_means_pick.predict(df_drop)

    end = time.time()

    print("Time taken in kmeans recompute computation is {}.".format(end - start))
    vis_kmean_lat_long(train_df, k_means_pick)
    import matplotlib.pyplot as plt
    plt.savefig('after_removal_kmeans.png')
    plt.show()

    return train_df
