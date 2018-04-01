import pandas as pd
import os
import time
import cPickle

def load_train_data():

    s = time.time()
    train_cache_file = './data/cache/train_df.pkl'
    if os.path.exists(train_cache_file):
        with open(train_cache_file, 'rb') as fid:
            train_df = cPickle.load(fid)
    else:
        train_fr_1 = pd.read_csv('./data/fastest_routes_train_part_1.csv')
        train_fr_2 = pd.read_csv('./data/fastest_routes_train_part_2.csv')
        train_fr = pd.concat([train_fr_1, train_fr_2])

        train_fr_new = train_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
        train_df = pd.read_csv('./data/train.csv')
        train = pd.merge(train_df, train_fr_new, on='id', how='left')
        train_df = train.copy()

        with open(train_cache_file, 'wb') as fid:
            cPickle.dump(train_df, fid, cPickle.HIGHEST_PROTOCOL)

    end = time.time()
    print("Time taken to load data is {}.".format((end - s)))
    train_df.head()

    return train_df
