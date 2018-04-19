from __future__ import print_function
from data_loading.data_loading import *
from data_preprocessing.data_pre_process import data_preprocessing
from visualizations.exploratory_analysis import visualize_data
from data_preprocessing.sub_sample_data import *
from lrt.mle import *


# Model 1
def spatial_anomalies(train_df):
    start = time.time()
    print('Model 1 : Spatial Anomalies modeling start')
    train_df = model1_sub_sample_data(train_df)
    lrt_values = lrt_taxi_data(train_df, type='spatial')
    print(lrt_values)
    end = time.time()
    print("Time taken in Spatial Anomalies modeling is {}.".format(end - start))
    print('Model 1 : Spatial Anomalies modeling end')


# Model 2
def temporal_anomalies(train_df):
    start = time.time()
    print('Model 2 : Temporal Anomalies modeling start')

    train_df = model2_sub_sample_data(train_df)
    lrt_values = lrt_taxi_data(train_df,type = 'temporal')
    print(lrt_values)
    end = time.time()
    print("Time taken in Temporal Anomalies modeling is {}.".format(end - start))
    print('Model 2 : Temporal Anomalies modeling end')

def spatial_temporal_anomalies(train_df):
    start = time.time()
    print('Model 3 : Spatial Temporal Anomalies modeling start')

    train_df = model2_sub_sample_data(train_df)
    lrt_values = lrt_taxi_data(train_df, type='spatial_temporal')
    print(lrt_values)
    end = time.time()
    print("Time taken in Spatial Temporal Anomalies modeling is {}.".format(end - start))
    print('Model 3 : Spatial Temporal Anomalies modeling end')

def main():
    print('New York Taxi Data Mining')
    s = time.time()
    train_cache_file = './data/cache/pre_processed_train_df.pkl'
    if os.path.exists(train_cache_file):
        with open(train_cache_file, 'rb') as fid:
            train_df = cPickle.load(fid)
    else:
        train_df = load_train_data()
        train_df = data_preprocessing(train_df)

        with open(train_cache_file, 'wb') as fid:
            cPickle.dump(train_df, fid, cPickle.HIGHEST_PROTOCOL)

    # visualize_data(train_df)
    spatial_anomalies(train_df)
    temporal_anomalies(train_df)
    # spatial_temporal_anomalies(train_df)


    print('End of Project')


if __name__ == '__main__':
    main()
