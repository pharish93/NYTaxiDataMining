from __future__ import print_function
from data_loading.data_loading import *
from data_preprocessing.data_pre_process import data_preprocessing
from visualizations.exploratory_analysis import visualize_data
from data_preprocessing.sub_sample_data import model1_sub_sample_data
from lrt.mle import *

def spatial_anomalies(train_df):
    train_df = model1_sub_sample_data(train_df)
    p_values = lrt_taxi_data(train_df)
    print(p_values)


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

    print('End of Project')

if __name__ == '__main__':
    main()
