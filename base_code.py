from __future__ import print_function
from data_loading.data_loading import *
from data_preprocessing.data_pre_process import *
from visualizations.exploratory_analysis import *

def main():

    print('New York Taxi Data Mining')
    train_df = load_train_data()
    check_unique_id(train_df)
    vis_trip_duration(train_df)
    print('End of Project')

if __name__ == '__main__':
    main()
