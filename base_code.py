from __future__ import print_function
from data_loading.data_loading import *
from data_preprocessing.data_pre_process import *
from visualizations.exploratory_analysis import *


def main():

    print('New York Taxi Data Mining')
    train_df = load_train_data()
    train_df = data_preprocessing(train_df)
    visualize_data(train_df)
    print('End of Project')

if __name__ == '__main__':
    main()
