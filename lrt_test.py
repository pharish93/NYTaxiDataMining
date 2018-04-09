from data_loading.data_loading import *
from data_preprocessing.data_pre_process import *
from mle import *

def main():
    train_df = load_train_data()
    train_df = data_preprocessing(train_df)

    print(train_df.columns)

    #p_values = lrt_taxi_data(train_df)


if __name__ == '__main__':
    main()