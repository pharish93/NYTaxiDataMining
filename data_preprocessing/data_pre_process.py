import pandas as pd
import time

pd.options.mode.chained_assignment = None
from lat_long_experiments import *
from sub_sample_data import sub_sample_data
from distance_experiments import add_distance_measures


def data_preprocessing(train_df):
    train_df = check_unique_id(train_df)
    train_df = extract_datetime(train_df)
    train_df = lat_long_bounds(train_df)
    train_df = lat_long_labeling(train_df)
    train_df = add_distance_measures(train_df)
    # For part 1
    train_df = sub_sample_data(train_df)

    return train_df


def check_unique_id(train_df):
    # checking if Ids are unique,
    train_data = train_df.copy()
    start = time.time()
    print("Number of columns and rows and columns are {} and {} respectively.".format(train_data.shape[1],
                                                                                      train_data.shape[0]))
    # if train_data.id.nunique() == train_data.shape[0]:
    #     print("Train ids are unique")
    # print("Number of Nulls - {}.".format(train_data.isnull().sum().sum()))
    train_data = train_data.dropna()
    end = time.time()
    print("Time taken to check_unique_id is {}.".format(end - start))
    return train_data


def extract_datetime(train_data):
    start = time.time()

    train_data['pickup_datetime'] = pd.to_datetime(train_data.pickup_datetime)
    train_data.loc[:, 'pick_date'] = train_data['pickup_datetime'].dt.date
    train_data.loc[:, 'pick_month'] = train_data['pickup_datetime'].dt.month
    train_data.loc[:, 'hour'] = train_data['pickup_datetime'].dt.hour
    train_data.loc[:, 'week_of_year'] = train_data['pickup_datetime'].dt.weekofyear
    train_data.loc[:, 'day_of_year'] = train_data['pickup_datetime'].dt.dayofyear
    train_data.loc[:, 'day_of_week'] = train_data['pickup_datetime'].dt.dayofweek

    # print(train_df.head())
    end = time.time()
    print("Time taken to modify datetime field is {}.".format(end - start))
    return train_data

