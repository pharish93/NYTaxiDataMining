import pandas as pd
import time


def model1_sub_sample_data(train_df):
    start = time.time()
    days = [1, 4]
    hours = [8, 17]

    train_df = train_df[(train_df.day_of_week >= days[0]) & (train_df.day_of_week <= days[1])]
    train_df = train_df[(train_df.hour >= hours[0]) & (train_df.hour <= hours[1])]

    end = time.time()
    print("Time taken in sub_sample_data is {}.".format(end - start))
    return train_df
