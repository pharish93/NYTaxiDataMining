import pandas as pd
import time

def check_unique_id(train_df):
    # checking if Ids are unique,
    start = time.time()
    train_data = train_df.copy()
    start = time.time()
    print("Number of columns and rows and columns are {} and {} respectively.".format(train_data.shape[1], train_data.shape[0]))
    if train_data.id.nunique() == train_data.shape[0]:
        print("Train ids are unique")
    print("Number of Nulls - {}.".format(train_data.isnull().sum().sum()))
    end = time.time()
    print("Time taken to check_unique_id is {}.".format(end-start))