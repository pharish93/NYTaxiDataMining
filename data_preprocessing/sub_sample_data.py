import pandas as pd
import time
from visualizations.exploratory_analysis import vis_kmean_lat_long
import matplotlib.pyplot as plt
from lat_long_experiments import lat_long_bounds

def model1_sub_sample_data(train_df):
    start = time.time()
    days = [1, 4]
    hours = [8, 17]

    train_df = train_df[(train_df.day_of_week >= days[0]) & (train_df.day_of_week <= days[1])]
    train_df = train_df[(train_df.hour >= hours[0]) & (train_df.hour <= hours[1])]

    end = time.time()
    print("Time taken in sub_sample_data is {}.".format(end - start))
    return train_df

def model2_sub_sample_data(train_df):

    # Bound the data to lower middle manhattan
    xlim = [-74.03, -73.90]
    ylim = [40.69, 40.77]
    train_df = lat_long_bounds(train_df,xlim,ylim)

    # Bound on days and hours to remove seasonal nature od data
    days = [1, 4]
    hours = [6, 23]
    train_df = train_df[(train_df.day_of_week >= days[0]) & (train_df.day_of_week <= days[1])]
    train_df = train_df[(train_df.hour >= hours[0]) & (train_df.hour <= hours[1])]


    vis_kmean_lat_long(train_df)
    plt.savefig('model2_kmeans.png')
    plt.show()

    return train_df