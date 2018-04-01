import seaborn as sns  # for making plots
import matplotlib.pyplot as plt  # for plotting
import time
import numpy as np
import pandas as pd
from data_preprocessing.data_pre_process import modify_datetime
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, show


def visualize_data(train_df):

    vis_lat_long(train_df)
    vis_trip_duration(train_df)

    # vis_trip_duration_month(train_df)
    plt.show()

def vis_trip_duration(train_df):
    start = time.time()
    sns.set(style="white", palette="muted", color_codes=True)
    f, axes = plt.subplots( 1, 1, figsize=(11, 7), sharex=True,num='Trip Duration')
    sns.despine(left=True)
    sns.distplot(np.log(train_df['trip_duration'].values + 1), axlabel='Log(trip_duration)', label='log(trip_duration)',
                 bins=50, color="r")
    plt.setp(axes, yticks=[])
    plt.title('Log Trip duration plot ')
    plt.tight_layout()
    end = time.time()
    print("Time taken to vis_trip_duration is {}.".format((end - start)))



def vis_trip_duration_month(train_data):
    start = time.time()
    temp = train_data.copy()
    # train_data = modify_datetime(train_data)
    ts_v1 = pd.DataFrame(train_data.loc[train_data['vendor_id'] == 1].groupby('pick_date')['trip_duration'].mean())
    ts_v1.reset_index(inplace=True)
    ts_v2 = pd.DataFrame(train_data.loc[train_data.vendor_id == 2].groupby('pick_date')['trip_duration'].mean())
    ts_v2.reset_index(inplace=True)

    p = figure(plot_width=800, plot_height=250, x_axis_type="datetime")
    p.title.text = 'Click on legend entries to hide the corresponding lines'

    for data, name, color in zip([ts_v1, ts_v2], ["vendor 1", "vendor 2"], Spectral4):
        df = data
        p.line(df['pick_date'], df['trip_duration'], line_width=2, color=color, alpha=0.8, legend=name)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    show(p)
    end = time.time()
    train_data = temp
    print("Time Taken by vis_trip_duration_month is {}.".format(end - start))

def vis_lat_long(df):

    start = time.time()
    longitude = list(df.pickup_longitude) + list(df.dropoff_longitude)
    latitude = list(df.pickup_latitude) + list(df.dropoff_latitude)
    plt.figure("pick up/drop points ",figsize=(10, 10))
    plt.plot(longitude, latitude, '.', alpha=0.4, markersize=0.05)
    # plt.show()

    end = time.time()
    print("Time Taken by vis_lat_long is {}.".format(end - start))