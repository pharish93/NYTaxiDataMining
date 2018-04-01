import seaborn as sns  # for making plots
import matplotlib.pyplot as plt  # for plotting
import time
import numpy as np


def vis_trip_duration(train_df):
    start = time.time()
    sns.set(style="white", palette="muted", color_codes=True)
    f, axes = plt.subplots(1, 1, figsize=(11, 7), sharex=True)
    sns.despine(left=True)
    sns.distplot(np.log(train_df['trip_duration'].values + 1), axlabel='Log(trip_duration)', label='log(trip_duration)',
                 bins=50, color="r")
    plt.setp(axes, yticks=[])
    plt.tight_layout()
    end = time.time()
    print("Time taken to vis_trip_duration is {}.".format((end - start)))
    plt.show()
