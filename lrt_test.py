from data_loading.data_loading import *
from data_preprocessing.data_pre_process import *
from lrt.mle import *
import matplotlib.pyplot as plt  # for plotting

def main():

    train_df = load_train_data()
    train_df = data_preprocessing(train_df)
    plt.savefig('a1.png')
    lrt_cols = ['total_distance', 'total_travel_time', 'label_pick', 'label_drop', 'distance_haversine', 'trip_duration']
    train_data = train_df[lrt_cols]

    p_values = lrt_taxi_data(train_data)

    print(p_values)

if __name__ == '__main__':
    main()