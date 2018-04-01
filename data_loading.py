import pandas as pd

def load_full_data():
    train_fr_1 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_1.csv')
    train_fr_2 = pd.read_csv('../input/new-york-city-taxi-with-osrm/fastest_routes_train_part_2.csv')
    train_fr = pd.concat([train_fr_1, train_fr_2])
    train_fr_new = train_fr[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
    train_df = pd.read_csv('../input/new-york-city-taxi-with-osrm/train.csv')
    train = pd.merge(train_df, train_fr_new, on='id', how='left')
    train_df = train.copy()

    print("Time taken by above cell is {}.".format((end - s)))
    train_df.head()