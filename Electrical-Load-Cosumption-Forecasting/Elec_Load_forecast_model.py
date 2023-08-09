import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import requests
from csv import writer
import json

raw_df = pd.DataFrame({})
weather_df = pd.DataFrame({})


def load_data():
    global raw_df
    raw_df = pd.read_csv(r'C:\Users\SR-19\Desktop\analog_report\LoadData\elec_load.csv')
    global weather_df
    weather_df = pd.read_csv(r'C:\Users\SR-19\Desktop\analog_report\LoadData\weather_data.csv')


def create_model(data, weather):
    if data.shape[0] == weather.shape[0]:
        for i in range(1, 7):
            data[f"t_{i}"] = data.KW.shift(i)
        for i in range(1, 6):
            data[f"day_{i}"] = data.KW.shift(24 * i)
        for i in range(1, 4):
            data[f"week_{i}"] = data.KW.shift(24 * 7 * i)

        # merging data and weather....
        df1 = pd.merge(data, weather, left_index=True, right_index=True)
        df2 = df1.dropna()

        arr = df2[[f"day_{i}" for i in range(1, 6)]].values
        df2.loc[:,'prev_5days_mean'] = np.mean(arr, axis=1)

        x = np.sort(arr, axis=1)
        df2.loc[:,'prev_2peak_mean'] = np.mean(x[:, 3:5], axis=1)

        arr1 = df2[[f"week_{i}" for i in range(1, 4)]].values
        df2.loc[:,'prev_3week_mean'] = np.mean(arr1, axis=1)

        cols = ['t_1', 't_2', 't_3', 't_4', 't_5', 't_6', 'prev_5days_mean', 'prev_2peak_mean', 'prev_3week_mean',
                'temp', 'pressure', 'humidity', 'dew_point',
                'speed', 'deg']

        Input = df2[cols].values
        output = df2.KW.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(Input)
        scaled = scaler.transform(Input)
        # applying pca.
        pca = PCA(0.99)
        pca = pca.fit(scaled)
        input_data = pca.transform(scaled)
        # crating input ..
        model = RandomForestRegressor(random_state=42,
                                      n_jobs=-1,
                                      n_estimators=100,
                                      max_depth=15,
                                      max_leaf_nodes=2 ** 10,
                                      min_impurity_decrease=1e-6)

        model.fit(input_data, output)

        return model, scaler, pca
    else:
        print("load data and weather had not have same inputs..")


def get_weather_data(starting_time):
    path = r'https://api.openweathermap.org/data/2.5/onecall?lat=26.5123&lon=80.2329&units=metric&appid=f4e0db85f5df8342e2d366d403220126'
    response_obj = requests.get(path)
    data = response_obj.json()
    LIST = []
    len = 0
    for ele in data['hourly']:
        if ele['dt'] > starting_time.total_seconds() and len < 24:
            LIST.append([data['temp'], data['pressure'], data['humidity'], data['dew_point'], data['wind_speed'],
                         data['wind_deg']])

    return LIST

def get_present(start_idx):
    data = weather_df.iloc[start_idx:(start_idx+ 25)]
    cols = ['temp', 'pressure', 'humidity', 'dew_point', 'speed', 'deg']
    return data[cols].values

def predict_day_ahead():
    day_ahead_load_forecast = []
    load_data()
    last_index_load = raw_df.tail(1).index.values[0]
    last_index_weather = weather_df.tail(1).index.values[0]
    # starting time should be just one hour ahead of last_index_load timing data
    load = raw_df[['KW']]
    weather = weather_df.iloc[:last_index_load + 1]
    cols = ['temp', 'pressure', 'humidity', 'dew_point', 'speed', 'deg']
    model, scaler, pca = create_model(load, weather[cols])

    if model is not None:
        # day_ahead_weather = forecast_weather_data(TYPE='DAY_AHEAD')
        day_ahead_weather = get_present(start_idx=(last_index_load+1))

        # taking approximately 4 week prior data...
        len_load_arr = 800
        load_arr = load.KW.values[-len_load_arr:].tolist()
        # preparaing Inputs...AND PREDICT AT EACH HOUR FOR 24 HOUR
        for i in range(24):
            # adding. t-1,t-2,t-3,t-4,t-5,t-6
            inp = [load_arr[i] for i in range(-1, -7, -1)]
            # adding..'prev_5days_mean'
            inp.append(np.mean([load_arr[-24 * i] for i in range(1, 6)]))
            # adding ..'prev_2peak_mean'
            inp.append(np.mean(np.sort([load_arr[-24 * i] for i in range(1, 6)])[3:5]))
            # adding ..'prev_3week_mean'
            inp.append(np.mean([load_arr[(-24 * 7 * i)] for i in range(1, 4)]))
            # day_ahead_weather is list of list each hour weather frorecatsed data..
            for val in day_ahead_weather[i]:
                inp.append(val)

            Inp = pca.transform(scaler.transform(np.array(inp).reshape(1, -1)))
            val = model.predict(Inp)
            load_arr.append(val[0])
            day_ahead_load_forecast.append(val[0])

    return day_ahead_load_forecast


# currently we have laod data till 2022-04-25 23:00:00
# currently we have laod data till 2022-05-03 23:00:00
# we will predict load at 26 april..24 hours...

if __name__ == "__main__":
    print(predict_day_ahead())