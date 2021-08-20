import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels
import sklearn
import requests
import socket
from sklearn import preprocessing

import scipy
from scipy.stats import pearsonr
from math import sqrt
from pandas.plotting import register_matplotlib_converters
import os

dirname = os.path.dirname(__file__)
scaler = sklearn.preprocessing.MinMaxScaler()


# Define the path to CSV datasets here:
PATH_TO_DATASET_H = ""
PATH_TO_DATASET_M = ""
PATH_TO_DATASET_L = ""


def create_features(df, resolution):
    
    # Month sin and cos features:
    df['mnth_sin'] = np.sin((df.index.month-1)*(2.*np.pi/12))
    df['mnth_cos'] = np.cos((df.index.month-1)*(2.*np.pi/12))
    
    # Workday Feature -> True if not weekand and not swedish public holiday
    df["is_workday"] = (df.index.weekday < 5) & (np.logical_not(np.isin(df.index.date, sweden_holidays)))*1
    
    months = [i for i in range(1, 13)]
    weekdays = [i for i in range(0, 7)]
    hours = [i for i in range(0, 24)]
    for month in months:
        df["month_" + str(month)] = (df.index.month == month)*1 # One hot encoding of current month
    for weekday in weekdays:
        df["weekday_" + str(weekday)] = (df.index.weekday == weekday)*1 # One hot encoding of current weekday
    if(not resolution == "1d"):
        # The hourly one hot encoded feature is only used for the 15min and 1h resolution, not for daily data:
        for hour in hours:
            df["hour_" + str(hour)] = (df.index.hour == hour)*1
            df['hr_sin'] = np.sin((df.index.hour + (df.index.minute/60)) * (2.*np.pi/24))
            df['hr_cos'] = np.cos((df.index.hour + (df.index.minute/60)) * (2.*np.pi/24))
    return df



def load_h(resolution = "1h", full_dataset = False, scaler_only=False):
    
    # We import the values of the datasets and define the index
    df = pd.read_csv(os.path.join(dirname, PATH_TO_DATASET_H), delimiter=";", index_col="Time")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M")
    df.index.freq = df.index.inferred_freq
    
    # Renaming of to english Feature Names:
    df["Windspeed"] = df["Vindhastighet"]
    df["Temperature"] = df["Lufttemperatur"]
 
    if(full_dataset):
        df = df["2017-01-24":"2019-03-07"]
    else:
        df = df["2017-01-24":"2018-12-07"]
        
    if(resolution=="15min"):
        # Down sampling to quarter-hourly resolution
        df = df.resample("15min").interpolate()
    if(resolution=="1d"):
        # Resampling to daily resolution
        df = df.resample("1d").mean()
        
    # Now we readd our newly generated features
    df = create_features(df, resolution)
    
    if(scaler_only==False):
        # Returns the scaled dataset
        df[["Load", "Windspeed","Temperature"]] = scaler.fit_transform(df[["Load", "Windspeed","Temperature"]].to_numpy())
        return df
    else:
        # Returns the scaler only -> used in economic part for inverse scaling back to original values
        return scaler.fit(df[["Load"]])
    
  
    

def load_m(resolution = "1h", full_dataset = False, scaler_only=False):
    df = pd.read_csv(PATH_TO_DATASET_M,index_col="Time")
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M")
    df.index.freq = df.index.inferred_freq
    
    if(full_dataset):
        df = df["2017-01-24":"2019-03-07"]
    else:
        df = df["2017-01-24":"2018-12-07"]
        
    if(resolution=="15min"):
        df = df.resample("15min").interpolate()
    if(resolution=="1d"):
        df = df.resample("1d").mean()
        
    # Now we readd the features
    df = create_features(df, resolution)
    
    if(scaler_only==False):
        df[["Load", "Windspeed","Temperature"]] = scaler.fit_transform(df[["Load", "Windspeed","Temperature"]].to_numpy())
        return df
    else:
        return scaler.fit(df[["Load"]])
    
    
    
    
    
def load_customer_l(resolution = "1h", full_dataset = False):
    df = pd.read_csv(os.path.join(dirname, PATH_TO_DATASET_L), usecols=["Time", "e507b", "Windspeed", "Temperature"], index_col="Time")
    df.index = pd.to_datetime(df.index)
    df.index.freq = df.index.inferred_freq

    if(full_dataset):
        df = df["2017-01-24":"2019-03-07"]
    else:
        df = df["2017-01-24":"2018-12-07"]
        
    if(resolution=="15min"):
        df = df.resample("15min").interpolate()
    if(resolution=="1d"):
        df = df.resample("1d").mean()
        
    # Now we readd the features
    df = create_features(df, resolution)
    
    df["Load"] = scaler.fit_transform(df[["Load"]].to_numpy())
    df["Windspeed"] = scaler.fit_transform(df[["Windspeed"]].to_numpy())
    df["Temperature"] = scaler.fit_transform(df[["Temperature"]].to_numpy())
    return df
    
    
    
sweden_holidays = ["2017-01-01", "2017-01-06", "2017-04-14", "2017-04-17", "2017-05-01", "2017-05-25", "2017-06-04", "2017-06-06", "2017-06-23", "2017-06-24", "2017-11-04", "2017-12-24", "2017-12-25", "2017-12-26", "2017-12-31", "2018-01-01", "2018-01-06", "2018-03-30", "2018-04-01", "2018-04-02", "2018-05-01", "2018-05-10", "2018-05-20", "2018-06-06", "2018-06-22", "2018-06-23", "2018-11-03", "2018-12-24", "2018-12-25", "2018-12-26", "2018-12-31", "2019-01-01", "2019-01-06"]
    
    
    
def get_datasets_with_config():
    return get_datasets_for_final_evaluation(full_dataset=False)
    
def get_datasets_for_final_evaluation(full_dataset=True):
    return [
        {   
            "name": "H@15min",
            "set": load_h(resolution="15min", full_dataset=True), 
            "horizon":4, 
            "seasonal": 96,
            "hw": {
                "smoothing_level": 1,
                "smoothing_trend": 0.99355,
                "smoothing_seasonal": 0.07195,
                "sample_size": 800,
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 96 
            },
            "svr": {
                "C": 0.125,
                "epsilon": 0.03125
            },
            "rf": {
                "n_estimators": 1100,
                "max_features": 0.5,
                "min_samples_leaf": 50
            },
            "knn": {
                "n_neighbors": 5,
                "weights": "distance"
            },
            "arima": {
                "order": (5,0,2),
                "sample_size": 650
            },
            "arimax": {
              "features": ["Temperature", "Windspeed", "hr_cos", "hr_sin"]  
            },
            "ann": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0.1
            },
            "rnn": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0.1,
                "cell_type": "gru"
            }
        },
        { 
            "name": "M@15min",
            "set": load_m(resolution="15min", full_dataset=True),
            "horizon":4,
            "seasonal": 96,
            "hw": {
                "smoothing_level": 0.35716,
                "smoothing_trend": 0.62701,
                "smoothing_seasonal": 0.57316,
                "sample_size": 500,
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 96 
            },
            "svr": {
                "C": 0.3125,
                "epsilon": 0.125
            },
            "rf": {
                "n_estimators": 800,
                "max_features": 0.33,
                "min_samples_leaf": 50
            },
            "knn": {
                "n_neighbors": 50,
                "weights": "uniform"
            },
            "arima": {
                "order": (5,0,3),
                "sample_size": 500
            },
            "arimax": {
              "features": ["Temperature", "Windspeed", "hr_cos", "mnth_cos", "mnth_sin", "is_workday"]  
            },
            "ann": {
                "layers": 2,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0.1
            },
            "rnn": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0.1,
                "cell_type": "gru"
            }
        },
        { 
            "set": load_l(resolution="15min", full_dataset=True),
            "name": "L@15min",
            "horizon":4,
            "seasonal": 96,
            "hw": {
                "smoothing_level": 0.16035,
                "smoothing_trend": 0.19047,
                "smoothing_seasonal": 0.58363,
                "sample_size": 500,
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 96 
            },
            "svr": {
                "C": 0.125,
                "epsilon": 0.125
            },
            "rf": {
                "n_estimators": 1400,
                "max_features": 0.2,
                "min_samples_leaf": 50
            },
            "knn": {
                "n_neighbors": 150,
                "weights": "uniform"
            },
            "arima": {
                "order": (1,0,5),
                "sample_size": 650
            },
            "arimax": {
              "features": ["Temperature", "Windspeed", "hr_cos"]  
            },
            "ann": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0
            },
            "rnn": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0,
                "cell_type": "lstm"
            }
        },
        { 
            "name": "H@1h",
            "set": load_h(full_dataset=full_dataset),
            "horizon":24, 
            "seasonal": 24,
            "svr": {
                "C": 0.5,
                "epsilon": 0.03125
            },
            "rf": {
                "n_estimators": 800,
                "max_features": 0.5,
                "min_samples_leaf": 5
            },
            "knn": {
                "n_neighbors": 8,
                "weights": "uniform"
            },
            "arima": {
                "order": (4,0,5),
                "sample_size": 800
            },
            "arimax": {
              "features": ["Temperature", "Windspeed", "hr_cos", "mnth_cos", "mnth_sin"]  
            },
            "sarima": {
                "order": (1,0,0),
                "seasonal_order": (2,0,0,24),
                "sample_size": 800
            },
            "hw": {
                "smoothing_level": 0.00836,
                "smoothing_trend": 0.0145,
                "smoothing_seasonal": 0.01246,
                "sample_size": 850,
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 24 
            },
            "ann": {
                "layers": 2,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0.1
            },
            "rnn": {
                "layers": 2,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0.1,
                "cell_type": "gru"
            }
        },
        { 
            "name": "M@1h",
            "set":load_m(full_dataset=full_dataset),
            "horizon":24,
            "seasonal": 24,
            "svr": {
                "C": 0.3125,
                "epsilon": 0.0625
            },
            "rf": {
                "n_estimators": 200,
                "max_features": 0.33,
                "min_samples_leaf": 25
            },
            "knn": {
                "n_neighbors": 15,
                "weights": "uniform"
            },
            "arima": {
                "order":(5,0,5),
                "sample_size": 900
            },
            "arimax": {
              "features": ["Temperature", "Windspeed", "hr_cos"]  
            },
            "sarima": {
                "order": (2,1,3),
                "seasonal_order": (2,0,1,24),
                "sample_size": 900

            },
            "hw": {
                "smoothing_level": 0.894132,
                "smoothing_trend": 0.31404,
                "smoothing_seasonal": 0.31394,
                "sample_size": 600,
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 24 
            },
            "ann": {
                "layers": 2,
                "learning_rate": 0.001,
                "neurons_per_layer": 128,
                "optimizer": "adam",
                "dropout": 0.1
            },
            "rnn": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0.1,
                "cell_type": "gru"
            }
        },
        { 
            "name": "L@1h",
            "set": load_l(full_dataset=full_dataset),
            "horizon":24,
            "seasonal": 24,
            "svr": {
                "C": 0.5,
                "epsilon": 0.125
            },
            "rf": {
                "n_estimators": 500,
                "max_features": 0.5,
                "min_samples_leaf": 50
            },
            "knn": {
                "n_neighbors": 150,
                "weights": "uniform"
            },
            "arima": {
                "order": (2,1,1),
                "sample_size": 500
            },
            "arimax": {
              "features": ["Temperature", "Windspeed", "hr_cos", ]  
            },
            "sarima": {
                "order": (3,1,0),
                "seasonal_order": (2,0,0,24),
                "sample_size": 500

            },
            "hw": {
                "smoothing_level": 0.00154,
                "smoothing_trend": 0.17787,
                "smoothing_seasonal": 0.17986,
                "sample_size": 200,
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 7
            },
            "ann": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 32,
                "optimizer": "adam",
                "dropout": 0
            },
            "rnn": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 32,
                "optimizer": "adam",
                "dropout": 0,
                "cell_type": "gru"
            }
        },
        {
            "name": "H@1d",
            "set": load_h(resolution="1d", full_dataset=full_dataset), 
            "horizon":30,
            "seasonal": 7,
            "svr": {
                "C": 2,
                "epsilon": 0.0078125
            },
            "rf": {
                "n_estimators": 1100,
                "max_features": 0.5,
                "min_samples_leaf": 5
            },
            "knn": {
                "n_neighbors": 70,
                "weights": "distance"
            },
            "arima": {
                "order": (5,1,3),
                "sample_size": 250
            },
            "arimax": {
              "features": ["Temperature", "Windspeed", "is_workday", ]  
            },
            "sarima": {
                "order": (0,1,1),
                "seasonal_order": (1,0,1,7),
                "sample_size": 450

            },
            "hw": {
                "smoothing_level": 0.03257,
                "smoothing_trend": 0.00125,
                "smoothing_seasonal": 0.00937,
                "sample_size": 500,
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 7
            },
            "ann": {
                "layers": 2,
                "learning_rate": 0.001,
                "neurons_per_layer": 128,
                "optimizer": "adam",
                "dropout": 0.1
            },
            "rnn": {
                "layers": 2,
                "learning_rate": 0.001,
                "neurons_per_layer": 128,
                "optimizer": "adam",
                "dropout": 0,
                "cell_type": "gru"
            }
        },
        
        {
            "name": "M@1d",        
            "set": load_m(resolution="1d", full_dataset=full_dataset), 
            "horizon":30,
            "seasonal": 7,
            "svr": {
                "C": 2,
                "epsilon": 0.125
            },
            "rf": {
                "n_estimators": 1100,
                "max_features": 0.33,
                "min_samples_leaf": 5
            },
            "knn": {
                "n_neighbors": 70,
                "weights": "distance"
            },
            "arima": {
                "order": (4,1,2),
                "sample_size": 450
            },
            "arimax": {
              "features": ["Temperature", "Windspeed", "mnth_sin", "is_workday" ]  
            },
            "sarima": {
                "order": (5,0,2),
                "seasonal_order": (1,0,0,7),
                "sample_size": 450

            },
            "hw": {
                "smoothing_level": 0.00009,
                "smoothing_trend": 0.35317,
                "smoothing_seasonal": 0.00027,
                "sample_size": 450,
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 7
            },
            "ann": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 128,
                "optimizer": "adam",
                "dropout": 0
            },
            "rnn": {
                "layers": 2,
                "learning_rate": 0.001,
                "neurons_per_layer": 16,
                "optimizer": "adam",
                "dropout": 0.1,
                "cell_type": "gru"
            }
        },
       
        {
            "name": "L@1d",
            "set": load_l(resolution="1d", full_dataset=full_dataset), 
            "horizon":30,
            "seasonal": 7,
            "svr": {
                "C": 0.125,
                "epsilon": 0.125
            },
            "rf": {
                "n_estimators": 1100,
                "max_features": 0.5,
                "min_samples_leaf": 50
            },
            "knn": {
                "n_neighbors": 120,
                "weights": "uniform"
            },
            "arima": {
                "order": (5,0,5),
                "sample_size": 450
            },
            "arimax": {
              "features": ["Temperature", "Windspeed", "mnth_cos", "is_workday" ]  
            },
            "sarima": {
                "order": (1,1,2),
                "seasonal_order": (2,0,2,7),
                "sample_size": 450

            },
            "hw": {
                "smoothing_level": 0.048975,
                "smoothing_trend": 0.162792,
                "smoothing_seasonal": 0.16720,
                "sample_size": 500,
                "trend": "add",
                "seasonal": "add",
                "seasonal_periods": 7
            },
            "ann": {
                "layers": 2,
                "learning_rate": 0.001,
                "neurons_per_layer": 128,
                "optimizer": "rms",
                "dropout": 0
            },
            "rnn": {
                "layers": 1,
                "learning_rate": 0.001,
                "neurons_per_layer": 64,
                "optimizer": "rms",
                "dropout": 0.1,
                "cell_type": "lstm"
            }
        }
    ]

def get_test_windows(resolution="1h"):
    if(resolution=="15min"):
        return [{'id': 0, 'train_end': '2018-12-07 00:00:00', 'test_start': '2018-12-07 00:15:00'}, {'id': 1, 'train_end': '2018-12-07 00:45:00', 'test_start': '2018-12-07 01:00:00'}, {'id': 2, 'train_end': '2018-12-07 01:30:00', 'test_start': '2018-12-07 01:45:00'}, {'id': 3, 'train_end': '2018-12-07 02:15:00', 'test_start': '2018-12-07 02:30:00'}, {'id': 4, 'train_end': '2018-12-07 03:15:00', 'test_start': '2018-12-07 03:30:00'}, {'id': 5, 'train_end': '2018-12-07 04:00:00', 'test_start': '2018-12-07 04:15:00'}, {'id': 6, 'train_end': '2018-12-07 04:45:00', 'test_start': '2018-12-07 05:00:00'}, {'id': 7, 'train_end': '2018-12-07 05:45:00', 'test_start': '2018-12-07 06:00:00'}, {'id': 8, 'train_end': '2018-12-07 06:30:00', 'test_start': '2018-12-07 06:45:00'}, {'id': 9, 'train_end': '2018-12-07 07:15:00', 'test_start': '2018-12-07 07:30:00'}, {'id': 10, 'train_end': '2018-12-07 08:15:00', 'test_start': '2018-12-07 08:30:00'}, {'id': 11, 'train_end': '2018-12-07 09:00:00', 'test_start': '2018-12-07 09:15:00'}, {'id': 12, 'train_end': '2018-12-07 09:45:00', 'test_start': '2018-12-07 10:00:00'}, {'id': 13, 'train_end': '2018-12-07 10:45:00', 'test_start': '2018-12-07 11:00:00'}, {'id': 14, 'train_end': '2018-12-07 11:30:00', 'test_start': '2018-12-07 11:45:00'}, {'id': 15, 'train_end': '2018-12-07 12:15:00', 'test_start': '2018-12-07 12:30:00'}, {'id': 16, 'train_end': '2018-12-07 13:15:00', 'test_start': '2018-12-07 13:30:00'}, {'id': 17, 'train_end': '2018-12-07 14:00:00', 'test_start': '2018-12-07 14:15:00'}, {'id': 18, 'train_end': '2018-12-07 14:45:00', 'test_start': '2018-12-07 15:00:00'}, {'id': 19, 'train_end': '2018-12-07 15:45:00', 'test_start': '2018-12-07 16:00:00'}]
    if(resolution=="1h"):
        return [{'id': 0, 'train_end': '2018-12-07 00:00:00', 'test_start': '2018-12-07 01:00:00'}, {'id': 1, 'train_end': '2018-12-11 17:00:00', 'test_start': '2018-12-11 18:00:00'}, {'id': 2, 'train_end': '2018-12-16 10:00:00', 'test_start': '2018-12-16 11:00:00'}, {'id': 3, 'train_end': '2018-12-21 04:00:00', 'test_start': '2018-12-21 05:00:00'}, {'id': 4, 'train_end': '2018-12-25 21:00:00', 'test_start': '2018-12-25 22:00:00'}, {'id': 5, 'train_end': '2018-12-30 14:00:00', 'test_start': '2018-12-30 15:00:00'}, {'id': 6, 'train_end': '2019-01-04 08:00:00', 'test_start': '2019-01-04 09:00:00'}, {'id': 7, 'train_end': '2019-01-09 01:00:00', 'test_start': '2019-01-09 02:00:00'}, {'id': 8, 'train_end': '2019-01-13 18:00:00', 'test_start': '2019-01-13 19:00:00'}, {'id': 9, 'train_end': '2019-01-18 12:00:00', 'test_start': '2019-01-18 13:00:00'}, {'id': 10, 'train_end': '2019-01-23 05:00:00', 'test_start': '2019-01-23 06:00:00'}, {'id': 11, 'train_end': '2019-01-27 23:00:00', 'test_start': '2019-01-28 00:00:00'}, {'id': 12, 'train_end': '2019-02-01 16:00:00', 'test_start': '2019-02-01 17:00:00'}, {'id': 13, 'train_end': '2019-02-06 09:00:00', 'test_start': '2019-02-06 10:00:00'}, {'id': 14, 'train_end': '2019-02-11 03:00:00', 'test_start': '2019-02-11 04:00:00'}, {'id': 15, 'train_end': '2019-02-15 20:00:00', 'test_start': '2019-02-15 21:00:00'}, {'id': 16, 'train_end': '2019-02-20 13:00:00', 'test_start': '2019-02-20 14:00:00'}, {'id': 17, 'train_end': '2019-02-25 07:00:00', 'test_start': '2019-02-25 08:00:00'}, {'id': 18, 'train_end': '2019-03-02 00:00:00', 'test_start': '2019-03-02 01:00:00'}, {'id': 19, 'train_end': '2019-03-06 18:00:00', 'test_start': '2019-03-06 19:00:00'}]
    if(resolution=="1d"):
        return [{'id': 0, 'train_end': '2018-12-07 00:00:00', 'test_start': '2018-12-08 00:00:00'}, {'id': 1, 'train_end': '2018-12-10 00:00:00', 'test_start': '2018-12-11 00:00:00'}, {'id': 2, 'train_end': '2018-12-13 00:00:00', 'test_start': '2018-12-14 00:00:00'}, {'id': 3, 'train_end': '2018-12-16 00:00:00', 'test_start': '2018-12-17 00:00:00'}, {'id': 4, 'train_end': '2018-12-19 00:00:00', 'test_start': '2018-12-20 00:00:00'}, {'id': 5, 'train_end': '2018-12-23 00:00:00', 'test_start': '2018-12-24 00:00:00'}, {'id': 6, 'train_end': '2018-12-26 00:00:00', 'test_start': '2018-12-27 00:00:00'}, {'id': 7, 'train_end': '2018-12-29 00:00:00', 'test_start': '2018-12-30 00:00:00'}, {'id': 8, 'train_end': '2019-01-01 00:00:00', 'test_start': '2019-01-02 00:00:00'}, {'id': 9, 'train_end': '2019-01-04 00:00:00', 'test_start': '2019-01-05 00:00:00'}, {'id': 10, 'train_end': '2019-01-08 00:00:00', 'test_start': '2019-01-09 00:00:00'}, {'id': 11, 'train_end': '2019-01-11 00:00:00', 'test_start': '2019-01-12 00:00:00'}, {'id': 12, 'train_end': '2019-01-14 00:00:00', 'test_start': '2019-01-15 00:00:00'}, {'id': 13, 'train_end': '2019-01-17 00:00:00', 'test_start': '2019-01-18 00:00:00'}, {'id': 14, 'train_end': '2019-01-20 00:00:00', 'test_start': '2019-01-21 00:00:00'}, {'id': 15, 'train_end': '2019-01-24 00:00:00', 'test_start': '2019-01-25 00:00:00'}, {'id': 16, 'train_end': '2019-01-27 00:00:00', 'test_start': '2019-01-28 00:00:00'}, {'id': 17, 'train_end': '2019-01-30 00:00:00', 'test_start': '2019-01-31 00:00:00'}, {'id': 18, 'train_end': '2019-02-02 00:00:00', 'test_start': '2019-02-03 00:00:00'}, {'id': 19, 'train_end': '2019-02-06 00:00:00', 'test_start': '2019-02-07 00:00:00'}]
def return_pearson_r(x,y):
    return scipy.stats.pearsonr(x,y)[0]
