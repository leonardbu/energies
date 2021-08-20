%load_ext autoreload
%autoreload 2
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from math import sqrt
import pickle

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import statsmodels
from time import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import sklearn.metrics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.svm import SVR

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
import tensorflow as tf

import scipy
from scipy.stats import pearsonr

from pandas.plotting import register_matplotlib_converters
import helpers
import warnings
register_matplotlib_converters()
import math
import sklearn
warnings.simplefilter("ignore")

import json


#######################################


# Standard RNN Model
def create_model_rnn(learning_rate=1e-3, layers=1, neurons_per_layer=50, input_shape="", dense=True, cell_type="gru", optimizer="adam", dropout=0, **kwargs):
    model = Sequential()
    if(cell_type=="gru"):
        model.add(GRU(units=neurons_per_layer, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        if(layers==2):
            model.add(GRU(units=neurons_per_layer, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(dropout))
    if(cell_type=="lstm"):
        model.add(LSTM(units=neurons_per_layer, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout))
        if(layers==2):
            model.add(LSTM(units=neurons_per_layer, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(dropout))
            
    model.add(Dense(1, activation="relu"))
    model.add(Flatten())
    if(optimizer=="adam"):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if(optimizer=="rms"):
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    return model

# Standard ANN Model
def create_model_ann(learning_rate=1e-3, layers=1, number_of_features=10, neurons_per_layer=50, optimizer="adam", dropout=0, **kwargs):

    model = Sequential()
    model.add(Dense(neurons_per_layer, input_dim=number_of_features))
    model.add(Dropout(dropout))
    if(layers==2):
        model.add(Dense(neurons_per_layer, input_dim=neurons_per_layer))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="relu"))
    if(optimizer=="adam"):
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if(optimizer=="rms"):
        opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(loss="mean_squared_error", optimizer=opt, metrics=["accuracy"])
    return model
  
  

##########################################################

# We start with some configuration and initialization:

##########################################################

# Loading the combined datasets with best parameter configuration: 
datasets = helpers.importer.get_datasets_for_final_evaluation()
#    dataset in datasets is a combined dict that holds the data (key: dataset["set"])
#    and the configuration for each model, e.g. dataset["arima"] holds the selected order and sample_size
#    see helpers/importer.py->get_datasets_for_final_evaluation() for more details

# This is for the time elapsed printer after each window:
start_time = time()

# We can choose datasets to evaluate here. Set to selected_datasets = datasets if all datasets
# are to be evaluated.
# [0-2]: H, M, L -> VSTLF (15min resolution)
# [3-5]: H, M, L -> STLF (1h resolution)
# [6-8]: H, M, L -> MTLF (1d resolution)
selected_datasets = [ datasets[0], datasets[1], datasets[2] ]

# The prediction results are later stored in a dict by dataset and then by expanding window id
# e.g: all_predictions["H1@1h"][2] will return the DataFrame for the third window of the H1 STLF predictions
# Therefore the dict structure is here initialized:
all_predictions = {}
all_fitting_times = {}
all_prediction_times = {}


# train_start and test_end are static! They never change from one expanding window to another
# However they differ between the forecast horizons, as the lagged features are NaN for the first
# training samples. Therefore e.g. the train_start for the 30 day horizon is 2017-02-24 -> 30 days after the
# training set theoretically begins (2017-01-24).
cv_tests = {
    4: {
        "train_start": "2017-01-25",
        "test_end": "2019-03-07",
        "windows": helpers.importer.get_test_windows("15min"),
        # The windows then only contain information about where the training ends and the test starts.
        # The windows are loaded from helpers.importer and are different for each horizon
    },
    24: {
        "train_start": "2017-01-25",
        "test_end": "2019-03-07",
        "windows": helpers.importer.get_test_windows("1h"),
    },
    30: {
        "train_start": "2017-02-24",
        "test_end": "2019-03-07",
        "windows": helpers.importer.get_test_windows("1d")
    }
}

##########################################################

# Here we start Looping through the datasets and
# expanding windows

##########################################################


for dataset in selected_datasets:
    
    print("--  Started dataset", dataset["name"], "  --")
    
    # Initializing dict structure for the dataset: 
    all_predictions[dataset["name"]] = {}
    all_fitting_times[dataset["name"]] = {}
    all_prediction_times[dataset["name"]] = {}
    
    # Now we go through all the 20 expanding windows:
    for window in cv_tests[dataset["horizon"]]["windows"]:
    
        print("  Started window", window["id"]+1, " of ", len(cv_tests[dataset["horizon"]]["windows"]))

        train_start = cv_tests[dataset["horizon"]]["train_start"]
        train_end = window["train_end"]
        test_start = window["test_start"]
        test_end = cv_tests[dataset["horizon"]]["test_end"]

        # Creating lagged value feature: 
        dataset["set"]["Lagged"] = dataset["set"]["Load"].shift(periods=dataset["horizon"])

        # For the models that do not use lagged values we remove the Load and the Lagged values
        train_X = dataset["set"].loc[train_start:train_end].drop(labels=["Load","Lagged"], axis=1)
        test_X = dataset["set"].loc[test_start:test_end].drop(labels=["Load","Lagged"], axis=1)

        # For the models that make use of lagged values we remove only the values
        train_X_with_lag = dataset["set"].loc[train_start:train_end].drop(labels="Load", axis=1)
        test_X_with_lag = dataset["set"].loc[test_start:test_end].drop(labels="Load", axis=1)

        # Training y is more or less self explainatory
        train_y = dataset["set"].loc[train_start:train_end]["Load"]
        test_y = dataset["set"].loc[test_start:test_end]["Load"]

        # Set up of DataFrames for all predictions, fitting_times and prediction_times in this window:
        predictions = pd.DataFrame(columns=["Real"], index=dataset["set"].loc[test_start:test_end].index)
        fitting_times = pd.DataFrame(columns=["Real"], index=dataset["set"].loc[test_start:test_end].index)
        prediction_times = pd.DataFrame(columns=["Real"], index=dataset["set"].loc[test_start:test_end].index)
       
        # The measured Loads are stored in the Real column of the predictions dataframe
        predictions["Real"] = test_y
        
        fitted_models = {}

        #Naive 1:
        predictions["naive1"] = [train_y.iloc[-1]]*len(test_y)

        #Naive 2:
        naive2 = [*train_y.iloc[-dataset["seasonal"]:]]*(math.ceil(len(test_y)/dataset["seasonal"]))
        predictions["naive2"] = naive2[:len(test_y)]


        #ARIMA:
        
        # The sample size and the model order are parameters that are specified in the dataset config dict:
        model = ARIMA(train_y[-dataset["arima"]["sample_size"]:], order=dataset["arima"]["order"])
        timer_start = time()
        model_fit = model.fit() # The model is fitted
        fitting_times["arima"] = time() - timer_start
        timer_start = time()
        predictions["arima"] = model_fit.forecast(steps=len(test_X)).values 
        prediction_times["arima"] = time() - timer_start

        #SARIMA:
        if "sarima" in dataset:
            model = ARIMA(train_y[-dataset["sarima"]["sample_size"]:], order=dataset["sarima"]["order"], seasonal_order=dataset["sarima"]["seasonal_order"])
            timer_start = time()
            model_fit = model.fit()
            fitting_times["sarima"] = time() - timer_start
            timer_start = time()
            predictions["sarima"] = model_fit.forecast(steps=len(test_X)).values
            prediction_times["sarima"] = time() - timer_start

        if(len(dataset["arimax"]["features"])>0):
            #ARIMAX:
            
            # Here we select only the features that were specified
            arimax_exog_train = train_X[[*dataset["arimax"]["features"]]][-dataset["arima"]["sample_size"]:]
            arimax_exog_test = test_X[[*dataset["arimax"]["features"]]]
            
            
            # Problem: Static columns for ARIMAX are not allowed.
            # In VSTLF and STLF the mnth_cos and mnth_sin variables can be static!
            # Therefore we check if the standard deviation is very low. If it is the feature is removed.
            if('mnth_cos' in arimax_exog_train.columns): 
                if(arimax_exog_train["mnth_cos"].std() <= 1e-10):
                    arimax_exog_train = arimax_exog_train.drop(labels="mnth_cos", axis=1)
                    arimax_exog_test = arimax_exog_test.drop(labels="mnth_cos", axis=1)
            if('mnth_sin' in arimax_exog_train.columns):
                if(arimax_exog_train["mnth_sin"].std() <= 1e-10):
                    arimax_exog_train = arimax_exog_train.drop(labels="mnth_sin", axis=1)
                    arimax_exog_test = arimax_exog_test.drop(labels="mnth_sin", axis=1)
                    

            model = ARIMA(train_y[-dataset["arima"]["sample_size"]:], exog=arimax_exog_train)
            timer_start = time()
            model_fit = model.fit()
            fitting_times["arimax"] = time() - timer_start
            timer_start = time()
            predictions["arimax"] = model_fit.forecast(steps=len(test_X), exog=arimax_exog_test).values
            prediction_times["arimax"] = time() - timer_start
            
            
            #SARIMAX:
            if "sarima" in dataset:
                sarimax_exog_train = train_X[[*dataset["arimax"]["features"]]][-dataset["sarima"]["sample_size"]:]
                sarimax_exog_test = test_X[[*dataset["arimax"]["features"]]]
                model = ARIMA(train_y[-dataset["sarima"]["sample_size"]:], order=dataset["sarima"]["order"], seasonal_order=dataset["sarima"]["seasonal_order"], exog=sarimax_exog_train)
                timer_start = time()
                model_fit = model.fit()
                fitting_times["sarimax"] = time() - timer_start
                timer_start = time()
                predictions["sarimax"] =model_fit.forecast(steps=len(test_X), exog=sarimax_exog_test).values
                prediction_times["sarimax"] = time() - timer_start
            
        else: 
            # If we do not evaluate ARIMAX and SARIMAX we set their columns to NaN in order to have the
            # same column sizes everywhere
            fitting_times["arimax"] = np.nan
            fitting_times["sarimax"] = np.nan
            prediction_times["arimax"] = np.nan
            prediction_times["sarimax"] = np.nan
            predictions["sarimax"] = np.nan
            predictions["arimax"] = np.nan
           
        
        #HW: ES with both a trend and a seasonality
        model = ExponentialSmoothing(train_y[-dataset["hw"]["sample_size"]:], trend=dataset["hw"]["trend"], seasonal=dataset["hw"]["seasonal"], seasonal_periods=dataset["hw"]["seasonal_periods"])
        timer_start = time()
        model_fit = model.fit(smoothing_level=dataset["hw"]["smoothing_level"], smoothing_trend=dataset["hw"]["smoothing_trend"], smoothing_seasonal=dataset["hw"]["smoothing_seasonal"])
        fitting_times["hw"] = time() - timer_start
        timer_start = time()
        predictions["hw"] = model_fit.forecast(steps=len(test_X)).values
        prediction_times["hw"] = time() - timer_start
        

        #RF Implementation:
        model = RandomForestRegressor(**dataset["rf"], n_jobs=-1)
        timer_start = time()
        fitted_models["rf"] = model.fit(train_X_with_lag, train_y)
        fitting_times["rf"] = time() - timer_start
        timer_start = time()
        predictions["rf"] = fitted_models["rf"].predict(test_X_with_lag)
        prediction_times["rf"] = time() - timer_start

        #SVR Implementation:
        model = SVR(**dataset["svr"])
        timer_start = time()
        fitted_models["svr"] = model.fit(train_X, train_y)
        fitting_times["svr"] = time() - timer_start
        timer_start = time()
        predictions["svr"] = fitted_models["svr"].predict(test_X)
        prediction_times["svr"] = time() - timer_start

        #KNN Implementation:
        model = KNeighborsRegressor(**dataset["knn"], n_jobs=-1)
        timer_start = time()
        fitted_models["knn"] = model.fit(train_X_with_lag, train_y)
        fitting_times["knn"] = time() - timer_start
        timer_start = time()
        predictions["knn"] = fitted_models["knn"].predict(test_X_with_lag)
        prediction_times["knn"] = time() - timer_start
        
        
        # Keras Callback config for the early stopping after 50 validation losses:
        # This is a quite high number, but the best weights are then restored.
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
        
        # In addition to the early stopping we reduce the learning rate by 50% after 30 validation losses in a row
        rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30, min_lr=0.00001)
        
        
        # ANN 
        print("   ---> Starting ANN training")
        model=create_model_ann(**dataset["ann"], number_of_features=train_X_with_lag.shape[1])
        timer_start = time()
        model_fit = model.fit(train_X_with_lag, train_y, batch_size=30, epochs=400, shuffle=False, verbose=0, validation_split=0.1, callbacks=[early_stop_callback, rlrop])#, callbacks=[early_stop_callback, rlrop]
        fitting_times["ann"] = time() - timer_start
        timer_start = time()
        predictions["ann"] = model.predict(test_X_with_lag).flatten()
        prediction_times["ann"] = time() - timer_start
        
        #RNN
        print("   ---> Starting RNN training")
        train_X_rnn = np.expand_dims(train_X, 1) # RNN requires different input shape
        test_X_rnn = np.expand_dims(test_X, 1) # RNN requires different input shape
        model=create_model_rnn(**dataset["rnn"], input_shape=train_X_rnn.shape[1:])
        timer_start = time()
        model_fit = model.fit(train_X_rnn, train_y, batch_size=30, epochs=650, shuffle=False, verbose=0, validation_split=0.1, callbacks=[early_stop_callback, rlrop])
        fitting_times["rnn"] = time() - timer_start
        timer_start = time()
        predictions["rnn"] = model.predict(test_X_rnn).flatten()
        prediction_times["rnn"] = time() - timer_start
        
        
        # Finally we safe all the results in their corresponding dict:
        # Structured by Dataset Name and then by window id:
        
        all_predictions[dataset["name"]][window["id"]] = predictions[:dataset["horizon"]]
        all_fitting_times[dataset["name"]][window["id"]] = fitting_times[:dataset["horizon"]]
        all_prediction_times[dataset["name"]][window["id"]] = prediction_times[:dataset["horizon"]]
                
            
        print("---> Done with this step. Time elapsed:", time() - start_time,"s")
        
    # Better be safe: We save all prediction result after each dataset is done
    # There's nothing worse than an error in the last datset after hours of calculation with all predictions gone 
    pickle.dump( {"predictions": all_predictions[dataset["name"]], "fitting_times": all_fitting_times[dataset["name"]], "prediction_times": all_prediction_times[dataset["name"]]}, open( "results/forecasts/1d_"+ dataset["name"] +".p", "wb" ))

# In case everything has gone well we save the aggregated results for all datasets in one file
# In case something went wrong we can still aggregate this file by putting together all individual prediction files
pickle.dump( {"predictions": all_predictions, "fitting_times": all_fitting_times, "prediction_times": all_prediction_times}, open( "results/forecasts/1d_all.p", "wb" ))
