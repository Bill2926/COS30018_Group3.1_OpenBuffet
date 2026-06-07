# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)
# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import tensorflow as tf
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Input

#----------------------------------------------------------------
# Task C.2: Allow flexible input date range for stock dataset
#----------------------------------------------------------------
def load_and_process_dataset(
        company: str, 
        start_date: str, 
        end_date: str, 
        features: list | None = None, 
        data_dir: str = 'data', 
        force_download: bool = False,
        nan_method: str = 'forward_fill',
        split_method: str = 'date',
        split_param: str = '2023-01-01',
        scale_columns: bool = False
    ):
    
    # [REQUIREMENT A] Specify start date and end date for the whole dataset
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if start_dt >= end_dt:
        raise ValueError("Start Date must be before the End Date.")
    
    # [REQUIREMENT D] Store downloaded data locally and load locally to save time
    os.makedirs(data_dir, exist_ok=True)
    data_file = os.path.join(data_dir, f'{company}_{start_date}_{end_date}.csv')
    
    df = pd.DataFrame()
    if os.path.exists(data_file) and not force_download:
        print(f"Loading cached dataset from {data_file}...")
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        # Can't find the cache dataset file
        print(f"Downloading full dataset for {company}...")
        df = yf.download(company, start_dt, end_dt, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(data_file)

    if features is not None:
        available_features = [f for f in features if f in df.columns]
        df = df[available_features]

    # [REQUIREMENT B] Deal with the NaN issue in the data
    # forward_fill: take the last value to fill (e.g. on Friday)
    # backward_fill: take the first value from the future to fill -> data leakage
    if nan_method == 'forward_fill':
        df = df.ffill()
    elif nan_method == 'drop':
        df = df.dropna()
    elif nan_method == 'mean':
        df = df.fillna(df.mean())
    df = df.bfill() # Safety fallback

    # [REQUIREMENT C] Use different methods to split the data into train/test data
    # based on split-method and split-param arguments
    if split_method == 'date':                              # Split with date
        cutoff = pd.Timestamp(split_param)
        train_df = df[df.index < cutoff].copy()
        test_df = df[df.index >= cutoff].copy()
    elif split_method == 'ratio':                           # Split with ratio
        split_idx = int(len(df) * float(split_param))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    else:
        train_df = df.copy()
        test_df = pd.DataFrame()

    # [REQUIREMENT E] Option to scale feature columns and store scalers in a data structure
    scalers = {}
    if scale_columns:
        for col in train_df.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_df[col] = scaler.fit_transform(train_df[[col]])
            if len(test_df) > 0:
                test_df[col] = scaler.transform(test_df[[col]])
            scalers[col] = scaler
            
    return train_df, test_df, scalers

#----------------------------------------------------------------
# Load Data
#----------------------------------------------------------------
COMPANY = 'NVDA'
TRAIN_START = '2020-01-01'     
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'       

DATA_DIR = 'data'

# Note: scale_columns is False so your existing scaling logic below remains perfectly intact
train_data, test_data, feature_scalers = load_and_process_dataset(
    company=COMPANY,
    start_date=TRAIN_START,
    end_date=TEST_END,
    features=None,
    data_dir=DATA_DIR,
    force_download=False,
    nan_method='forward_fill',
    split_method='date',
    split_param=TEST_START,
    scale_columns=False
)

# Bridge the function's output to the exact variable name your script expects
data = train_data

#----------------------------------------------------------------
# Prepare Data
#----------------------------------------------------------------
PRICE_VALUE = "Close"

scaler = MinMaxScaler(feature_range=(0, 1))
# Note that, by default, feature_range=(0, 1). Thus, if you want a different
# feature_range (min,max) then you'll need to specify it here
scaled_data = scaler.fit_transform(data[PRICE_VALUE].values.reshape(-1, 1))

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60 

# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:, 0]  # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x-PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now reshape x_train into a 3D array(p, q, 1); Note that x_train
# is an array of p inputs with each input being a 2D array

#----------------------------------------------------------------
# Build the Model
#----------------------------------------------------------------
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
model_file = os.path.join(MODEL_DIR, f'{COMPANY}_model.keras')

if os.path.exists(model_file):
    print(f"Loading saved model from {model_file}...")
    model = load_model(model_file)
else:
    print("Building and training new model...")
    model = Sequential()  # Basic neural network

    # Use Input layer (functional-style) instead of deprecated InputLayer
    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))

    model.add(Dropout(0.2))
    # The Dropout layer randomly sets input units to 0 with a frequency of
    # rate (= 0.2 above) at each step during training time, which helps
    # prevent overfitting (one of the major problems of ML).

    model.add(LSTM(units=50, return_sequences=True))
    # More on Stacked LSTM:
    # https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))
    # Prediction of the next closing value of the stock price

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Now we are going to train this model with our training data
    # (x_train, y_train)
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Save the model for future use
    model.save(model_file)
    print(f"Model saved to {model_file}")

#----------------------------------------------------------------
# Test the model accuracy on existing data
#----------------------------------------------------------------
# The test data downloading logic was removed here because the load_and_process_dataset 
# function already downloaded, split, and returned the test_data above!

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((data[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the first
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# The above line reshapes model_inputs from a 1D array of shape (n,)
# to a 2D array of shape (n, 1). This is required by scaler.transform()
# which expects a 2D array as input.

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we
# can use part of it for training and the rest for testing. You need to
# implement such a way

#----------------------------------------------------------------
# Make predictions on test data
#----------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# Explanation of the above 5 lines:
# 1. x_test = [] : Initialize an empty list to hold input sequences
# 2. The for loop: for each time step x (starting after PREDICTION_DAYS),
#    we slice the previous PREDICTION_DAYS values as one input window
# 3. x_test = np.array(x_test): Convert the list of windows into a 2D NumPy array
#    of shape (num_samples, PREDICTION_DAYS)
# 4. x_test = np.reshape(..., (x_test.shape[0], x_test.shape[1], 1)):
#    Reshape to 3D array (num_samples, PREDICTION_DAYS, 1) as required
#    by the LSTM layer which expects input shape (batch, timesteps, features)

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation

#----------------------------------------------------------------
# Plot the test predictions
#----------------------------------------------------------------
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#----------------------------------------------------------------
# Predict next day
#----------------------------------------------------------------
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")