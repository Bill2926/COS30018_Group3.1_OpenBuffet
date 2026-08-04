# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3); 30/05/2026 (v4 - Task C.1)
#          06/06/2026 (v5 - Task C.2d, C.2e: local caching & feature scaling)

# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance
# pip install joblib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import tensorflow as tf
import joblib  # For saving/loading scaler objects to/from disk (Task 1e)

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

#==============================================================================
# TASK 1d - Load Data with Local Caching Option
#==============================================================================
# This function encapsulates the entire data-loading workflow:
#   - Downloads stock data from Yahoo Finance via the yfinance library.
#   - Optionally caches the downloaded data as a CSV file on the local machine.
#   - On subsequent runs, if the cached file exists, it loads from disk instead
#     of re-downloading — saving time and reducing API calls.
#   - Also handles NaN values (Task 1b) using the specified strategy.
#
# Parameters:
#   ticker (str):       The stock ticker symbol (e.g., 'CBA.AX', 'AAPL').
#   start_date (str):   The start date for data retrieval in 'YYYY-MM-DD' format.
#   end_date (str):     The end date for data retrieval in 'YYYY-MM-DD' format.
#   save_local (bool):  If True (default), saves downloaded data to a local CSV
#                       file and loads from it on future calls. If False, always
#                       downloads fresh data from the internet.
#   data_dir (str):     The directory where cached CSV files are stored.
#                       Defaults to 'data'. The directory is auto-created if needed.
#   nan_strategy (str): How to handle NaN/missing values in the data.
#                       Options: 'drop' (remove rows with NaN),
#                                'ffill' (forward-fill: propagate last valid value),
#                                'bfill' (backward-fill: use next valid value),
#                                None (leave NaN values as-is).
#                       Defaults to 'ffill' which is the most common approach
#                       for time-series financial data.
#
# Returns:
#   pd.DataFrame: A DataFrame containing the stock data with columns like
#                 'Open', 'High', 'Low', 'Close', 'Volume', etc.
#                 The index is a DatetimeIndex.
#------------------------------------------------------------------------------
def load_data(ticker, start_date, end_date, save_local=True,
              data_dir='data', nan_strategy='ffill'):
    """
    Download stock data from Yahoo Finance with optional local caching.

    Task 1d: Provides the option to store downloaded data on the local machine
    for future uses and to load data locally to save time.

    Task 1b (integrated): Handles NaN values using the specified strategy.
    """
    # Build a unique filename based on ticker and date range so that
    # different queries don't overwrite each other's cached files.
    # Example: 'data/CBA.AX_2020-01-01_2023-08-01.csv'
    data_file = os.path.join(data_dir, f'{ticker}_{start_date}_{end_date}.csv')

    # --- Attempt to load from local cache first (if caching is enabled) ---
    if save_local and os.path.exists(data_file):
        # The file already exists on disk from a previous download.
        # Load it directly — this is much faster than downloading again.
        print(f"[Cache HIT] Loading saved data from {data_file}...")
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
        # index_col=0: Use the first column (Date) as the DataFrame index.
        # parse_dates=True: Automatically parse the index as datetime objects.
    else:
        # Either caching is disabled, or the file doesn't exist yet.
        # Download fresh data from Yahoo Finance.
        print(f"[Download] Fetching data for {ticker} "
              f"from {start_date} to {end_date}...")
        import yfinance as yf  # Lazy import: only loaded when actually needed
        df = yf.download(ticker, start_date, end_date, auto_adjust=True)
        # auto_adjust=True: Automatically adjusts OHLC prices for splits and
        # dividends, giving us the "adjusted" prices directly.

        # yfinance 0.2+ returns MultiIndex columns (e.g., ('Close', 'CBA.AX')).
        # We flatten them to simple column names (e.g., 'Close') for easier use.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # --- Save to local disk if caching is enabled ---
        if save_local:
            # Create the data directory if it doesn't exist yet.
            # exist_ok=True prevents an error if the directory already exists.
            os.makedirs(data_dir, exist_ok=True)
            df.to_csv(data_file)
            print(f"[Saved] Data cached to {data_file}")

    # --- Handle NaN/missing values (Task 1b) ---
    # Financial data can have missing values due to holidays, data gaps, etc.
    # We provide multiple strategies to deal with this.
    if nan_strategy == 'drop':
        # Remove any row that contains at least one NaN value.
        # This is the simplest approach but may lose data points.
        df.dropna(inplace=True)
    elif nan_strategy == 'ffill':
        # Forward-fill: replace NaN with the last known valid value.
        # This is the most common strategy for time-series data because
        # it assumes the price stays the same until a new value arrives.
        df.ffill(inplace=True)
    elif nan_strategy == 'bfill':
        # Backward-fill: replace NaN with the next valid value.
        # Useful in some scenarios but less common for financial data.
        df.bfill(inplace=True)
    # If nan_strategy is None, we leave NaN values untouched.

    return df


#==============================================================================
# TASK 1e - Scale Feature Columns with Scaler Storage
#==============================================================================
# This function scales one or more feature columns in a DataFrame using
# MinMaxScaler, and stores each fitted scaler in a dictionary so they can be
# reused later (e.g., for inverse-transforming predictions back to original
# price values, or for transforming new test data consistently).
#
# Why store scalers in a dictionary?
#   When we use multiple features (e.g., 'Open', 'High', 'Low', 'Close'),
#   each feature has its own range (min/max). We need a SEPARATE scaler for
#   each feature so that we can correctly inverse-transform predictions later.
#   The dictionary maps each column name to its fitted MinMaxScaler instance.
#
# Parameters:
#   df (pd.DataFrame):           The DataFrame containing the raw stock data.
#   feature_columns (list[str]): A list of column names to scale.
#                                Example: ['Close'] or ['Open', 'High', 'Low', 'Close', 'Volume']
#   scale_range (tuple):         The target range for scaling. Defaults to (0, 1).
#                                MinMaxScaler maps all values to this range.
#   save_scalers (bool):         If True, saves the scaler dictionary to disk
#                                using joblib for future reuse. Defaults to False.
#   scaler_file (str):           Path to save/load the scaler dictionary.
#                                Defaults to 'scalers/scalers.pkl'.
#
# Returns:
#   tuple: (scaled_data, scaler_dict)
#     - scaled_data (np.ndarray): A 2D array of shape (n_samples, n_features)
#                                 with all values scaled to the specified range.
#     - scaler_dict (dict):       A dictionary mapping each feature column name
#                                 to its fitted MinMaxScaler object.
#                                 Example: {'Close': MinMaxScaler(...), 'Volume': MinMaxScaler(...)}
#                                 These scalers can be used later to call:
#                                   scaler_dict['Close'].inverse_transform(predictions)
#                                 to convert scaled values back to original prices.
#------------------------------------------------------------------------------
def scale_features(df, feature_columns, scale_range=(0, 1),
                   save_scalers=False, scaler_file='scalers/scalers.pkl'):
    """
    Scale specified feature columns using MinMaxScaler and store the scalers.

    Task 1e: Provides an option to scale feature columns and store the scalers
    in a data structure (dictionary) to allow future access to these scalers.

    Each feature column gets its own independent MinMaxScaler so that
    inverse_transform can be applied per-feature when making predictions.
    """
    # Dictionary to store one MinMaxScaler per feature column.
    # This is the "data structure" mentioned in Task 1e that allows
    # future access to scalers (e.g., for inverse_transform on predictions).
    scaler_dict = {}

    # List to collect each column's scaled values, which will be
    # horizontally stacked into a single 2D array at the end.
    scaled_columns = []

    for col in feature_columns:
        # Create a new MinMaxScaler for this specific feature column.
        # Each column gets its own scaler because different features
        # (e.g., 'Close' vs 'Volume') have vastly different value ranges.
        scaler = MinMaxScaler(feature_range=scale_range)

        # Extract the column values and reshape from 1D array (n,) to 2D (n, 1)
        # because sklearn's fit_transform() requires a 2D array as input.
        col_values = df[col].values.reshape(-1, 1)

        # fit_transform does two things:
        #   1. fit(): Learns the min/max of this column's data.
        #   2. transform(): Scales all values to the specified range (0, 1).
        scaled_col = scaler.fit_transform(col_values)

        # Store the fitted scaler in the dictionary, keyed by column name.
        # This allows us to later retrieve it as: scaler_dict['Close']
        scaler_dict[col] = scaler

        # Collect the scaled column for later horizontal stacking.
        scaled_columns.append(scaled_col)

    # Stack all scaled columns horizontally into a single 2D NumPy array.
    # If feature_columns = ['Open', 'Close', 'Volume'], the result will have
    # shape (n_samples, 3), where each column corresponds to one feature.
    scaled_data = np.hstack(scaled_columns)

    # --- Optionally save the scaler dictionary to disk using joblib ---
    # This allows us to reload the exact same scalers in a later session
    # (e.g., when deploying the model or running inference on new data)
    # without needing to re-fit on the training data.
    if save_scalers:
        scaler_dir = os.path.dirname(scaler_file)
        if scaler_dir:  # Only create directory if path includes a directory
            os.makedirs(scaler_dir, exist_ok=True)
        joblib.dump(scaler_dict, scaler_file)
        print(f"[Saved] Scaler dictionary saved to {scaler_file}")
        # To reload later: scaler_dict = joblib.load('scalers/scalers.pkl')

    return scaled_data, scaler_dict


#==============================================================================
# Configuration Constants
#==============================================================================
# DATA_SOURCE = "yahoo"
COMPANY = 'CBA.AX'

TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

#------------------------------------------------------------------------------
# Load Training Data using load_data() (Task 1d)
#------------------------------------------------------------------------------
# save_local=True: The first run downloads from Yahoo Finance and saves to
#   'data/CBA.AX_2020-01-01_2023-08-01.csv'. All subsequent runs load from
#   this local CSV file instantly, avoiding slow internet downloads.
# nan_strategy='ffill': Forward-fills any missing data points.
data = load_data(COMPANY, TRAIN_START, TRAIN_END,
                 save_local=True, data_dir='data', nan_strategy='ffill')

#------------------------------------------------------------------------------
# Prepare Data
# TO DO (v4 - implemented):
# 1) Check if data has been prepared before.
#    If so, load the saved data
#    If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------
PRICE_VALUE = "Close"

# --- FEATURE_COLUMNS: Define which columns to use as input features ---
# Currently using only 'Close' price. You can extend this to use multiple
# features, e.g.: FEATURE_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
# The scale_features() function (Task 1e) handles scaling each one independently.
FEATURE_COLUMNS = [PRICE_VALUE]

# --- Scale the feature columns using scale_features() (Task 1e) ---
# This replaces the old inline MinMaxScaler code with our new function.
# scaled_data: 2D numpy array with shape (n_samples, n_features), values in [0,1]
# scaler_dict: Dictionary mapping each feature name to its fitted MinMaxScaler.
#              e.g., {'Close': MinMaxScaler(...)}
#              We can later use scaler_dict['Close'].inverse_transform(predictions)
#              to convert predicted values back to actual stock prices.
# save_scalers=True: Persists the scaler dictionary to 'scalers/scalers.pkl'
#                    so we can reload it in future sessions without re-fitting.
scaled_data, scaler_dict = scale_features(
    data, FEATURE_COLUMNS, scale_range=(0, 1),
    save_scalers=True, scaler_file='scalers/scalers.pkl'
)

# For backward compatibility, extract the scaler for the primary price column.
# This 'scaler' variable is used later for inverse_transform on predictions.
scaler = scaler_dict[PRICE_VALUE]

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60  # Original

# To store the training data
x_train = []
y_train = []

# --- Prepare training sequences ---
# If we have multiple features, scaled_data has shape (n_samples, n_features).
# For the current single-feature case, we extract column index 0 (the 'Close' price).
# The target (y) is always the primary PRICE_VALUE column (index 0 in FEATURE_COLUMNS).
price_col_index = FEATURE_COLUMNS.index(PRICE_VALUE)
# Extract the 1D array of scaled prices for the target column
scaled_price = scaled_data[:, price_col_index]

# Create sliding window sequences for training:
# For each time step t (starting from PREDICTION_DAYS), we create:
#   x_train sample: the previous PREDICTION_DAYS values of scaled features
#   y_train sample: the scaled price at time t (what we want to predict)
for x in range(PREDICTION_DAYS, len(scaled_data)):
    # Each x_train sample is a window of PREDICTION_DAYS rows from scaled_data.
    # If using multiple features, each row contains all feature values.
    x_train.append(scaled_data[x - PREDICTION_DAYS:x])
    # The target is always the scaled price at the current time step.
    y_train.append(scaled_price[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 3D array(p, q, f) where:
#   p = len(scaled_data) - PREDICTION_DAYS (number of samples)
#   q = PREDICTION_DAYS (number of time steps per sample)
#   f = len(FEATURE_COLUMNS) (number of features per time step)
# y_train is a 1D array(p) containing the target price for each sample.

# Note: When using a single feature, x_train already has shape (p, q, 1)
# after np.array() conversion since each window row is [value].
# For safety, we ensure the correct 3D shape for the LSTM input.
if len(x_train.shape) == 2:
    # If only 1 feature, reshape from (p, q) to (p, q, 1)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We now have x_train as a 3D array(p, q, f); the LSTM layer expects
# input shape (batch_size, timesteps, features)

#------------------------------------------------------------------------------
# Build the Model
# TO DO (v4 - implemented):
# 1) Check if model has been saved before.
#    If so, load the saved model
#    If not, train and save the model
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
model_file = os.path.join(MODEL_DIR, f'{COMPANY}_model.keras')

# Determine the number of features for the LSTM input layer.
# This adapts automatically whether we use 1 feature or multiple features.
n_features = len(FEATURE_COLUMNS)

if os.path.exists(model_file):
    print(f"Loading saved model from {model_file}...")
    model = load_model(model_file)
else:
    print("Building and training new model...")
    model = Sequential()  # Basic neural network
    # See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
    # for some useful examples

    # InputLayer shape is (PREDICTION_DAYS, n_features).
    # Using n_features instead of hardcoded 1 so the model adapts
    # when we add more feature columns (e.g., Open, High, Low, Volume).
    model.add(InputLayer(shape=(x_train.shape[1], n_features)))
    model.add(LSTM(units=50, return_sequences=True))
    # This is our first hidden layer. Using InputLayer separately is more
    # compatible with newer versions of Keras/TensorFlow.
    # For some advanced explanation of return_sequences:
    # https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
    # https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
    # As explained there, for a stacked LSTM, you must set return_sequences=True
    # when stacking LSTM layers so that the next LSTM layer has a
    # three-dimensional sequence input.

    # Finally, units specifies the number of nodes in this layer.
    # This is one of the parameters you want to play with to see what number
    # of units will give you better prediction quality (for your problem)

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

    # We compile the model by specify the parameters for the model
    # See lecture Week 6 (COS30018)
    model.compile(optimizer='adam', loss='mean_squared_error')
    # The optimizer and loss are two important parameters when building an
    # ANN model. Choosing a different optimizer/loss can affect the prediction
    # quality significantly. You should try other settings to learn; e.g.

    # optimizer='rmsprop'/'sgd'/'adadelta'/...
    # loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

    # Now we are going to train this model with our training data
    # (x_train, y_train)
    model.fit(x_train, y_train, epochs=25, batch_size=32)
    # Other parameters to consider: How many rounds(epochs) are we going to
    # train our model? Typically, the more the better, but be careful about
    # overfitting!
    # What about batch_size? Well, again, please refer to
    # Lecture Week 6 (COS30018): If you update your model for each and every
    # input sample, then there are potentially 2 issues: 1. If your training
    # data is very big (billions of input samples) then it will take VERY long;
    # 2. Each and every input can immediately makes changes to your model
    # (a source of overfitting). Thus, we do this in batches: We'll look at
    # the aggregated errors/losses from a batch of, say, 32 input samples
    # and update our model based on this aggregated loss.

    # Save the model for future use
    model.save(model_file)
    print(f"Model saved to {model_file}")

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
# Load the test data
TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

# --- Load test data using load_data() (Task 1d) ---
# Reusing the same load_data() function ensures consistent behavior:
# caching, NaN handling, and MultiIndex flattening are all handled.
test_data = load_data(COMPANY, TEST_START, TEST_END,
                      save_local=True, data_dir='data', nan_strategy='ffill')

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

#------------------------------------------------------------------------------
# Make predictions on test data
#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Plot the test predictions
# TO DO:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------
real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??