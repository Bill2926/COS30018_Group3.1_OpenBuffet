# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3); 30/05/2026 (v4 - Task C.1); [Current Date] (v5 - Task C.2)

# Code modified from:
# Title: Predicting Stock Prices with Python
# YouTube link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# uv add numpy matplotlib pandas tensorflow keras scikit-learn yfinance
# or: pip install numpy matplotlib pandas tensorflow keras scikit-learn yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM, Input

# ==============================================================================
# ★ TASK C.2 IMPROVEMENTS ★
# Part 1a: Flexible data loading with date ranges
# Part 1b: Systematic NaN handling
# Part 1c: Multiple train/test splitting methods
# ==============================================================================

# ==============================================================================
# PART 1a: LOAD DATA WITH FLEXIBLE DATE RANGE
# ==============================================================================
#
# ★ IMPROVEMENT FROM v0.1:
# v0.1 had hardcoded dates:
#   TRAIN_START = '2020-01-01'
#   TRAIN_END = '2023-08-01'
#   (Had to edit code to change dates - not flexible)
#
# Task 2 creates a function that accepts any date range:
#   data = load_data_with_dates('CBA.AX', '2020-01-01', '2023-08-01')
#   (Flexible - can change dates without editing code)
#
# ★ WHY THIS IS BETTER:
#   - More reusable code
#   - Easy to test different date ranges
#   - Better for different stocks/time periods
#   - Validates dates before use
#

def load_data_with_dates(company, start_date, end_date, data_dir='data', features=None, force_download=False):
    """
    # 1a. LOAD AND PROCESS DATASET WITH DATE RANGE
    #
    # Purpose: Download stock data for specified company and date range
    #          Caches data locally to avoid redundant downloads
    #
    # Parameters:
    #   company (str): Stock ticker (e.g., 'CBA.AX')
    #   start_date (str): Start date in format 'YYYY-MM-DD'
    #   end_date (str): End date in format 'YYYY-MM-DD'
    #   data_dir (str): Folder to store/load CSV files
    #   features (list): Which columns to keep (e.g., ['Close', 'Volume', 'Open'])
    #   force_download (bool): If True, always download fresh data (ignore cache)
    #
    # Returns:
    #   pd.DataFrame: Stock data with dates as index
    #
    # Changes from v0.1:
    #   v0.1: Hardcoded dates directly in script
    #   Task 2: Function with flexible date parameters
    """
    
    # Step 1: VALIDATE DATE FORMAT
    # WHY: Prevent errors from invalid dates like '2023-13-01' or '2023-01-01' > '2022-12-31'
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except:
        raise ValueError(f"Invalid date format. Must be 'YYYY-MM-DD'. Got: {start_date}, {end_date}")
    
    # Check if start < end
    if start_dt >= end_dt:
        raise ValueError(f"start_date must be before end_date. Got: {start_date} >= {end_date}")
    
    # Step 2: CREATE DATA DIRECTORY
    # WHY: os.makedirs(path, exist_ok=True) safely creates directories
    #      exist_ok=True prevents error if directory already exists
    os.makedirs(data_dir, exist_ok=True)
    
    # Step 3: GENERATE CACHE FILENAME
    # WHY: Easy naming convention makes it easy to identify and manage cached files
    #      Format: data/CBA.AX_2020-01-01_2023-08-01.csv
    data_file = os.path.join(data_dir, f'{company}_{start_date}_{end_date}.csv')
    
    # Step 4: CHECK IF DATA ALREADY CACHED
    # WHY: Save time and bandwidth - downloading takes 5-10 seconds,
    #      loading from CSV takes <1 second
    if os.path.exists(data_file) and not force_download:
        print(f"✓ Loading cached data from: {data_file}")
        # parse_dates=True: Convert date column to datetime objects
        # index_col=0: Use first column (dates) as index
        # WHY: Important for time-series operations
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
    else:
        print(f"↓ Downloading {company} from {start_date} to {end_date}...")
        try:
            import yfinance as yf
            # yf.download() fetches historical data from Yahoo Finance API
            # auto_adjust=True: Adjusts prices for stock splits and dividends
            # WHY: Gives us "adjusted close" price (more accurate for analysis)
            data = yf.download(company, start_date, end_date, auto_adjust=True, progress=False)
        except Exception as e:
            raise ConnectionError(f"Failed to download {company}: {str(e)}")
        
        # Step 5: HANDLE MULTIINDEX COLUMNS FROM YFINANCE 0.2+
        # WHY: yfinance changed column format in version 0.2
        #      v0.1 returns: columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        #      v0.2+ returns: MultiIndex = [('CBA.AX', 'Open'), ('CBA.AX', 'High'), ...]
        #      We need to flatten to simple column names
        #      Without this, columns become tuples (ticker, feature) instead of just feature names
        if isinstance(data.columns, pd.MultiIndex):
            # get_level_values(0) extracts level 0 of MultiIndex = feature names
            data.columns = data.columns.get_level_values(0)
        
        # Save to CSV for future use
        # WHY: Avoid re-downloading same data next time
        data.to_csv(data_file)
        print(f"✓ Data saved to: {data_file}")
    
    # Step 6: FILTER SPECIFIC FEATURES IF REQUESTED
    # WHY: User might only want ['Close', 'Volume']
    #      Reduces memory usage and processing time
    #      v0.1 used only Close, Task 2 supports multiple features
    if features is not None:
        available_features = [f for f in features if f in data.columns]
        missing = [f for f in features if f not in data.columns]
        if missing:
            print(f"⚠ Warning: These features not in data: {missing}")
        data = data[available_features]
    
    print(f"✓ Data loaded: {data.shape} ({len(data)} rows, {len(data.columns)} columns)")
    return data


# ==============================================================================
# PART 1b: HANDLE NaN VALUES
# ==============================================================================
#
# ★ IMPROVEMENT FROM v0.1:
# v0.1 had NO NaN handling at all
#   (Assumed data was always clean - unrealistic!)
#
# Task 2 provides systematic NaN handling with 5 methods:
#   - 'drop': Remove rows with NaN
#   - 'forward_fill': Use previous value (BEST for stock data)
#   - 'backward_fill': Use next value
#   - 'interpolate': Linear estimation between points
#   - 'mean': Fill with average
#
# ★ WHY THIS IS BETTER:
#   - Real data has gaps (weekends, holidays, market closures)
#   - ML models cannot process NaN (causes errors)
#   - Provides multiple strategies for different scenarios
#   - Shows statistics on how much data is missing
#

def handle_nan_values_part1b(data, method='forward_fill', threshold=0.5):
    """
    # 1b. DEAL WITH NaN ISSUE IN THE DATA
    #
    # Purpose: Handle missing values (NaN) using different strategies
    #          NaN = Not a Number (missing data point)
    #
    # Parameters:
    #   data (pd.DataFrame): Input data that might have NaN values
    #   method (str): Strategy to handle NaN:
    #     'drop': Remove rows containing NaN
    #     'forward_fill': Fill with previous value (recommended for stock)
    #     'backward_fill': Fill with next value
    #     'interpolate': Estimate between surrounding values
    #     'mean': Fill with column average
    #   threshold (float): Warn if NaN% > threshold (0.5 = 50%)
    #
    # Returns:
    #   pd.DataFrame: Data with NaN handled
    #
    # WHY NaN HAPPENS IN STOCK DATA:
    #   - Market holidays (no trading, no data)
    #   - Weekends (markets closed)
    #   - System outages or data errors
    #   - Newly listed stocks (no historical data)
    #
    # WHY WE MUST HANDLE IT:
    #   - ML models crash or produce errors with NaN
    #   - Cannot train neural networks with missing values
    #   - v0.1 ignored this - unrealistic for real data
    """
    
    # Step 1: ANALYZE NaN SITUATION
    # WHY: Understand scope of problem before fixing it
    data_cleaned = data.copy()  # Don't modify original
    nan_count = data_cleaned.isna().sum()
    total_values = len(data_cleaned) * len(data_cleaned.columns)
    nan_percentage = (nan_count.sum() / total_values) * 100
    
    print(f"\n[1b] NaN Analysis:")
    print(f"     Total NaN values: {nan_count.sum()}")
    print(f"     Percentage: {nan_percentage:.2f}%")
    if nan_count.sum() > 0:
        print(f"     NaN per column: {dict(nan_count[nan_count > 0])}")
    
    # Step 2: WARN IF TOO MUCH MISSING DATA
    # WHY: If >50% missing, any imputation becomes unreliable
    if nan_percentage > threshold * 100:
        print(f"⚠ WARNING: NaN exceeds {threshold*100:.0f}% threshold!")
    
    # Step 3: CHOOSE AND APPLY METHOD
    print(f"     Using '{method}' method to handle NaN...")
    
    if method == 'drop':
        # Remove any row containing NaN
        # ADVANTAGE: Simple, no assumptions
        # DISADVANTAGE: Breaks time-series continuity
        #              Example: If gaps on weekends, removes Fridays & Mondays
        data_cleaned = data_cleaned.dropna()
        print(f"     ✓ Rows remaining: {len(data_cleaned)}")
    
    elif method == 'forward_fill':
        # Forward fill (ffill): Fill with previous value
        # ADVANTAGE: Realistic for stock prices
        #           Assumption: No trading day = price doesn't change
        #           Common/standard method for stock data
        # DISADVANTAGE: Unrealistic for long gaps (e.g., month-long closure)
        # EXAMPLE:
        #   Before: [10.5, 10.7, NaN, NaN, 11.2]
        #   After:  [10.5, 10.7, 10.7, 10.7, 11.2]
        #   Why: Friday=10.7, Sat(NaN)→10.7, Sun(NaN)→10.7, Mon=11.2
        data_cleaned = data_cleaned.fillna(method='ffill', limit=None)
        print(f"     ✓ Remaining NaN: {data_cleaned.isna().sum().sum()}")
    
    elif method == 'backward_fill':
        # Backward fill (bfill): Fill with next value
        # ADVANTAGE: Can fill from beginning of series
        # DISADVANTAGE: Uses "future" information (data leakage risk)
        #              Less common than forward_fill
        data_cleaned = data_cleaned.fillna(method='bfill', limit=None)
        print(f"     ✓ Remaining NaN: {data_cleaned.isna().sum().sum()}")
    
    elif method == 'interpolate':
        # Linear interpolation: Estimate between surrounding values
        # ADVANTAGE: Mathematically smooth, no sudden jumps
        # DISADVANTAGE: Creates artificial prices
        #              Stock prices don't follow smooth curves
        # EXAMPLE:
        #   Before: [10.0, NaN, NaN, 14.0]
        #   After:  [10.0, 11.33, 12.67, 14.0]  # Linear spacing
        data_cleaned = data_cleaned.interpolate(method='linear')
        print(f"     ✓ Remaining NaN: {data_cleaned.isna().sum().sum()}")
    
    elif method == 'mean':
        # Fill each column with its average
        # ADVANTAGE: Very simple
        # DISADVANTAGE: Completely ignores temporal relationships
        #              All NaN get same value = very unrealistic
        # ✗ NOT RECOMMENDED for stock data
        for col in data_cleaned.columns:
            mean_val = data_cleaned[col].mean()
            data_cleaned[col].fillna(mean_val, inplace=True)
        print(f"     ✓ Remaining NaN: {data_cleaned.isna().sum().sum()}")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Step 4: SAFETY CHECK
    # WHY: Ensure absolutely no NaN remains
    remaining_nan = data_cleaned.isna().sum().sum()
    if remaining_nan > 0:
        print(f"⚠ {remaining_nan} NaN still remain, applying forward_fill backup...")
        data_cleaned = data_cleaned.fillna(method='ffill')
        data_cleaned = data_cleaned.fillna(method='bfill')  # For first rows if any
    
    print(f"✓ [1b] NaN handling complete. Final shape: {data_cleaned.shape}")
    return data_cleaned


# ==============================================================================
# PART 1c: SPLIT DATA INTO TRAIN/TEST
# ==============================================================================
#
# ★ IMPROVEMENT FROM v0.1:
# v0.1 used only hardcoded date splitting:
#   TEST_START = '2023-08-02'
#   TEST_END = '2024-07-02'
#   (Only one way to split, not flexible)
#
# Task 2 provides 3 splitting methods:
#   - 'ratio': 80/20 split (randomly shuffles)
#   - 'date': Date-based split (BEST for time-series)
#   - 'random': Random subset selection
#
# ★ WHY THIS IS BETTER:
#   - Explains tradeoffs between methods
#   - DATE-BASED is correct for stock prediction (no data leakage)
#   - Shows why RATIO is wrong for time-series
#   - Flexible parameters
#

def split_data_part1c(data, split_method='date', split_param='2023-01-01', random_state=None):
    """
    # 1c. SPLIT DATA INTO TRAIN/TEST SETS
    #
    # Purpose: Divide data into training and testing sets
    #          Different methods suit different purposes
    #
    # Parameters:
    #   data (pd.DataFrame): Data to split
    #   split_method (str): How to split:
    #     'ratio': Percentage split (e.g., 80/20 randomly)
    #     'date': Date-based split (RECOMMENDED for stock)
    #     'random': Random subset selection
    #   split_param: Depends on method:
    #     ratio method: float 0.8 = 80% train
    #     date method: string '2023-01-01' = before this = train
    #     random method: float 0.7 = 70% train
    #   random_state: Seed for reproducibility
    #
    # Returns:
    #   (train_df, test_df, metadata_dict)
    #
    # WHY SPLITTING METHOD MATTERS FOR STOCK DATA:
    #   - Stock prices are TIME-DEPENDENT (today affects tomorrow)
    #   - Random shuffle breaks temporal relationships
    #   - Date-based split preserves temporal order (realistic)
    #   - Test set should be FUTURE data (true forecasting test)
    """
    
    split_info = {'method': split_method, 'total_size': len(data)}
    print(f"\n[1c] Splitting data using '{split_method}' method...")
    
    if split_method == 'ratio':
        # RATIO-BASED SPLIT
        # Randomly shuffles ALL data, then splits by percentage
        #
        # PROCESS:
        #   Original: [2020, 2021, 2022, 2023, 2024]
        #   Shuffled: [2022, 2024, 2020, 2023, 2021]
        #   Train (80%): [2022, 2024, 2020]
        #   Test (20%): [2023, 2021]
        #
        # PROBLEM: ✗ Data leakage! Model trained on 2022-2024 data,
        #            then tested on 2023 (which was in training!)
        #            This gives UNREALISTIC results.
        #
        # ✗ NOT RECOMMENDED for stock prediction
        
        if not isinstance(split_param, float) or not (0 < split_param < 1):
            raise ValueError(f"For ratio, split_param must be 0-1. Got: {split_param}")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # np.random.permutation(n) creates random reordering of 0 to n-1
        # Example: permutation(5) = [3, 0, 4, 1, 2] (shuffled order)
        indices = np.random.permutation(len(data))
        split_idx = int(len(data) * split_param)
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        # iloc[indices] selects rows by integer position
        train_data = data.iloc[train_indices].sort_index()  # Restore time order
        test_data = data.iloc[test_indices].sort_index()
        
        split_info['train_size'] = len(train_data)
        split_info['test_size'] = len(test_data)
        print(f"     Train: {len(train_data)} ({split_param*100:.0f}%)")
        print(f"     Test:  {len(test_data)} ({(1-split_param)*100:.0f}%)")
        print(f"     ⚠ WARNING: Ratio split NOT ideal for time-series!")
    
    elif split_method == 'date':
        # DATE-BASED SPLIT (★ RECOMMENDED FOR STOCK PREDICTION ★)
        # Splits at specific date - everything before = train, after = test
        #
        # PROCESS:
        #   Train: [2020, 2021, 2022, 2023] (before cutoff)
        #   Test:  [2024]                   (after cutoff)
        #
        # ADVANTAGE: ★ PRESERVES TEMPORAL ORDER
        #           ★ NO DATA LEAKAGE (test = truly future)
        #           ★ REALISTIC (simulates real forecasting)
        #           This is the CORRECT way to evaluate time-series models
        #
        # EXAMPLE SCENARIO:
        #   "Given all data up to 2023, can I predict 2024?"
        #   This is what we want to test!
        
        try:
            cutoff = pd.Timestamp(split_param)
        except:
            raise ValueError(f"For date, split_param must be valid date. Got: {split_param}")
        
        # Boolean indexing: data[condition] selects rows where condition=True
        # data.index < cutoff: Creates True/False array by comparing dates
        train_data = data[data.index < cutoff]
        test_data = data[data.index >= cutoff]
        
        split_info['train_range'] = (data.index.min(), train_data.index.max())
        split_info['test_range'] = (test_data.index.min(), test_data.index.max())
        split_info['train_size'] = len(train_data)
        split_info['test_size'] = len(test_data)
        
        print(f"     Cutoff date: {cutoff.date()}")
        print(f"     Train: {len(train_data)} ({train_data.index.min().date()} to {train_data.index.max().date()})")
        print(f"     Test:  {len(test_data)} ({test_data.index.min().date()} to {test_data.index.max().date()})")
        print(f"     ✓ BEST METHOD FOR STOCK PREDICTION!")
    
    elif split_method == 'random':
        # RANDOM SPLIT (without full shuffle)
        # Randomly selects which rows go to test (doesn't shuffle all)
        #
        # ADVANTAGE: Random but preserves some temporal structure
        # DISADVANTAGE: Still mixes temporal order (not ideal for time-series)
        
        if not isinstance(split_param, float) or not (0 < split_param < 1):
            raise ValueError(f"For random, split_param must be 0-1. Got: {split_param}")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        test_size = int(len(data) * (1 - split_param))
        # np.random.choice(n, size, replace=False) picks 'size' random integers 0 to n-1
        # replace=False = no duplicates (each index at most once)
        test_indices = np.random.choice(len(data), size=test_size, replace=False)
        
        train_mask = np.ones(len(data), dtype=bool)
        train_mask[test_indices] = False
        
        train_data = data[train_mask]
        test_data = data[~train_mask]
        
        split_info['train_size'] = len(train_data)
        split_info['test_size'] = len(test_data)
        print(f"     Train: {len(train_data)} ({split_param*100:.0f}%)")
        print(f"     Test:  {len(test_data)} ({(1-split_param)*100:.0f}%)")
        print(f"     ~ Compromise between ratio and date methods")
    
    else:
        raise ValueError(f"Unknown method: {split_method}")
    
    # Validate split
    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError("Split resulted in empty train or test set!")
    
    print(f"✓ [1c] Split complete!")
    return train_data, test_data, split_info


# ==============================================================================
# ORIGINAL v0.1 CODE - Now using improved Task 2 functions
# ==============================================================================

print("\n" + "="*80)
print("STOCK PRICE PREDICTION - Task C.2 Implementation")
print("="*80)

#------------------------------------------------------------------------------
# Load Data - NOW USING PART 1a
#------------------------------------------------------------------------------
print("\n[PART 1a] LOAD DATA")
print("-" * 80)

COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-08-01'       # End date to read

DATA_DIR = 'data'

# ★ TASK 2 CHANGE: Use function instead of hardcoded date loading
# 1a. Load data with flexible date range
data = load_data_with_dates(
    company=COMPANY,
    start_date=TRAIN_START,
    end_date=TRAIN_END,
    data_dir=DATA_DIR,
    features=['Open', 'High', 'Low', 'Close', 'Volume'],  # Multiple features!
    force_download=False  # Use cache if available
)

#------------------------------------------------------------------------------
# Handle NaN - NOW USING PART 1b
#------------------------------------------------------------------------------
print("\n[PART 1b] HANDLE NaN VALUES")
print("-" * 80)

# ★ TASK 2 CHANGE: Systematic NaN handling
data = handle_nan_values_part1b(
    data=data,
    method='forward_fill',  # Best for stock data
    threshold=0.5
)

#------------------------------------------------------------------------------
# Split Data - NOW USING PART 1c
#------------------------------------------------------------------------------
print("\n[PART 1c] SPLIT DATA")
print("-" * 80)

# ★ TASK 2 CHANGE: Use date-based split (best for time-series)
# Split at 2023-01-01: everything before = train, after = test
train_data_full, test_data_full, split_info = split_data_part1c(
    data=data,
    split_method='date',  # Date-based split (BEST for stock)
    split_param='2023-01-01'
)

#------------------------------------------------------------------------------
# Prepare Data - Using Close price for model
#------------------------------------------------------------------------------
print("\n[PREPARE DATA] Scaling and sequence creation")
print("-" * 80)

PRICE_VALUE = "Close"

# Scale training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(train_data_full[PRICE_VALUE].values.reshape(-1, 1))
# Explanation of reshape(-1, 1):
# - data.values = 1D array [10.5, 10.7, 10.8, ...]
# - reshape(-1, 1) = convert to 2D array [[10.5], [10.7], [10.8], ...]
# - WHY: sklearn's fit_transform() requires 2D input
# - -1 means "auto-calculate this dimension to keep same # of elements"

# Number of days to look back to base the prediction
PREDICTION_DAYS = 60  # Original

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

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# We reshape x_train into a 3D array(p, q, 1)
# LSTM requires input shape (batch, timesteps, features)

#------------------------------------------------------------------------------
# Build the Model
#------------------------------------------------------------------------------
print("\n[BUILD MODEL]")
print("-" * 80)

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
model_file = os.path.join(MODEL_DIR, f'{COMPANY}_model.keras')

if os.path.exists(model_file):
    print(f"Loading saved model from {model_file}...")
    model = load_model(model_file)
else:
    print("Building and training new model...")
    model = Sequential()

    model.add(Input(shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    model.save(model_file)
    print(f"Model saved to {model_file}")

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
print("\n[TEST MODEL]")
print("-" * 80)

TEST_START = '2023-08-02'
TEST_END = '2024-07-02'

test_data_file = os.path.join(DATA_DIR, f'{COMPANY}_{TEST_START}_{TEST_END}.csv')

if os.path.exists(test_data_file):
    print(f"Loading saved test data from {test_data_file}...")
    test_data = pd.read_csv(test_data_file, index_col=0, parse_dates=True)
else:
    print(f"Downloading test data for {COMPANY}...")
    import yfinance as yf
    test_data = yf.download(COMPANY, TEST_START, TEST_END, auto_adjust=True, progress=False)
    if isinstance(test_data.columns, pd.MultiIndex):
        test_data.columns = test_data.columns.get_level_values(0)
    test_data.to_csv(test_data_file)
    print(f"Test data saved to {test_data_file}")

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((train_data_full[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values

model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

#------------------------------------------------------------------------------
# Plot the test predictions
#------------------------------------------------------------------------------
print("\n[PLOT RESULTS]")
print("-" * 80)

plt.figure(figsize=(12, 6))
plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price", linewidth=2)
plt.plot(predicted_prices, color="green", label=f"Predicted {COMPANY} Price", linewidth=2)
plt.title(f"{COMPANY} Share Price - Task C.2 Implementation")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------
print("\n[NEXT DAY PREDICTION]")
print("-" * 80)

real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(f"Next day prediction: ${prediction[0][0]:.2f}")

print("\n" + "="*80)
print("✓ TASK C.2 COMPLETE!")
print("="*80)