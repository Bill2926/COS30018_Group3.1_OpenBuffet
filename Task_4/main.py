# File: main.py
# Authors: Bao Vo, Cheong Koo (original);\
# Version: v0.3 - Task C4
#
# This is the the Entry Point.
# All logic lives in the four modules:
#   data_loader.py   – download / cache / split / scale stock data (Task C.2)
#   visualiser.py    – candlestick & boxplot charts, prediction plot (Task C.3)
#   model_builder.py – build_model() + save/load helpers              (Task C.4)
#   predictor.py     – scale test data, slide windows, inverse-transform

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from data_loader   import load_and_process_dataset, prepare_lstm_data
from visualiser    import (plot_candlestick_chart, plot_boxplot_chart,
                           plot_predictions, ask_int, ask_date, slice_range)
from model_builder import build_model, get_model_path, load_or_build
from predictor     import build_test_inputs, make_predictions, predict_next_day


COMPANY     = 'NVDA'
TRAIN_START = '2020-01-01'
TEST_START  = '2023-08-02'
TEST_END    = '2024-07-02'

DATA_DIR    = 'data'
MODEL_DIR   = 'models'

PRICE_VALUE     = 'Close'
PREDICTION_DAYS = 60        # look-back window fed to the RNN

# ==============================
# Task C.4: Model Architecture Hyperparameters
# Add/remove elements in LAYER_SIZES to change the number and width of layers.
# ==============================
DL_NETWORK   = 'GRU'      # (LSTM, GRU, RNN)
LAYER_SIZES  = [64, 32]   # N number of recurrent layers, must always a Python List []
DROPOUT_RATE = 0.2

# Training hyperparameters
EPOCHS     = 20
BATCH_SIZE = 32

# Retrain even when the model existed
FORCE_RETRAIN = True


# ==============================
# Step 1 – Load and split data  (Task C.2)
# ==============================
train_data, test_data, _ = load_and_process_dataset(
    company      = COMPANY,
    start_date   = TRAIN_START,
    end_date     = TEST_END,
    features     = None,            # keep all OHLCV columns for charting
    data_dir     = DATA_DIR,
    force_download = False,
    nan_method   = 'forward_fill',
    split_method = 'date',
    split_param  = TEST_START,
    scale_columns = False           # we scale manually below (Close only)
)


# ==============================
# Step 2 – Visualise the training data  (Task C.3)
# ==============================
print("\n─── Task C.3: chart options (press Enter to accept the [default]) ───")
n_days = ask_int("Days per candle / box window (n) [1]: ", default=1)
start  = ask_date("Start date  – year / year-month / date [all data]: ", is_end=False)
end    = ask_date("End date    – year / year-month / date [all data]: ", is_end=True)

view = slice_range(train_data.copy(), start, end)

if len(view) == 0:
    print(f"No trading days in that range — nothing to plot. "
          f"Data spans {train_data.index[0].date()} → {train_data.index[-1].date()}.")
else:
    print(f"Plotting {len(view)} trading days "
          f"({view.index[0].date()} → {view.index[-1].date()}).")

    plot_candlestick_chart(view, n_days=n_days,
                           title=f'{COMPANY} {n_days}-Day Candles', show=False)

    if n_days >= 2 and len(view) >= n_days:
        plot_boxplot_chart(view, column='Close', n_days=n_days,
                           title=f'{COMPANY} Close Prices', show=False)
    else:
        print("Skipping boxplot (need n >= 2 and at least n days in the range).")


# ==============================
# Step 3 – Scale Close prices and build LSTM input sequences
# ==============================
# Fit scaler on training data ONLY
scaler      = MinMaxScaler(feature_range=(0, 1))
train_close = train_data[PRICE_VALUE].values.reshape(-1, 1)
scaled_train = scaler.fit_transform(train_close).flatten()

x_train, y_train = prepare_lstm_data(scaled_train, PREDICTION_DAYS)

# ==============================
# Step 4 – Build / load model  (Task C.4)
#
# build_model() is the required Task C.4 function.  It accepts:
#   sequence_length – so it can declare the correct Input shape
#   layer_sizes     – list whose length = number of recurrent layers and whose values = units per layer
#   layer_type      – 'LSTM', 'GRU', or 'RNN'
#   dropout_rate    – regularisation between recurrent layers
#   dense_units     – width of the final Dense output layer
#   optimizer/loss  – compilation options
# ==============================
model_path = get_model_path(MODEL_DIR, COMPANY,
                            tag=f'{DL_NETWORK}_{"x".join(map(str, LAYER_SIZES))}')

build_kwargs = dict(
    sequence_length = PREDICTION_DAYS,
    layer_sizes     = LAYER_SIZES,
    layer_type      = DL_NETWORK,
    dropout_rate    = DROPOUT_RATE,
    dense_units     = 1,
    optimizer       = 'adam',
    loss            = 'mean_squared_error',
)

model = load_or_build(
    model_path   = model_path,
    build_kwargs = build_kwargs,
    x_train      = x_train,
    y_train      = y_train,
    epochs       = EPOCHS,
    batch_size   = BATCH_SIZE,
    force_retrain = FORCE_RETRAIN,
)


# ==============================
# Step 5 – Evaluate on the test set
# ==============================
actual_prices = test_data[PRICE_VALUE].values

# Build the scaled input array that covers [end-of-training … end-of-test].
model_inputs = build_test_inputs(
    train_prices    = train_data[PRICE_VALUE],
    test_prices     = test_data[PRICE_VALUE],
    scaler          = scaler,
    prediction_days = PREDICTION_DAYS,
)

# Slide the look-back window across the test period.
x_test = make_predictions(model, model_inputs, PREDICTION_DAYS)

# Run inference and invert the scaling to get prices in USD.
predicted_scaled = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_scaled)

plot_predictions(actual_prices, predicted_prices, COMPANY)


# ==============================
# Step 6 – Forecast next trading day
# ==============================
next_day_price = predict_next_day(model, model_inputs, PREDICTION_DAYS, scaler)
print(f"\n[main] Next-day predicted closing price for {COMPANY}: ${next_day_price:.2f}")


# ==============================
# Task C.4 Experiment block
# ==============================
# Uncomment any experiment below to compare architectures.
# Each call to build_model() demonstrates that the function works for any
# combination of DL_NETWORK × layer_sizes × hyperparameters.

# ── Experiment A: GRU, same depth as baseline ────────────────────────────
# gru_kwargs = {**build_kwargs, 'DL_NETWORK': 'GRU'}
# gru_path   = get_model_path(MODEL_DIR, COMPANY, tag='GRU_128x64x32')
# gru_model  = load_or_build(gru_path, gru_kwargs, x_train, y_train,
#                            epochs=25, batch_size=32, force_retrain=False)

# ── Experiment B: SimpleRNN, shallow ────────────────────────────────────
# rnn_kwargs = {**build_kwargs, 'DL_NETWORK': 'RNN', 'layer_sizes': [64, 32]}
# rnn_path   = get_model_path(MODEL_DIR, COMPANY, tag='RNN_64x32')
# rnn_model  = load_or_build(rnn_path, rnn_kwargs, x_train, y_train,
#                            epochs=25, batch_size=32, force_retrain=False)

# ── Experiment C: LSTM, wider layers, higher dropout ─────────────────────
# big_kwargs = {**build_kwargs, 'layer_sizes': [256, 128, 64], 'dropout_rate': 0.3}
# big_path   = get_model_path(MODEL_DIR, COMPANY, tag='LSTM_256x128x64')
# big_model  = load_or_build(big_path, big_kwargs, x_train, y_train,
#                            epochs=50, batch_size=16, force_retrain=False)
