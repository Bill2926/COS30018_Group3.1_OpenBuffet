# File: main.py
# Authors: Bao Vo, Cheong Koo (original);\
# Version: v0.4 - Task C5
#
# This is the the Entry Point.
# All logic lives in the four modules:
#   data_loader.py   – download / cache / split / scale stock data (Task C.2)
#   visualiser.py    – candlestick & boxplot charts, prediction plot (Task C.3)
#   model_builder.py – build_model() + save/load helpers              (Task C.4)
#   predictor.py     – scale test data, slide windows, inverse-transform

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from data_loader   import load_and_process_dataset, prepare_lstm_data, prepare_sequences
from visualiser    import (plot_candlestick_chart, plot_boxplot_chart,
                           plot_predictions, plot_multistep_forecast,
                           ask_int, ask_date, slice_range)
from model_builder import build_model, get_model_path, load_or_build
from predictor     import (build_test_inputs, make_predictions, predict_next_day,
                           build_test_inputs_mv, make_windows, inverse_close,
                           forecast_future)


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

# Retrain even when a saved model exists.
# False -> reuse cached models (fast); each model caches per config via its tag,
# so changing k or the architecture trains a fresh one automatically.
FORCE_RETRAIN = False


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
# Task C.5 — Machine Learning 2: multivariate & multistep prediction
#
# Three scenarios, each built from the SAME generalised functions:
#   prepare_sequences()  – windows with n_features inputs and k_steps targets
#   build_model()        – n_features sets the Input width, dense_units = k
#   forecast_future()    – k future closes from one window (inverse-scaled)
# ==============================
FEATURES   = ['Open', 'High', 'Low', 'Close', 'Volume']   # multivariate inputs
TARGET_COL = 'Close'                                       # what we predict

print("\n─── Task C.5: multistep / multivariate prediction ───")
K_STEPS = ask_int("Forecast horizon k (days into the future) [5]: ", default=5)

# Re-load the SAME data scaled per-column (the CSV cache makes this near-free).
# Returns scaled train/test frames + a {column: fitted MinMaxScaler} dict; each
# scaler is fitted on training data only, so there is no test-set leakage.
train_scaled, test_scaled, scalers = load_and_process_dataset(
    company      = COMPANY,
    start_date   = TRAIN_START,
    end_date     = TEST_END,
    features     = FEATURES,
    data_dir     = DATA_DIR,
    force_download = False,
    nan_method   = 'forward_fill',
    split_method = 'date',
    split_param  = TEST_START,
    scale_columns = True,
)
target_idx     = FEATURES.index(TARGET_COL)
close_scaler   = scalers[TARGET_COL]
train_features = train_scaled[FEATURES].values        # scaled (rows, n_features)
train_close    = train_scaled[[TARGET_COL]].values    # scaled (rows, 1)
recent_close   = train_data[PRICE_VALUE].values[-PREDICTION_DAYS:]   # raw, for plots


# ── [C.5-1] Multistep: univariate Close → next k closes ──────────────────────
X1, y1 = prepare_sequences(train_close, target_idx=0,
                           prediction_days=PREDICTION_DAYS, k_steps=K_STEPS)
model_multistep = load_or_build(
    model_path   = get_model_path(MODEL_DIR, COMPANY, tag=f'multistep_k{K_STEPS}'),
    build_kwargs = dict(sequence_length=PREDICTION_DAYS, layer_sizes=LAYER_SIZES,
                        layer_type=DL_NETWORK, dropout_rate=DROPOUT_RATE,
                        dense_units=K_STEPS, n_features=1),
    x_train=X1, y_train=y1, epochs=EPOCHS, batch_size=BATCH_SIZE,
    force_retrain=FORCE_RETRAIN,
)
forecast_1 = forecast_future(model_multistep, train_close[-PREDICTION_DAYS:],
                             close_scaler, K_STEPS)
print(f"[C.5-1] Multistep {K_STEPS}-day Close forecast (USD): {np.round(forecast_1, 2)}")
plot_multistep_forecast(recent_close, forecast_1, COMPANY)


# ── [C.5-2] Multivariate: OHLCV → next-day Close ─────────────────────────────
X2, y2 = prepare_sequences(train_features, target_idx=target_idx,
                           prediction_days=PREDICTION_DAYS, k_steps=1)
model_multivariate = load_or_build(
    model_path   = get_model_path(MODEL_DIR, COMPANY, tag='multivariate'),
    build_kwargs = dict(sequence_length=PREDICTION_DAYS, layer_sizes=LAYER_SIZES,
                        layer_type=DL_NETWORK, dropout_rate=DROPOUT_RATE,
                        dense_units=1, n_features=len(FEATURES)),
    x_train=X2, y_train=y2, epochs=EPOCHS, batch_size=BATCH_SIZE,
    force_retrain=FORCE_RETRAIN,
)
# Evaluate across the whole test period using multivariate look-back windows.
mv_inputs   = build_test_inputs_mv(train_data, test_data, FEATURES, scalers, PREDICTION_DAYS)
x_test_mv   = make_windows(mv_inputs, PREDICTION_DAYS)
predicted_mv = inverse_close(model_multivariate.predict(x_test_mv), close_scaler)
plot_predictions(test_data[TARGET_COL].values, predicted_mv, COMPANY)


# ── [C.5-3] Combined: multivariate OHLCV → next k closes ─────────────────────
X3, y3 = prepare_sequences(train_features, target_idx=target_idx,
                           prediction_days=PREDICTION_DAYS, k_steps=K_STEPS)
model_combined = load_or_build(
    model_path   = get_model_path(MODEL_DIR, COMPANY, tag=f'mv_multistep_k{K_STEPS}'),
    build_kwargs = dict(sequence_length=PREDICTION_DAYS, layer_sizes=LAYER_SIZES,
                        layer_type=DL_NETWORK, dropout_rate=DROPOUT_RATE,
                        dense_units=K_STEPS, n_features=len(FEATURES)),
    x_train=X3, y_train=y3, epochs=EPOCHS, batch_size=BATCH_SIZE,
    force_retrain=FORCE_RETRAIN,
)
forecast_3 = forecast_future(model_combined, train_features[-PREDICTION_DAYS:],
                             close_scaler, K_STEPS)
print(f"[C.5-3] Multivariate {K_STEPS}-day Close forecast (USD): {np.round(forecast_3, 2)}")
plot_multistep_forecast(recent_close, forecast_3, COMPANY)

# ==============================
# Task C.6: Machine Learning 3 - Ensemble Learning Options
# ==============================
print("\n─── Task C.6: Thong so Ensemble ───")
print("Chon phuong phap Machine Learning de ket hop voi", DL_NETWORK, ":")
print("1. ARIMA")
print("2. Random Forest")

# Cho user chon phuong phap
user_choice = input("Nhap lua chon cua ban (1 hoac 2) [1]: ").strip()
if user_choice not in ['1', '2']:
    user_choice = '1'

# Thiet lap trong so cho viec ket hop ket qua (Weighted Average)
weight_ml = 0.4
weight_dl = 0.6
predicted_dl_flat = predicted_prices.flatten()

if user_choice == '1':
    print(f"\n[C.6] Dang huan luyen ARIMA ket hop voi {DL_NETWORK}...")
    
    # Lay du lieu 1D cho ARIMA
    train_prices_1d = train_data[PRICE_VALUE].values
    
    # Huan luyen ARIMA
    arima_model = ARIMA(train_prices_1d, order=(5, 1, 0))
    arima_fit = arima_model.fit()
    
    # Du doan tren tap test
    ml_predictions = arima_fit.forecast(steps=len(test_data))
    method_name = "ARIMA"

else:
    print(f"\n[C.6] Dang huan luyen Random Forest ket hop voi {DL_NETWORK}...")
    
    # Reshape du lieu 3D (samples, timesteps, features) thanh 2D cho Random Forest
    x_train_rf = x_train.reshape(x_train.shape[0], -1)
    x_test_rf = x_test.reshape(x_test.shape[0], -1)
    
    # Huan luyen Random Forest
    rf_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    rf_model.fit(x_train_rf, y_train)
    
    # Du doan va giai chuan hoa (inverse transform)
    rf_pred_scaled = rf_model.predict(x_test_rf).reshape(-1, 1)
    ml_predictions = scaler.inverse_transform(rf_pred_scaled).flatten()
    method_name = "Random Forest"

# Tinh toan ket qua Ensemble cuoi cung
ensemble_predicted_prices = (weight_ml * ml_predictions) + (weight_dl * predicted_dl_flat)

# Truc quan hoa
print(f"[C.6] Ve bieu do so sanh ket qua Ensemble...")
plot_predictions(
    actual_prices, 
    ensemble_predicted_prices, 
    f"{COMPANY} (Ensemble {method_name} + {DL_NETWORK})"
)
