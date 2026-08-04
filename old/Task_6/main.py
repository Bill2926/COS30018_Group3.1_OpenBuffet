# File: main.py
# Authors: Bao Vo, Cheong Koo (original)
# Version: v0.4 - Task C6
#
# This is the Entry Point.
# All logic lives in the four modules:
#   data_loader.py   : download / cache / split / scale stock data (Task C.2)
#   visualiser.py    : candlestick & boxplot charts, prediction plot (Task C.3)
#   model_builder.py : build_model() + save/load helpers              (Task C.4)
#   predictor.py     : scale test data, slide windows, inverse-transform

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from old.Task_6.data_loader   import load_and_process_dataset, prepare_lstm_data, prepare_sequences
from old.Task_6.visualiser    import (plot_candlestick_chart, plot_boxplot_chart,
                           plot_predictions, plot_multistep_forecast,
                           ask_int, ask_date, slice_range)
from old.Task_6.model_builder import build_model, get_model_path, load_or_build
from old.Task_6.predictor     import (build_test_inputs, make_predictions, predict_next_day,
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
FORCE_RETRAIN = False


# ==============================
# Step 1 : Load and split data  (Task C.2)
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
# Step 2 : Visualise the training data  (Task C.3)
# ==============================
print("\n--- Task C.3: chart options (press Enter to accept the [default]) ---")
n_days = ask_int("Days per candle / box window (n) [1]: ", default=1)
start  = ask_date("Start date : year / year-month / date [all data]: ", is_end=False)
end    = ask_date("End date   : year / year-month / date [all data]: ", is_end=True)

view = slice_range(train_data.copy(), start, end)

if len(view) == 0:
    print(f"No trading days in that range : nothing to plot. "
          f"Data spans {train_data.index[0].date()} -> {train_data.index[-1].date()}.")
else:
    print(f"Plotting {len(view)} trading days "
          f"({view.index[0].date()} -> {view.index[-1].date()}).")

    plot_candlestick_chart(view, n_days=n_days,
                           title=f'{COMPANY} {n_days}-Day Candles', show=False)

    if n_days >= 2 and len(view) >= n_days:
        plot_boxplot_chart(view, column='Close', n_days=n_days,
                           title=f'{COMPANY} Close Prices', show=False)
    else:
        print("Skipping boxplot (need n >= 2 and at least n days in the range).")


# ==============================
# Step 3 : Scale Close prices and build LSTM input sequences
# ==============================
scaler      = MinMaxScaler(feature_range=(0, 1))
train_close = train_data[PRICE_VALUE].values.reshape(-1, 1)
scaled_train = scaler.fit_transform(train_close).flatten()

x_train, y_train = prepare_lstm_data(scaled_train, PREDICTION_DAYS)


# ==============================================================================
# Global variables for lazy loading (Avoid training before menu selection)
# ==============================================================================
model_base = None
predicted_prices_base = None
actual_prices = test_data[PRICE_VALUE].values


# ==============================================================================
# Interactive Execution Menu (Task C.6 Latest)
# ==============================================================================
while True:
    print("\n" + "="*60)
    print("      NVIDIA (NVDA) PREDICTION SCENARIO INTERACTIVE MENU")
    print("="*60)
    print("1. [Task C.4] Baseline Single-step Univariate Prediction (Test Set Plot)")
    print("2. [Task C.4] Next-Day Closing Price Forecast Point")
    print("3. [Task C.5-1] Multistep Univariate Forecast (Next k Days Future)")
    print("4. [Task C.5-2] Multivariate Single-step Prediction (OHLCV -> Next Close)")
    print("5. [Task C.5-3] Combined Multivariate Multistep Forecast (OHLCV -> k Days)")
    print("6. [Task C.6] Ensemble Learning Operations (ARIMA / Random Forest + DL)")
    print("7. Exit Program")
    print("-" * 60)
    
    choice = input("Enter your choice (1-7): ").strip()
    
    # Helper to ensure base model is trained/loaded only when needed
    if choice in ['1', '2', '6'] and model_base is None:
        print(f"\n[Initializing Task C.4 Base Model] Loading/Training {DL_NETWORK}...")
        model_path = get_model_path(MODEL_DIR, COMPANY, tag=f'{DL_NETWORK}_{"x".join(map(str, LAYER_SIZES))}')
        build_kwargs = dict(sequence_length=PREDICTION_DAYS, layer_sizes=LAYER_SIZES, layer_type=DL_NETWORK,
                            dropout_rate=DROPOUT_RATE, dense_units=1, optimizer='adam', loss='mean_squared_error')
        model_base = load_or_build(model_path, build_kwargs, x_train, y_train, EPOCHS, BATCH_SIZE, FORCE_RETRAIN)
        
        # Build base test inputs & pre-compute predictions
        model_inputs = build_test_inputs(train_data[PRICE_VALUE], test_data[PRICE_VALUE], scaler, PREDICTION_DAYS)
        x_test = make_predictions(model_base, model_inputs, PREDICTION_DAYS)
        predicted_scaled = model_base.predict(x_test)
        predicted_prices_base = scaler.inverse_transform(predicted_scaled)

    if choice == '1':
        # ----------------------------------------------------------------------
        # Prediction Run 1 : [Task C.4] Baseline Single-step Univariate Prediction
        # Purpose: Validates the baseline Deep Learning model performance on the
        # test dataset by plotting actual vs single-step univariate predictions.
        # ----------------------------------------------------------------------
        print(f"\n[Executing Run 1 : Task C.4] Plotting single-step evaluation...")
        plot_predictions(actual_prices, predicted_prices_base, COMPANY)
        
    elif choice == '2':
        # ----------------------------------------------------------------------
        # Prediction Run 2 : [Task C.4] Next-Day Closing Price Forecast
        # Purpose: Predicts a single price target representing the immediate next 
        # trading session following the terminal date of the test framework.
        # ----------------------------------------------------------------------
        print(f"\n[Executing Run 2 : Task C.4] Forecasting next single trading day...")
        model_inputs = build_test_inputs(train_data[PRICE_VALUE], test_data[PRICE_VALUE], scaler, PREDICTION_DAYS)
        next_day_price = predict_next_day(model_base, model_inputs, PREDICTION_DAYS, scaler)
        print(f"\n[main] Next-day predicted closing price for {COMPANY}: ${next_day_price:.2f}")
        
    elif choice == '3':
        # ----------------------------------------------------------------------
        # Prediction Run 3 : [Task C.5-1] Multistep Univariate Forecast
        # Purpose: Generates an isolated vector of 'k' future daily prices purely
        # from past closing figures using multi-dense output configurations.
        # ----------------------------------------------------------------------
        print("\n--- Task C.5-1 Configuration ---")
        K_STEPS = ask_int("Forecast horizon k (days into the future) [5]: ", default=5)
        
        train_scaled, _, scalers = load_and_process_dataset(
            COMPANY, TRAIN_START, TEST_END, ['Close'], DATA_DIR, False, 'forward_fill', 'date', TEST_START, True
        )
        close_scaler = scalers['Close']
        train_close_mv = train_scaled[['Close']].values
        recent_close = train_data[PRICE_VALUE].values[-PREDICTION_DAYS:]

        X1, y1 = prepare_sequences(train_close_mv, target_idx=0, prediction_days=PREDICTION_DAYS, k_steps=K_STEPS)
        model_multistep = load_or_build(
            get_model_path(MODEL_DIR, COMPANY, tag=f'multistep_k{K_STEPS}'),
            dict(sequence_length=PREDICTION_DAYS, layer_sizes=LAYER_SIZES, layer_type=DL_NETWORK, dropout_rate=DROPOUT_RATE, dense_units=K_STEPS, n_features=1),
            X1, y1, EPOCHS, BATCH_SIZE, FORCE_RETRAIN
        )
        
        print(f"\n[Executing Run 3 : Task C.5-1] Executing univariate multistep forecast...")
        forecast_1 = forecast_future(model_multistep, train_close_mv[-PREDICTION_DAYS:], close_scaler, K_STEPS)
        print(f"[C.5-1] Multistep {K_STEPS}-day Close forecast (USD): {np.round(forecast_1, 2)}")
        plot_multistep_forecast(recent_close, forecast_1, COMPANY)
        
    elif choice == '4':
        # ----------------------------------------------------------------------
        # Prediction Run 4 : [Task C.5-2] Multivariate Single-step Prediction
        # Purpose: Evaluates test dataset performance when tracking multi-feature 
        # parameters (OHLCV) concurrently to calculate a single future sequence step.
        # ----------------------------------------------------------------------
        FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
        train_scaled, test_scaled, scalers = load_and_process_dataset(
            COMPANY, TRAIN_START, TEST_END, FEATURES, DATA_DIR, False, 'forward_fill', 'date', TEST_START, True
        )
        target_idx = FEATURES.index('Close')
        close_scaler = scalers['Close']
        train_features = train_scaled[FEATURES].values

        X2, y2 = prepare_sequences(train_features, target_idx=target_idx, prediction_days=PREDICTION_DAYS, k_steps=1)
        model_multivariate = load_or_build(
            get_model_path(MODEL_DIR, COMPANY, tag='multivariate'),
            dict(sequence_length=PREDICTION_DAYS, layer_sizes=LAYER_SIZES, layer_type=DL_NETWORK, dropout_rate=DROPOUT_RATE, dense_units=1, n_features=len(FEATURES)),
            X2, y2, EPOCHS, BATCH_SIZE, FORCE_RETRAIN
        )
        
        print(f"\n[Executing Run 4 : Task C.5-2] Executing multi-feature evaluation...")
        mv_inputs = build_test_inputs_mv(train_data, test_data, FEATURES, scalers, PREDICTION_DAYS)
        x_test_mv = make_windows(mv_inputs, PREDICTION_DAYS)
        predicted_mv = inverse_close(model_multivariate.predict(x_test_mv), close_scaler)
        plot_predictions(test_data['Close'].values, predicted_mv, COMPANY)
        
    elif choice == '5':
        # ----------------------------------------------------------------------
        # Prediction Run 5 : [Task C.5-3] Combined Multivariate Multistep Forecast
        # Purpose: Simulates a highly comprehensive scenario mapping high-dimensional
        # features (OHLCV) directly into a vector prediction window spanning 'k' days.
        # ----------------------------------------------------------------------
        print("\n--- Task C.5-3 Configuration ---")
        K_STEPS = ask_int("Forecast horizon k (days into the future) [5]: ", default=5)
        
        FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
        train_scaled, test_scaled, scalers = load_and_process_dataset(
            COMPANY, TRAIN_START, TEST_END, FEATURES, DATA_DIR, False, 'forward_fill', 'date', TEST_START, True
        )
        target_idx = FEATURES.index('Close')
        close_scaler = scalers['Close']
        train_features = train_scaled[FEATURES].values
        recent_close = train_data[PRICE_VALUE].values[-PREDICTION_DAYS:]

        X3, y3 = prepare_sequences(train_features, target_idx=target_idx, prediction_days=PREDICTION_DAYS, k_steps=K_STEPS)
        model_combined = load_or_build(
            get_model_path(MODEL_DIR, COMPANY, tag=f'mv_multistep_k{K_STEPS}'),
            dict(sequence_length=PREDICTION_DAYS, layer_sizes=LAYER_SIZES, layer_type=DL_NETWORK, dropout_rate=DROPOUT_RATE, dense_units=K_STEPS, n_features=len(FEATURES)),
            X3, y3, EPOCHS, BATCH_SIZE, FORCE_RETRAIN
        )
        
        print(f"\n[Executing Run 5 : Task C.5-3] Executing joint multivariate multistep forecast...")
        forecast_3 = forecast_future(model_combined, train_features[-PREDICTION_DAYS:], close_scaler, K_STEPS)
        print(f"[C.5-3] Multivariate {K_STEPS}-day Close forecast (USD): {np.round(forecast_3, 2)}")
        plot_multistep_forecast(recent_close, forecast_3, COMPANY)
        
    elif choice == '6':
        # ----------------------------------------------------------------------
        # Prediction Run 6 : [Task C.6] Ensemble Learning Framework
        # Purpose: Fuses standard mathematical/shallow algorithmic prediction weights 
        # (ARIMA / Random Forest) directly with the neural architecture (DL_NETWORK).
        # ----------------------------------------------------------------------
        print("\n--- Task C.6: Ensemble Configuration ---")
        print("Select the Machine Learning method to combine with", DL_NETWORK, ":")
        print("1. ARIMA")
        print("2. Random Forest")
        
        user_choice = input("Enter your choice (1 or 2) [1]: ").strip()
        if user_choice not in ['1', '2']:
            user_choice = '1'

        weight_ml = 0.4
        weight_dl = 0.6
        predicted_dl_flat = predicted_prices_base.flatten()

        if user_choice == '1':
            print(f"\n[C.6] Training ARIMA combined with {DL_NETWORK}...")
            train_prices_1d = train_data[PRICE_VALUE].values
            arima_model = ARIMA(train_prices_1d, order=(5, 1, 0))
            arima_fit = arima_model.fit()
            ml_predictions = arima_fit.forecast(steps=len(test_data))
            method_name = "ARIMA"
        else:
            print(f"\n[C.6] Training Random Forest combined with {DL_NETWORK}...")
            x_train_rf = x_train.reshape(x_train.shape[0], -1)
            
            model_inputs = build_test_inputs(train_data[PRICE_VALUE], test_data[PRICE_VALUE], scaler, PREDICTION_DAYS)
            x_test_local = make_predictions(model_base, model_inputs, PREDICTION_DAYS)
            x_test_rf = x_test_local.reshape(x_test_local.shape[0], -1)
            
            rf_model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
            rf_model.fit(x_train_rf, y_train)
            rf_pred_scaled = rf_model.predict(x_test_rf).reshape(-1, 1)
            ml_predictions = scaler.inverse_transform(rf_pred_scaled).flatten()
            method_name = "Random Forest"

        ensemble_predicted_prices = (weight_ml * ml_predictions) + (weight_dl * predicted_dl_flat)
        print(f"[C.6] Plotting Ensemble comparison results...")
        plot_predictions(actual_prices, ensemble_predicted_prices, f"{COMPANY} (Ensemble {method_name} + {DL_NETWORK})")
        
    elif choice == '7':
        print("\nExiting Interactive Pipeline. Close all active figures to terminate completely.")
        break
    else:
        print("\n[Selection Error] Please input a valid selection code (1-7).")