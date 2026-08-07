# File: main.py
# Pipeline entry point offering two choices:
#   1. Pure DL       - LSTM/GRU/RNN trained directly on (scaled) log features.
#   2. Hybrid        - ARIMA linear component + DL residual model.

import os
import os

# Tắt warning một DNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Tắt các log tin nhắn thông báo của TensorFlow (chỉ hiện Error)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from data import DataDownloader, DataHandler
from models.dl_model import ModelFactory
from validation import Validation
from hybrid_test import HybridTest

# ── Parameters ────────────────────────────────────
TICKER = "AAPL"

# Configuration for Log Features Pipeline
USE_LOG_FEATURES = True
PREDICTION_DAYS = 60             # look-back window length
K_STEPS = 1                      # future days to predict (pure DL only)
SPLIT_METHOD = "ratio"
SPLIT_PARAM = 0.8                # 80% train / 20% test

DL_TYPE = "GRU"                 # 'LSTM', 'GRU', or 'RNN'
NUM_LAYERS = 2
UNITS = 64
DROPOUT_RATE = 0.2
ARIMA_ORDER = (5, 1, 0)          # (p, d, q) for Hybrid's linear component

EPOCHS = 5
BATCH_SIZE = 32


def choose_pipeline() -> str:
    """Prompt the user to pick between the pure-DL and Hybrid pipelines."""
    print("Select pipeline:")
    print("  1. Pure DL  - LSTM/GRU/RNN trained directly on features")
    print("  2. Hybrid   - ARIMA linear component + DL residual model")
    choice = input("Enter 1 or 2 [default: 1]: ").strip()
    return "hybrid" if choice == "2" else "pure_dl"


def run_pure_dl():
    # 1. Download (or reuse cached CSV).
    downloader = DataDownloader(TICKER)
    csv_path = downloader.download()

    # 2. Process data: Clean, Engineer Log Features, Scale, Split, and Window.
    #    When use_log_features=True, feature_columns and target_column are automatically
    #    resolved inside DataHandler to the newly generated log features.
    handler = DataHandler(csv_path)
    (X_train, y_train), (X_test, y_test) = handler.get_train_test(
        window=PREDICTION_DAYS,
        k=K_STEPS,
        split_method=SPLIT_METHOD,
        split_param=SPLIT_PARAM,
        scale_columns=True,
        use_log_features=USE_LOG_FEATURES
    )
    
    print(f"[main.py] Features used ({len(handler.feature_columns)}): {handler.feature_columns}")
    print(f"[main.py] Target column: {handler.target_column}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # 3. Build the model via the factory.
    n_features = X_train.shape[-1]
    dl_model = ModelFactory.get_model(
        DL_TYPE,
        input_shape=(PREDICTION_DAYS, n_features),
        output_dim=K_STEPS
    )
    dl_model.build_model(num_layers=NUM_LAYERS, units=UNITS, dropout_rate=DROPOUT_RATE)
    dl_model.model.summary()

    # 4. Train model.
    dl_model.train(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test) if len(X_test) > 0 else None,
    )

    # 5. Test: Loss curve, convergence stats, log/price accuracy metrics, and price plots.
    model_path = os.path.join("trained_models", f"{type(dl_model).__name__}.keras")
    
    # Retrieve unscaled base Close prices (C_t) for price reconstruction in Test
    raw_close_test = handler.raw_close_test
    
    tester = Validation(model_path=model_path)
    tester.run(X_test, y_test, raw_close_today=raw_close_test)


def run_hybrid():
    # 1. Download (or reuse cached CSV).
    downloader = DataDownloader(TICKER)
    csv_path = downloader.download()

    # 2. Clean, engineer log features, split, and scale (fit on train only).
    handler = DataHandler(csv_path)
    handler.clean()
    
    if USE_LOG_FEATURES:
        handler.engineer_features()

    train_df, test_df = handler.split(split_method=SPLIT_METHOD, split_param=SPLIT_PARAM)
    train_df, test_df = handler.scale(train_df, test_df)

    feature_cols = handler.feature_columns
    target_col = handler.target_column

    exog_columns = [c for c in feature_cols if c != target_col]
    y_train = train_df[target_col].values
    y_test = test_df[target_col].values
    exog_train = train_df[exog_columns].values if exog_columns else None
    exog_test = test_df[exog_columns].values if exog_columns else None
    
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}, "
          f"exog features: {exog_columns if exog_columns else 'none'}")

    # 3. Build the Hybrid model via ModelFactory.
    n_features = 1 + len(exog_columns)
    hybrid = ModelFactory.get_model(
        "hybrid",
        input_shape=(PREDICTION_DAYS, n_features),
        output_dim=1,
        arima_order=ARIMA_ORDER,
        dl_type=DL_TYPE
    )
    hybrid.build_model(num_layers=NUM_LAYERS, units=UNITS, dropout_rate=DROPOUT_RATE)
    hybrid.model.summary()

    # 4. Train Hybrid model.
    hybrid.train(
        y_train, exog_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(y_test, exog_test),
    )

    # 5. Test Hybrid model.
    model_path = os.path.join("trained_models", f"{type(hybrid).__name__}.keras")
    tester = HybridTest(
        model_path=model_path,
        input_shape=(PREDICTION_DAYS, n_features),
        arima_order=ARIMA_ORDER,
        dl_type=DL_TYPE
    )
    tester.run(y_test, exog_test)


def main():
    mode = choose_pipeline()
    if mode == "hybrid":
        run_hybrid()
    else:
        run_pure_dl()


if __name__ == "__main__":
    main()