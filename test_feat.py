# File: main.py
# Pipeline entry point offering two choices:
#   1. Pure DL  - LSTM/GRU/RNN trained directly on (scaled) log features.
#   2. Hybrid   - VARIMAX linear component + DL residual model.

import os

# Tắt warning một DNN
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Tắt các log tin nhắn thông báo của TensorFlow (chỉ hiện Error)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from data import DataDownloader, DataHandler
from models.dl_model import ModelFactory
from pure_dl_validation import PureDLValidation
from hybrid_validation import HybridValidation

# ── Parameters ────────────────────────────────────
TICKER = "AAPL"
CROSS_ASSET_TICKER = "QQQ"

# Configuration for Log Features Pipeline
USE_LOG_FEATURES = True
INCLUDE_CROSS_ASSET = True       # Use QQQ as an input feature alongside AAPL
PREDICT_QQQ_PRICE = True         # Predict both AAPL and QQQ prices simultaneously
PREDICTION_DAYS = 60             # Look-back window length
K_STEPS = 1                      # Future days to predict (pure DL only)
SPLIT_METHOD = "ratio"
SPLIT_PARAM = 0.8                # 80% train / 20% test

DL_TYPE = "GRU"                  # 'LSTM', 'GRU', or 'RNN'
NUM_LAYERS = 3
UNITS = 64
DROPOUT_RATE = 0.2
VARIMAX_ORDER = (1, 1)           # (p, q) order for Hybrid's linear VARIMAX component

EPOCHS = 50
BATCH_SIZE = 32


def choose_pipeline() -> str:
    """Prompt the user to pick between the pure-DL and Hybrid pipelines."""
    print("Select pipeline:")
    print("  1. Pure DL  - LSTM/GRU/RNN trained directly on features")
    print("  2. Hybrid   - VARIMAX linear component + DL residual model")
    choice = input("Enter 1 or 2 [default: 1]: ").strip()
    return "hybrid" if choice == "2" else "pure_dl"


def run_pure_dl():
    # 1. Download (or reuse cached CSV).
    tickers = [TICKER, CROSS_ASSET_TICKER] if (INCLUDE_CROSS_ASSET or PREDICT_QQQ_PRICE) else [TICKER]
    downloader = DataDownloader(tickers)
    csv_paths = downloader.download()

    # 2. Process data: Clean, Engineer Log Features, Scale, Split, and Window.
    handler = DataHandler(
        csv_paths,
        target_ticker=TICKER,
        include_cross_asset=INCLUDE_CROSS_ASSET or PREDICT_QQQ_PRICE,
        predict_cross_asset=PREDICT_QQQ_PRICE,
    )
    (X_train, y_train), (X_test, y_test) = handler.get_train_test(
        window=PREDICTION_DAYS,
        k=K_STEPS,
        split_method=SPLIT_METHOD,
        split_param=SPLIT_PARAM,
        scale_columns=True,
        use_log_features=USE_LOG_FEATURES
    )

    print(f"[main.py] Features used ({len(handler.feature_columns)}): {handler.feature_columns}")
    print(f"[main.py] Target column(s): {handler.target_column}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # 3. Build the model via ModelFactory.
    n_features = X_train.shape[-1]
    output_dim = y_train.shape[-1] if y_train.ndim > 1 else 1
    dl_model = ModelFactory.get_model(
        DL_TYPE,
        input_shape=(PREDICTION_DAYS, n_features),
        output_dim=output_dim
    )
    dl_model.build_model(num_layers=NUM_LAYERS, units=UNITS, dropout_rate=DROPOUT_RATE)
    dl_model.model.summary()

    # 4. Train model.
    dl_model.train(
        X_train, y_train,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test) if len(X_test) > 0 else None,
    )

    # 5. Evaluate model.
    model_path = os.path.join("trained_models", f"{type(dl_model).__name__}.keras")

    tester = PureDLValidation(model_path=model_path)
    tester.run(
        X_test, y_test,
        model_info={"input_dim": [PREDICTION_DAYS, n_features], "output_dim": output_dim},
        target_names=handler.target_tickers,
        raw_close_map=handler.raw_close_test_map,
    )


def run_hybrid():
    # 1. Download (or reuse cached CSV).
    tickers = [TICKER, CROSS_ASSET_TICKER] if (INCLUDE_CROSS_ASSET or PREDICT_QQQ_PRICE) else [TICKER]
    downloader = DataDownloader(tickers)
    csv_paths = downloader.download()

    # 2. Process data: Clean, Engineer Log Features, Split, and Scale.
    handler = DataHandler(
        csv_paths,
        target_ticker=TICKER,
        include_cross_asset=INCLUDE_CROSS_ASSET or PREDICT_QQQ_PRICE,
        predict_cross_asset=PREDICT_QQQ_PRICE,
    )
    handler.clean()

    if USE_LOG_FEATURES:
        handler.engineer_features()

    train_df, test_df = handler.split(split_method=SPLIT_METHOD, split_param=SPLIT_PARAM)
    train_df, test_df = handler.scale(train_df, test_df)

    feature_cols = handler.feature_columns
    target_cols = handler.target_column if isinstance(handler.target_column, list) else [handler.target_column]

    exog_columns = [c for c in feature_cols if c not in target_cols]
    
    y_train = train_df[target_cols].values
    y_test = test_df[target_cols].values
    exog_train = train_df[exog_columns].values if exog_columns else None
    exog_test = test_df[exog_columns].values if exog_columns else None
    
    # Flatten to 1D if single target
    if len(target_cols) == 1:
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()

    output_dim = len(target_cols)
    n_features = output_dim + len(exog_columns)

    print(f"[main.py] Features used ({len(feature_cols)}): {feature_cols}")
    print(f"[main.py] Target column(s): {target_cols}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")
    print(f"Exog features: {exog_columns if exog_columns else 'none'}")

    # 3. Build the Hybrid model via ModelFactory.
    hybrid = ModelFactory.get_model(
        "hybrid",
        input_shape=(PREDICTION_DAYS, n_features),
        output_dim=output_dim,
        varimax_order=VARIMAX_ORDER,
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

    # 5. Evaluate Hybrid model.
    model_path = os.path.join("trained_models", f"{type(hybrid).__name__}.keras")
    tester = HybridValidation(
        model_path=model_path,
        input_shape=(PREDICTION_DAYS, n_features),
        output_dim=output_dim,
        varimax_order=VARIMAX_ORDER,
        dl_type=DL_TYPE,
    )
    tester.run(
        y_test, exog_test,
        target_names=handler.target_tickers,
        raw_close_map=handler.raw_close_test_map,
        model_info={"input_dim": [PREDICTION_DAYS, n_features], "output_dim": output_dim},
    )


def main():
    mode = choose_pipeline()
    if mode == "hybrid":
        run_hybrid()
    else:
        run_pure_dl()


if __name__ == "__main__":
    main()