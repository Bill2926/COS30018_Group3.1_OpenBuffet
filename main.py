# File: main.py
# Pipeline entry point offering two choices:
#   1. Pure DL  - LSTM/GRU/RNN trained directly on (scaled) log features.
#   2. Hybrid   - VARIMAX linear component + DL residual model.

import argparse
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
PREDICTION_DAYS = 30             # Look-back window length
K_STEPS = 1                      # Future days to predict (pure DL only)
SPLIT_METHOD = "ratio"
SPLIT_PARAM = 0.8                # 80% train / 20% test

DL_TYPE = "LSTM"                  # 'LSTM', 'GRU', or 'RNN'
NUM_LAYERS = 2
UNITS = (32,16)            # int (same width every layer) or a tuple with one entry per layer, last = final layer
DROPOUT_RATE = 0.1
VARIMAX_ORDER = (1, 1)           # (p, q) order for Hybrid's linear VARIMAX component

EPOCHS = 50
BATCH_SIZE = 32

START_DATE = "23-07-2012"
END_DATE = "27-01-2020"

# Cross asset
INCLUDE_CROSS_ASSET = True       # Use QQQ as an input feature alongside AAPL
PREDICT_QQQ_PRICE = True         # Predict both AAPL and QQQ prices simultaneously

# Sentiment
USE_SENTIMENT = True
SENTIMENT_PATH = "sentiments/aapl_sen_23-07-2012_27-01-2020"    # Currently only support AAPL

# TDA and ^VIX score
USE_TDA = True    # Add rolling Persistent Entropy (tda.py) as an extra feature; window matches PREDICTION_DAYS below
VIX_TICKER = "^VIX"
USE_VIX = True    # Add CBOE Volatility Index (VIX) daily close, fetched via yfinance, as an extra feature


def choose_pipeline() -> str:
    """Pick between the pure-DL and Hybrid pipelines via --mode (1=Pure DL, 2=Hybrid; default 1)."""
    parser = argparse.ArgumentParser(description="Run the Pure DL or Hybrid stock prediction pipeline.")
    parser.add_argument(
        "--mode", type=int, choices=[1, 2], default=1,
        help="1 = Pure DL (LSTM/GRU/RNN on features), 2 = Hybrid (VARIMAX + DL residual). Default: 1.",
    )
    args = parser.parse_args()
    return "hybrid" if args.mode == 2 else "pure_dl"


def run_pure_dl(plots_dir: str = "plots", results_dir: str = "results",
                 results_filename: str | None = None):
    # 1. Download (or reuse cached CSV).
    tickers = [TICKER, CROSS_ASSET_TICKER] if (INCLUDE_CROSS_ASSET or PREDICT_QQQ_PRICE) else [TICKER]
    downloader = DataDownloader(tickers, START_DATE, END_DATE)
    csv_paths = downloader.download()

    vix_csv_path = None
    if USE_VIX:
        vix_csv_path = DataDownloader(VIX_TICKER, START_DATE, END_DATE).download_ticker(VIX_TICKER)

    # 2. Process data: Clean, Engineer Log Features, Scale, Split, and Window.
    handler = DataHandler(
        csv_paths,
        target_ticker=TICKER,
        include_cross_asset=INCLUDE_CROSS_ASSET or PREDICT_QQQ_PRICE,
        predict_cross_asset=PREDICT_QQQ_PRICE,
        use_sentiment=USE_SENTIMENT,
        sentiment_path=SENTIMENT_PATH,
        use_tda=USE_TDA,
        tda_window_size=PREDICTION_DAYS,
        use_vix=USE_VIX,
        vix_csv_path=vix_csv_path,
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

    tester = PureDLValidation(model_path=model_path, plots_dir=plots_dir, results_dir=results_dir)
    return tester.run(
        X_test, y_test,
        model_info={
            "mode": "Pure_DL",
            "model_type": DL_TYPE,
            "include_cross_asset": INCLUDE_CROSS_ASSET,
            "use_sentiment": USE_SENTIMENT,
            "use_tda": USE_TDA,
            "use_vix": USE_VIX,
            "input_dim": [PREDICTION_DAYS, n_features],
            "output_dim": output_dim,
        },
        target_names=handler.target_tickers,
        raw_close_map=handler.raw_close_test_map,
        results_filename=results_filename,
    )


def run_hybrid(plots_dir: str = "plots", results_dir: str = "results",
                results_filename: str | None = None):
    # 1. Download (or reuse cached CSV).
    tickers = [TICKER, CROSS_ASSET_TICKER] if (INCLUDE_CROSS_ASSET or PREDICT_QQQ_PRICE) else [TICKER]
    downloader = DataDownloader(tickers, start_date=START_DATE, end_date=END_DATE)
    csv_paths = downloader.download()

    vix_csv_path = None
    if USE_VIX:
        vix_csv_path = DataDownloader(VIX_TICKER, START_DATE, END_DATE).download_ticker(VIX_TICKER)

    # 2. Process data: Clean, Engineer Log Features, Split, and Scale.
    handler = DataHandler(
        csv_paths,
        target_ticker=TICKER,
        include_cross_asset=INCLUDE_CROSS_ASSET or PREDICT_QQQ_PRICE,
        predict_cross_asset=PREDICT_QQQ_PRICE,
        use_sentiment=USE_SENTIMENT,
        sentiment_path=SENTIMENT_PATH,
        use_tda=USE_TDA,
        tda_window_size=PREDICTION_DAYS,
        use_vix=USE_VIX,
        vix_csv_path=vix_csv_path,
    )
    handler.clean()

    if USE_LOG_FEATURES:
        handler.engineer_features()

    train_df, test_df = handler.split(split_method=SPLIT_METHOD, split_param=SPLIT_PARAM)
    train_df, test_df = handler.scale(train_df, test_df)

    # get_train_test() (used by run_pure_dl) populates raw_close_test_map as a
    # side effect; run_hybrid() drives clean/engineer_features/split/scale
    # directly instead, so it must fill this in itself for price-space
    # metrics (reconstructed via C_t * exp(y_t)) to be available below.
    handler.raw_close_test_map = {}
    for ticker in handler.target_tickers:
        col = "Raw_Close" if ticker == TICKER else f"{ticker}_Raw_Close"
        if col in test_df.columns:
            handler.raw_close_test_map[ticker] = np.asarray(test_df[col].values)

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
        plots_dir=plots_dir,
        results_dir=results_dir,
    )
    return tester.run(
        y_test, exog_test,
        target_names=handler.target_tickers,
        raw_close_map=handler.raw_close_test_map,
        model_info={
            "mode": "Hybrid",
            "model_type": f"VARMAX+{DL_TYPE}",
            "include_cross_asset": INCLUDE_CROSS_ASSET,
            "use_sentiment": USE_SENTIMENT,
            "use_tda": USE_TDA,
            "use_vix": USE_VIX,
            "input_dim": [PREDICTION_DAYS, n_features],
            "output_dim": output_dim,
        },
        results_filename=results_filename,
    )


def main():
    mode = choose_pipeline()
    if mode == "hybrid":
        run_hybrid()
    else:
        run_pure_dl()


if __name__ == "__main__":
    main()