# File: data_loader.py
# Task C.2: Load, clean, split, and optionally scale stock data.
# Kept in its own module because every other module (visualiser, model, predictor)
# depends on clean data but none of them care HOW it was obtained.

import os
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


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
    """
    Download (or load from cache) a stock's OHLCV data, handle NaNs,
    split into train/test, and optionally scale each column.

    Parameters
    ----------
    company       : Yahoo Finance ticker symbol, e.g. 'NVDA'.
    start_date    : Inclusive start of the whole dataset ('YYYY-MM-DD').
    end_date      : Exclusive end of the whole dataset ('YYYY-MM-DD').
    features      : Subset of columns to keep; None keeps all columns.
    data_dir      : Directory where the CSV cache is stored / read from.
    force_download: Re-download even when a cache file exists.
    nan_method    : How to fill NaN values: 'forward_fill', 'drop', or 'mean'.
    split_method  : 'date' (split at a calendar date) or 'ratio' (0‒1 float).
    split_param   : The date string or ratio string matching split_method.
    scale_columns : If True, MinMaxScale every column to [0, 1] and return
                    the fitted scalers so they can be inverted later.

    Returns
    -------
    train_df : DataFrame of training rows.
    test_df  : DataFrame of test rows (empty if split_method is unknown).
    scalers  : Dict {column_name: MinMaxScaler} (empty if scale_columns=False).
    """

    # ── Validate date range ──────────────────────────────────────────────────
    start_dt = pd.to_datetime(start_date)
    end_dt   = pd.to_datetime(end_date)
    if start_dt >= end_dt:
        raise ValueError("start_date must be strictly before end_date.")

    # ── Load or download ─────────────────────────────────────────────────────
    os.makedirs(data_dir, exist_ok=True)
    cache_path = os.path.join(data_dir, f'{company}_{start_date}_{end_date}.csv')

    if os.path.exists(cache_path) and not force_download:
        print(f"[data_loader] Loading cached data from {cache_path} …")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    else:
        print(f"[data_loader] Downloading {company} from Yahoo Finance …")
        df = yf.download(company, start_dt, end_dt, auto_adjust=True)
        # yfinance may return a MultiIndex if multiple tickers were requested;
        # flatten it to a plain column index.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache_path)

    # ── Feature selection ────────────────────────────────────────────────────
    if features is not None:
        available = [f for f in features if f in df.columns]
        df = df[available]

    # ── NaN handling ─────────────────────────────────────────────────────────
    # 'forward_fill': propagate last valid observation forward (safe, no leakage).
    # 'drop'        : remove any row that contains a NaN.
    # 'mean'        : replace NaN with the column mean (loses temporal structure).
    if nan_method == 'forward_fill':
        df = df.ffill()
    elif nan_method == 'drop':
        df = df.dropna()
    elif nan_method == 'mean':
        df = df.fillna(df.mean())
    df = df.bfill()  # safety back-fill for leading NaNs not caught by ffill

    # ── Train / test split ───────────────────────────────────────────────────
    if split_method == 'date':
        cutoff   = pd.Timestamp(split_param)
        train_df = df[df.index < cutoff].copy()
        test_df  = df[df.index >= cutoff].copy()
    elif split_method == 'ratio':
        idx      = int(len(df) * float(split_param))
        train_df = df.iloc[:idx].copy()
        test_df  = df.iloc[idx:].copy()
    else:
        train_df = df.copy()
        test_df  = pd.DataFrame()

    # ── Scaling ──────────────────────────────────────────────────────────────
    # We fit the scaler ONLY on training data to avoid data leakage: the scaler
    # must not "know" any statistics from the test period.
    scalers = {}
    if scale_columns:
        for col in train_df.columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_df[col] = scaler.fit_transform(train_df[[col]])
            if len(test_df) > 0:
                test_df[col] = scaler.transform(test_df[[col]])
            scalers[col] = scaler

    return train_df, test_df, scalers


def prepare_lstm_data(series: np.ndarray, prediction_days: int):
    """
    Slide a window of length `prediction_days` over `series` to build
    supervised-learning input/output pairs for an LSTM.

    Each row of X contains `prediction_days` consecutive scaled prices;
    the matching element of y is the very next price.

    Parameters
    ----------
    series          : 1-D NumPy array of scaled prices.
    prediction_days : Look-back window length.

    Returns
    -------
    X : np.ndarray of shape (n_samples, prediction_days, 1).
    y : np.ndarray of shape (n_samples,).
    """
    X, y = [], []
    for i in range(prediction_days, len(series)):
        X.append(series[i - prediction_days:i])
        y.append(series[i])
    X, y = np.array(X), np.array(y)
    # Reshape X to (samples, timesteps, features=1) as required by Keras RNN layers.
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, y


def prepare_sequences(data: np.ndarray, target_idx: int,
                      prediction_days: int, k_steps: int = 1):
    """
    Build supervised windows for the multivariate and/or multistep problems (Task C.5).

    This generalises prepare_lstm_data() along two axes at once:
      * multivariate : `data` may have several feature columns, all of which go
                       into each input window.
      * multistep    : each target is the next `k_steps` values of the target
                       column (not just the single next value).

    Parameters
    ----------
    data            : 2-D array of shape (n_rows, n_features) of SCALED values.
    target_idx      : Column index of the feature we predict (e.g. Close).
    prediction_days : Look-back window length (timesteps fed to the model).
    k_steps         : How many future steps to predict (k >= 1).

    Returns
    -------
    X : np.ndarray (n_samples, prediction_days, n_features) — input windows.
    y : np.ndarray (n_samples, k_steps)                     — next k target values.

    Note: with a single feature column and k_steps=1 this reduces to the same
    pairs as prepare_lstm_data (only y is 2-D here, shape (n_samples, 1)).
    """
    if data.ndim != 2:
        raise ValueError("data must be 2-D (n_rows, n_features).")
    if k_steps < 1:
        raise ValueError("k_steps must be >= 1.")

    X, y = [], []
    # Stop k_steps early so every window has a full k-length target ahead of it.
    for i in range(prediction_days, len(data) - k_steps + 1):
        X.append(data[i - prediction_days:i, :])        # look-back, all features
        y.append(data[i:i + k_steps, target_idx])       # next k target values
    return np.array(X), np.array(y)
