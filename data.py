# File: data_handler.py
# Two classes, split by responsibility:
#   DataDownloader - pulls stock data from yfinance for a chosen ticker/timeframe
#                     and caches it to cache/ as CSV (the only network-facing piece).
#   DataHandler     - takes that CSV, cleans it (NaN / weekend-holiday gaps),
#                     scales it, splits into train/test, and windows each split
#                     into (X, y) sequences for the model (single or multi-step,
#                     single or multi-variate, or both at once).

import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler


class DataDownloader:
    """
    Pulls a ticker's OHLCV data from yfinance for a chosen timeframe and
    caches it to disk as CSV so repeated runs don't re-hit the network.

    Timeframe: explicit start/end date, or default = last 3 years from today.
    end_date can never go beyond today.
    """

    def __init__(self, ticker: str, start_date: str | None = None,
                 end_date: str | None = None, cache_dir: str = "cache"):
        self.ticker = ticker
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        today = datetime.today()
        if end_date is None:
            end_dt = today
        else:
            end_dt = pd.to_datetime(end_date)
            if end_dt > today:
                raise ValueError("end_date cannot go beyond today.")

        if start_date is None:
            start_dt = end_dt - timedelta(days=365 * 3)
        else:
            start_dt = pd.to_datetime(start_date)

        if start_dt >= end_dt:
            raise ValueError("start_date must be strictly before end_date.")

        self.start_date = start_dt.strftime("%Y-%m-%d")
        self.end_date = end_dt.strftime("%Y-%m-%d")

    def _cache_path(self) -> str:
        return os.path.join(
            self.cache_dir, f"{self.ticker}_{self.start_date}_{self.end_date}.csv"
        )

    def download(self, force_download: bool = False) -> str:
        """
        Fetch data from yfinance, or reuse the cached CSV if it already exists.
        Returns the path to the CSV (downloaded or cached) for DataHandler to consume.
        """
        cache_path = self._cache_path()

        if os.path.exists(cache_path) and not force_download:
            print(f"[DataDownloader] Using cached data at {cache_path} ...")
            return cache_path

        print(f"[DataDownloader] Downloading {self.ticker} from Yahoo Finance "
              f"({self.start_date} -> {self.end_date}) ...")
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date,
                          auto_adjust=True)
        if df is None or df.empty:
            raise ValueError(f"No data returned for ticker '{self.ticker}'.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache_path)
        return cache_path


class DataHandler:
    """
    Feature-engineers a CSV of raw OHLCV data into model-ready train/test sets.

    Pipeline: load -> clean (NaN + weekend/holiday gaps) -> split -> scale
    (fit on train only, to avoid leakage) -> window into (X, y) sequences.

    Windowing supports:
      - multivariate  : feature_columns has more than one column.
      - multi-step (k): predict k future values of target_column instead of 1.
      - hybrid        : both of the above at once.
    """

    def __init__(self, csv_path: str, feature_columns: list[str] | None = None,
                 target_column: str = "Close"):
        self.csv_path = csv_path
        self.feature_columns = feature_columns  # None -> resolved to all columns on load
        self.target_column = target_column

        self.raw_df: pd.DataFrame | None = None
        self.clean_df: pd.DataFrame | None = None
        self.scalers: dict = {}
        self.raw_close_test: np.ndarray | None = None

    def load(self) -> pd.DataFrame:
        """Read the CSV produced by DataDownloader into a DatetimeIndex-ed DataFrame."""
        df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        self.raw_df = df
        if self.feature_columns is None:
            self.feature_columns = list(df.columns)
        return df

    def clean(self, nan_method: str = "forward_fill", fill_gaps: bool = True) -> pd.DataFrame:
        """
        Reindex over the full business-day calendar (exposing weekend/holiday
        gaps as NaN) and fill NaNs. fill_gaps=False skips the reindex step.
        """
        if self.raw_df is None:
            self.load()

        df = self.raw_df.copy()

        if fill_gaps:
            full_range = pd.bdate_range(df.index.min(), df.index.max())
            df = df.reindex(full_range)

        if nan_method == "forward_fill":
            df = df.ffill()
        elif nan_method == "drop":
            df = df.dropna()
        elif nan_method == "mean":
            df = df.fillna(df.mean())
        df = df.bfill()  # safety back-fill for any leading NaNs

        self.clean_df = df
        return df

    def _compute_target_log_return(self, df: pd.DataFrame) -> pd.Series:
        """
        Next-day target log return: y_t = log(C_{t+1} / C_t) = log(C_{t+1}) - log(C_t).

        Aligned to row t so it can be windowed like any other column: by the
        time X's window ends at row t, everything needed to know y_t (the
        move from t to t+1) is still in the future, so it's a valid target,
        not a leaked feature.
        """
        close = df["Close"]
        target = pd.Series(np.log(close.shift(-1) / close), index=df.index, name="Target_Log_Return")
        return target

    def _compute_intraday_log_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Log ratios of Open/High/Low relative to Close, same-day (no leakage: all four are known together at close)."""
        out = pd.DataFrame(index=df.index)
        out["Open_Close"] = np.log(df["Open"] / df["Close"])
        out["High_Close"] = np.log(df["High"] / df["Close"])
        out["Low_Close"] = np.log(df["Low"] / df["Close"])
        return out

    def _compute_interday_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Daily close-to-close log return: Log_Return_t = log(C_t / C_{t-1})."""
        out = pd.DataFrame(index=df.index)
        out["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
        return out

    def _compute_volume_log_change(self, df: pd.DataFrame) -> pd.Series:
        """Smoothed volume change: Vol_Change_t = log((V_t + 1) / (V_{t-1} + 1))."""
        volume = df["Volume"]
        vol_change = pd.Series(np.log((volume + 1) / (volume.shift(1) + 1)), index=df.index, name="Vol_Change")
        return vol_change

    def engineer_features(self) -> pd.DataFrame:
        """
        Replace self.clean_df's raw OHLCV columns with log-based features
        (Log_Return, Open_Close, High_Close, Low_Close, Vol_Change) and the
        Target_Log_Return target, dropping the NaN rows the shift()-based
        computations leave at the series' edges.

        Must run after clean() (which supplies the OHLCV df to transform)
        and before split()/scale()/make_windows(), which now operate on the
        engineered columns instead of raw OHLCV.
        """
        df = self.clean_df if self.clean_df is not None else self.clean()

        engineered_df = pd.concat([
            self._compute_interday_log_returns(df),
            self._compute_intraday_log_ratios(df),
            self._compute_volume_log_change(df),
            self._compute_target_log_return(df),
            df["Close"].rename("Raw_Close"),
        ], axis=1).dropna()

        self.clean_df = engineered_df
        self.feature_columns = ["Log_Return", "Open_Close", "High_Close", "Low_Close", "Vol_Change"]
        self.target_column = "Target_Log_Return"

        return engineered_df

    def split(self, split_method: str = "date", split_param: str | float = "2023-01-01"):
        """
        Split the cleaned data into train/test DataFrames.

        split_method : 'date' (split at a calendar date) or 'ratio' (0-1 float).
        split_param  : the date string or ratio matching split_method.
        """
        df = self.clean_df
        if df is None:
            df = self.clean()

        if split_method == "date":
            cutoff = pd.Timestamp(split_param)
            train_df = df[df.index < cutoff].copy()
            test_df = df[df.index >= cutoff].copy()
        elif split_method == "ratio":
            idx = int(len(df) * float(split_param))
            train_df = df.iloc[:idx].copy()
            test_df = df.iloc[idx:].copy()
        else:
            raise ValueError("split_method must be 'date' or 'ratio'.")

        return train_df, test_df

    def scale(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
              columns: list[str] | None = None):
        """
        MinMax-scale each column to [0, 1], fitting only on train_df to avoid
        leaking test-period statistics into the scaler. Scalers are stored in
        self.scalers so predictions can be inverse-transformed later.
        """
        columns = columns or list(train_df.columns)
        train_df, test_df = train_df.copy(), test_df.copy()

        self.scalers = {}
        for col in columns:
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_df[col] = scaler.fit_transform(train_df[[col]])
            if len(test_df) > 0:
                test_df[col] = scaler.transform(test_df[[col]])
            self.scalers[col] = scaler

        return train_df, test_df

    def make_windows(self, df: pd.DataFrame, window: int, k: int = 1,
                      feature_columns: list[str] | None = None,
                      target_column: str | None = None):
        """
        Slide a `window`-length lookback over df to build supervised-learning
        (X, y) pairs.

        window          : number of past trading days fed into the model.
        k               : number of future days to predict (k=1 -> single-step,
                           k>1 -> multi-step).
        feature_columns : columns used as model input (multiple -> multivariate).
        target_column   : column being predicted. May be one of feature_columns
                           (e.g. windowing "Close" both as input and target), or
                           a separate column entirely (e.g. engineer_features()'s
                           "Target_Log_Return", which isn't itself a model input).

        Shapes returned:
          X : (n_samples, window, n_features)
          y : (n_samples,)       if k == 1
              (n_samples, k)     if k > 1
        """
        feature_columns = feature_columns or self.feature_columns or list(df.columns)
        target_column = target_column or self.target_column

        values = df[feature_columns].values
        if target_column in feature_columns:
            target_values = values[:, feature_columns.index(target_column)]
        else:
            target_values = df[target_column].values

        X, y = [], []
        for i in range(window, len(values) - k + 1):
            X.append(values[i - window:i])
            y.append(target_values[i] if k == 1 else target_values[i:i + k])

        return np.array(X), np.array(y)

    def get_train_test(
        self,
        window: int,
        k: int = 1,
        split_method: str = "date",
        split_param: str | float = "2023-01-01",
        nan_method: str = "forward_fill",
        fill_gaps: bool = True,
        scale_columns: bool = True,
        use_log_features: bool = True,
        feature_columns: list[str] | None = None,
        target_column: str | None = None,
    ):
        """Full pipeline: clean -> [engineer_features] -> split -> scale ->

        window.

        Parameters:
            window: Number of past days for look-back sequence.
            k: Number of future steps to predict.
            split_method: 'date' or 'ratio'.
            split_param: Cutoff date string or train ratio float (e.g., 0.8).
            nan_method: Method to handle NaNs ('forward_fill', 'drop', 'mean').
            fill_gaps: Whether to reindex over full business-day calendar.
            scale_columns: Whether to MinMax-scale feature columns.
            use_log_features: If True, computes stationary log features &
              target.
            feature_columns: Explicit list of input features (optional).
            target_column: Explicit target column name (optional).

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        # 1. Clean raw OHLCV data
        self.clean(nan_method=nan_method, fill_gaps=fill_gaps)

        # 2. Feature engineering choice
        if use_log_features:
            self.engineer_features()
        else:
            if feature_columns is None:
                self.feature_columns = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]
            if target_column is None:
                self.target_column = "Close"

        # Override explicit feature/target if provided
        if feature_columns is not None:
            self.feature_columns = feature_columns
        if target_column is not None:
            self.target_column = target_column

        # 3. Split dataset into train and test DataFrames
        train_df, test_df = self.split(
            split_method=split_method, split_param=split_param
        )

        # 4. Scale features (Fit ONLY on train_df to prevent data leakage)
        if scale_columns:
            train_df, test_df = self.scale(
                train_df, test_df, columns=self.feature_columns
            )

        # 5. Prepend context window to test_df so test samples at boundary are not lost
        context_df = train_df.tail(window)
        extended_test_df = pd.concat([context_df, test_df])

        # 6. Preserve unscaled base Close prices (C_t) for test evaluation reconstruction
        if "Raw_Close" in test_df.columns:
            self.raw_close_test = test_df["Raw_Close"].values
        elif "Close" in test_df.columns:
            self.raw_close_test = test_df["Close"].values

        # 7. Generate 3D sequence windows (X, y)
        X_train, y_train = self.make_windows(
            train_df,
            window,
            k,
            self.feature_columns,
            self.target_column,
        )
        X_test, y_test = self.make_windows(
            extended_test_df,
            window,
            k,
            self.feature_columns,
            self.target_column,
        )

        return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    # Quick manual test of log-based feature engineering: python data_handler.py
    downloader = DataDownloader("APPL")
    csv_path = downloader.download()
    print(f"CSV ready at: {csv_path}")

    handler = DataHandler(csv_path)
    handler.clean()
    engineered_df = handler.engineer_features()

    (X_train, y_train), (X_test, y_test) = handler.get_train_test(
        window=60, k=1, split_method="ratio", split_param=0.8
    )

    feature_columns = handler.feature_columns or []
    target_column = handler.target_column
    print(f"feature_columns: {feature_columns}")
    print(f"target_column:   {target_column}")
    print("\nEngineered features + target (head):")
    print(engineered_df[[*feature_columns, target_column]].head())

    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    has_nan = (np.isnan(X_train).any() or np.isnan(y_train).any()
               or np.isnan(X_test).any() or np.isnan(y_test).any())
    print(f"\nNaNs present in X_train/y_train/X_test/y_test: {has_nan}")
    print(f"Scalers fitted for columns: {list(handler.scalers.keys())}")
