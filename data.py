# File: data.py
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# ── Module Level Configuration / Flags ────────────
INCLUDE_QQQ_CROSS_ASSET: bool = True  # Toggle QQQ feature integration
PREDICT_CROSS_ASSET: bool = True  # Toggle also predicting the cross-asset's own price (e.g. QQQ), on top of the primary target
DEFAULT_CROSS_ASSET_TICKER: str = "QQQ"


class DataDownloader:
    """Pulls stock OHLCV data from yfinance for one or multiple tickers and

    caches them locally as CSV files.
    """

    def __init__(
        self,
        tickers: str | list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        cache_dir: str = "cache",
    ):
        if isinstance(tickers, str):
            self.tickers = [tickers]
        else:
            self.tickers = tickers

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

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(
            self.cache_dir, f"{ticker}_{self.start_date}_{self.end_date}.csv"
        )

    def download_ticker(
        self, ticker: str, force_download: bool = False
    ) -> str:
        """Download or reuse cached CSV for a single ticker."""
        cache_path = self._cache_path(ticker)

        if os.path.exists(cache_path) and not force_download:
            print(f"[DataDownloader] Using cached data for {ticker} at {cache_path} ...")
            return cache_path

        print(
            f"[DataDownloader] Downloading {ticker} from Yahoo Finance "
            f"({self.start_date} -> {self.end_date}) ..."
        )
        df = yf.download(
            ticker, start=self.start_date, end=self.end_date, auto_adjust=True
        )
        if df is None or df.empty:
            raise ValueError(f"No data returned for ticker '{ticker}'.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(cache_path)
        return cache_path

    def download(self, force_download: bool = False) -> str | dict[str, str]:
        """Download all tickers.
        If single ticker, returns str. If list of tickers, returns dict {ticker:
        csv_path}.
        """
        paths = {}
        for t in self.tickers:
            paths[t] = self.download_ticker(t, force_download=force_download)

        if len(self.tickers) == 1:
            return paths[self.tickers[0]]
        return paths


class DataHandler:
    """
    Feature-engineers OHLCV data from single or multiple stock CSVs into model-ready train/test sets.
    """

    def __init__(
        self,
        csv_input: str | dict[str, str],
        target_ticker: str = "AAPL",
        feature_columns: list[str] | None = None,
        target_column: str | list[str] | None = None,
        include_cross_asset: bool = INCLUDE_QQQ_CROSS_ASSET,
        predict_cross_asset: bool = PREDICT_CROSS_ASSET,
    ):
        """Parameters:

        csv_input: Single CSV path string OR dictionary of {ticker: csv_path}.
        target_ticker: Primary ticker whose close price is targeted for
        prediction. include_cross_asset: Flag to toggle cross-asset features.
        predict_cross_asset: Flag to also predict the cross-asset ticker(s)'
        own next-day log return (multi-target output), on top of the primary
        target_ticker. Only takes effect when include_cross_asset is True and
        more than one ticker is loaded; ignored otherwise.
        """
        if isinstance(csv_input, str):
            self.csv_map = {target_ticker: csv_input}
        else:
            self.csv_map = csv_input

        self.target_ticker = target_ticker
        self.include_cross_asset = include_cross_asset
        self.predict_cross_asset = predict_cross_asset
        self.feature_columns = feature_columns
        self.target_column = target_column

        self.raw_df_map: dict[str, pd.DataFrame] = {}
        self.clean_df: pd.DataFrame | None = None
        self.scalers: dict = {}
        self.raw_close_test: np.ndarray | None = None  # primary target_ticker's raw close (backward-compat)
        self.raw_close_test_map: dict[str, np.ndarray] = {}  # ticker -> raw close, for every predicted ticker
        self.target_tickers: list[str] = [target_ticker]  # ticker each column of target_column corresponds to, in order

    def load(self) -> dict[str, pd.DataFrame]:
        """Load raw CSVs for all configured tickers into DataFrames."""
        for ticker, path in self.csv_map.items():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            self.raw_df_map[ticker] = df
        return self.raw_df_map

    def clean(
        self, nan_method: str = "forward_fill", fill_gaps: bool = True
    ) -> pd.DataFrame:
        """Clean each ticker's DataFrame and inner-join them on DatetimeIndex."""
        if not self.raw_df_map:
            self.load()

        cleaned_dfs = []
        for ticker, raw_df in self.raw_df_map.items():
            df = raw_df.copy()

            if fill_gaps:
                full_range = pd.bdate_range(df.index.min(), df.index.max())
                df = df.reindex(full_range)

            if nan_method == "forward_fill":
                df = df.ffill()
            elif nan_method == "drop":
                df = df.dropna()
            elif nan_method == "mean":
                df = df.fillna(df.mean())
            df = df.bfill()

            # Add prefix if multiple tickers are being loaded
            if len(self.raw_df_map) > 1 or self.include_cross_asset:
                df = df.add_prefix(f"{ticker}_")

            cleaned_dfs.append(df)

        # Inner join to keep overlapping trading dates across assets
        if len(cleaned_dfs) == 1:
            merged_df = cleaned_dfs[0]
        else:
            merged_df = pd.concat(cleaned_dfs, axis=1, join="inner")

        self.clean_df = merged_df
        return merged_df

    def _compute_ticker_features(
        self, df: pd.DataFrame, prefix: str = ""
    ) -> pd.DataFrame:
        """Compute log features for a specific ticker's OHLCV columns."""
        c_col = f"{prefix}Close"
        o_col = f"{prefix}Open"
        h_col = f"{prefix}High"
        l_col = f"{prefix}Low"
        v_col = f"{prefix}Volume"

        out = pd.DataFrame(index=df.index)
        out[f"{prefix}Log_Return"] = np.log(df[c_col] / df[c_col].shift(1))
        out[f"{prefix}Open_Close"] = np.log(df[o_col] / df[c_col])
        out[f"{prefix}High_Close"] = np.log(df[h_col] / df[c_col])
        out[f"{prefix}Low_Close"] = np.log(df[l_col] / df[c_col])
        out[f"{prefix}Vol_Change"] = np.log((df[v_col] + 1) / (df[v_col].shift(1) + 1))

        out[f"{prefix}Log_Dist_SMA20"] = self._compute_sma_log_dist(df, prefix=prefix, window=20)
        
        return out

    def _compute_sma_log_dist(
        self, df: pd.DataFrame, prefix: str = "", window: int = 20
    ) -> pd.Series:
        """Log ratio of Close relative to SMA: log(Close / SMA_window)."""
        c_col = f"{prefix}Close"
        sma = df[c_col].rolling(window=window).mean()
        return pd.Series(
            np.log(df[c_col] / sma),
            index=df.index,
            name=f"{prefix}Log_Dist_SMA{window}",
        )

    def engineer_features(self) -> pd.DataFrame:
        """Engineer log features for primary asset (and cross-asset if enabled),

        and set up the primary target column.
        """
        df = self.clean_df if self.clean_df is not None else self.clean()

        feature_dfs = []
        feature_cols = []

        is_multi = len(self.raw_df_map) > 1 or self.include_cross_asset

        # Process each ticker present in clean_df
        for ticker in self.raw_df_map.keys():
            prefix = f"{ticker}_" if is_multi else ""
            t_features = self._compute_ticker_features(df, prefix=prefix)
            feature_dfs.append(t_features)
            feature_cols.extend(t_features.columns.tolist())

        # Target Log Return (always computed for target_ticker)
        target_prefix = f"{self.target_ticker}_" if is_multi else ""
        target_close = df[f"{target_prefix}Close"]
        target_series = pd.Series(
            np.log(target_close.shift(-1) / target_close),
            index=df.index,
            name="Target_Log_Return",
        )

        # Base Close price preserved for price reconstruction
        raw_close_series = target_close.rename("Raw_Close")

        # Optionally also predict each cross-asset ticker's own next-day log
        # return, alongside the primary target_ticker (multi-target output).
        extra_target_dfs = []
        extra_raw_close_dfs = []
        target_columns: list[str] = ["Target_Log_Return"]
        target_tickers: list[str] = [self.target_ticker]
        if self.predict_cross_asset and is_multi:
            for ticker in self.raw_df_map.keys():
                if ticker == self.target_ticker:
                    continue
                cross_close = df[f"{ticker}_Close"]
                cross_target_name = f"{ticker}_Target_Log_Return"
                extra_target_dfs.append(pd.Series(
                    np.log(cross_close.shift(-1) / cross_close),
                    index=df.index,
                    name=cross_target_name,
                ))
                extra_raw_close_dfs.append(cross_close.rename(f"{ticker}_Raw_Close"))
                target_columns.append(cross_target_name)
                target_tickers.append(ticker)

        engineered_df = pd.concat(
            [*feature_dfs, target_series, *extra_target_dfs,
             raw_close_series, *extra_raw_close_dfs], axis=1
        ).dropna()

        self.clean_df = engineered_df
        self.feature_columns = feature_cols
        self.target_column = target_columns if len(target_columns) > 1 else target_columns[0]
        self.target_tickers = target_tickers

        return engineered_df

    def split(
        self, split_method: str = "date", split_param: str | float = "2023-01-01"
    ):
        """Split data into train/test DataFrames."""
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

    def scale(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        columns: list[str] | None = None,
    ):
        """MinMax-scale specified columns fitting strictly on train_df."""
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

    def make_windows(
        self,
        df: pd.DataFrame,
        window: int,
        k: int = 1,
        feature_columns: list[str] | None = None,
        target_column: str | list[str] | None = None,
    ):
        """
        Generate (X, y) sliding window sequences.

        target_column : a single column name (y is 1-D per sample, or 2-D
                         when k > 1), or a list of column names for
                         multi-target output (y gains a trailing per-target
                         axis; if k > 1 too, that 3-D result is flattened to
                         (n_samples, k * n_targets) to stay Dense-compatible).
        """
        feature_columns = feature_columns or self.feature_columns or list(df.columns)
        target_column = target_column if target_column is not None else self.target_column
        is_multi_target = isinstance(target_column, (list, tuple))
        target_cols = list(target_column) if is_multi_target else [target_column]

        values = df[feature_columns].values
        if not is_multi_target and target_column in feature_columns:
            target_values = values[:, feature_columns.index(target_column)]
        else:
            target_values = df[target_cols].values
            if not is_multi_target:
                target_values = target_values[:, 0]

        X, y = [], []
        for i in range(window, len(values) - k + 1):
            X.append(values[i - window : i])
            y.append(target_values[i] if k == 1 else target_values[i : i + k])

        X_arr, y_arr = np.array(X), np.array(y)
        if y_arr.ndim == 3:
            y_arr = y_arr.reshape(y_arr.shape[0], -1)
        return X_arr, y_arr

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
        target_column: str | list[str] | None = None,
    ):
        """Full pipeline execution."""
        self.clean(nan_method=nan_method, fill_gaps=fill_gaps)

        if use_log_features:
            self.engineer_features()
        else:
            if feature_columns is None:
                self.feature_columns = [
                    c for c in self.clean_df.columns if c != "Target_Log_Return"
                ]
            if target_column is None:
                self.target_column = f"{self.target_ticker}_Close" if self.include_cross_asset else "Close"

        if feature_columns is not None:
            self.feature_columns = feature_columns
        if target_column is not None:
            self.target_column = target_column

        train_df, test_df = self.split(
            split_method=split_method, split_param=split_param
        )

        if scale_columns:
            train_df, test_df = self.scale(
                train_df, test_df, columns=self.feature_columns
            )

        context_df = train_df.tail(window)
        extended_test_df = pd.concat([context_df, test_df])

        if "Raw_Close" in test_df.columns:
            self.raw_close_test = test_df["Raw_Close"].values
        elif f"{self.target_ticker}_Close" in test_df.columns:
            self.raw_close_test = test_df[f"{self.target_ticker}_Close"].values

        self.raw_close_test_map = {}
        for ticker in self.target_tickers:
            col = "Raw_Close" if ticker == self.target_ticker else f"{ticker}_Raw_Close"
            if col in test_df.columns:
                self.raw_close_test_map[ticker] = np.asarray(test_df[col].values)
        if self.raw_close_test is not None and self.target_ticker not in self.raw_close_test_map:
            self.raw_close_test_map[self.target_ticker] = self.raw_close_test

        X_train, y_train = self.make_windows(
            train_df, window, k, self.feature_columns, self.target_column
        )
        X_test, y_test = self.make_windows(
            extended_test_df, window, k, self.feature_columns, self.target_column
        )

        return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    TARGET_STOCK = "AAPL"
    CROSS_STOCK = "QQQ"

    tickers_to_download = [TARGET_STOCK]
    if INCLUDE_QQQ_CROSS_ASSET:
        tickers_to_download.append(CROSS_STOCK)

    downloader = DataDownloader(tickers_to_download)
    csv_paths = downloader.download()

    handler = DataHandler(
        csv_input=csv_paths,
        target_ticker=TARGET_STOCK,
        include_cross_asset=INCLUDE_QQQ_CROSS_ASSET,
        predict_cross_asset=PREDICT_CROSS_ASSET,
    )

    (X_train, y_train), (X_test, y_test) = handler.get_train_test(
        window=60, k=1, split_method="ratio", split_param=0.8, use_log_features=True
    )

    print(f"\n[DataHandler] Features ({len(handler.feature_columns)}): {handler.feature_columns}")
    print(f"[DataHandler] Target Column: {handler.target_column}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")