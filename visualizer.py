# File: visualizer.py
# Visualizer: renders financial charts (candlestick, moving-window boxplot)
# from a CSV of OHLCV data, and saves them as PNGs to plots/.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf


class Visualizer:
    """
    Renders financial charts from a CSV file of OHLCV stock data.

    Responsibilities (Task C.3):
      1. candlestick(csv, n) - render a candlestick chart where each candle
         aggregates n consecutive trading days (n >= 1).
      2. boxplot(csv, n)     - render a boxplot of a feature's distribution
         over a moving window of n consecutive trading days.

    Both methods save their output as a PNG under plots/.
    """

    def __init__(self, plots_dir: str = "plots"):
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

    @staticmethod
    def _load_csv(csv: str) -> pd.DataFrame:
        """Load a CSV of OHLCV data with a DatetimeIndex."""
        df = pd.read_csv(csv, index_col=0, parse_dates=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("CSV must have a date column usable as a DatetimeIndex.")
        return df

    @staticmethod
    def _aggregate_n_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
        """
        Merge every n_days consecutive trading-day rows into a single OHLCV row.

        Positional grouping (row-index // n_days) is used instead of calendar
        resampling because stock data only exists on trading days; a calendar
        resample would introduce empty weekend/holiday gaps.
        """
        if n_days <= 1:
            return df

        groups = np.arange(len(df)) // n_days

        agg_rules = {}
        for col in df.columns:
            low = col.lower()
            if low == "open":
                agg_rules[col] = "first"
            elif low == "high":
                agg_rules[col] = "max"
            elif low == "low":
                agg_rules[col] = "min"
            elif low == "close":
                agg_rules[col] = "last"
            elif low == "volume":
                agg_rules[col] = "sum"
            else:
                agg_rules[col] = "last"

        agg = df.groupby(groups).agg(agg_rules)
        last_dates = df.index.to_series().groupby(groups).last().values
        agg.index = pd.DatetimeIndex(last_dates)
        return agg

    def candlestick(self, csv: str, n_days: int = 1, title: str = "Candlestick Chart",
                     filename: str | None = None) -> str:
        """
        Render a candlestick chart where each candle covers n_days trading days.

        csv    : path to a CSV with Open/High/Low/Close (+ optional Volume) columns.
        n_days : number of trading days aggregated per candle (n_days >= 1).

        Returns the path of the saved PNG.
        """
        if n_days < 1:
            raise ValueError("n_days must be >= 1.")

        df = self._load_csv(csv)
        required = {"Open", "High", "Low", "Close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        plot_df = self._aggregate_n_days(df, n_days)
        has_volume = "Volume" in plot_df.columns

        if filename is None:
            base = os.path.splitext(os.path.basename(csv))[0]
            filename = f"{base}_candlestick_n{n_days}.png"
        save_path = os.path.join(self.plots_dir, filename)

        mpf.plot(
            plot_df,
            type="candle",
            style="charles",
            volume=has_volume,
            title=f"{title} ({n_days}-day candles)",
            ylabel="Price",
            savefig=save_path,
        )
        print(f"[Visualizer] Candlestick chart saved to {save_path}")
        return save_path

    def boxplot(self, csv: str, n_days: int = 20, column: str = "Close",
                step: int | None = None, title: str = "Boxplot Chart",
                filename: str | None = None) -> str:
        """
        Render one box-and-whisker plot per moving window of n_days trading days.

        csv    : path to a CSV with a `column` to analyse.
        n_days : length of each moving window (n_days >= 1).
        step   : how far the window slides between boxes; defaults to n_days
                 (non-overlapping windows). step=1 gives a fully sliding window.

        Returns the path of the saved PNG.
        """
        if n_days < 1:
            raise ValueError("n_days must be >= 1.")

        df = self._load_csv(csv)
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found; available: {list(df.columns)}")
        if step is None:
            step = n_days

        series = df[column]
        windows, labels = [], []
        for start in range(0, len(series) - n_days + 1, step):
            window = series.iloc[start:start + n_days]
            windows.append(window.values)
            labels.append(window.index[-1].strftime("%Y-%m-%d"))

        if not windows:
            raise ValueError(f"Not enough rows ({len(series)}) for a {n_days}-day window.")

        fig, ax = plt.subplots(figsize=(max(8, len(windows) * 0.4), 6))
        ax.boxplot(windows, tick_labels=labels)
        ax.set_title(f"{title}: {column} - {n_days}-day windows (step={step})")
        ax.set_xlabel(f"Window end date (each box = {n_days} trading days)")
        ax.set_ylabel(column)
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
        fig.tight_layout()

        if filename is None:
            base = os.path.splitext(os.path.basename(csv))[0]
            filename = f"{base}_boxplot_{column}_n{n_days}.png"
        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[Visualizer] Boxplot chart saved to {save_path}")
        return save_path


if __name__ == "__main__":
    # Quick manual test: python visualizer.py
    # Uses DataDownloader to make sure a cached CSV exists, then renders both charts.
    from data import DataDownloader

    downloader = DataDownloader("AAPL")
    csv_path = downloader.download()

    viz = Visualizer()
    viz.candlestick(csv_path, n_days=30, title="AAPL")
    # viz.boxplot(csv_path, n_days=20, column="Close", title="AAPL")
