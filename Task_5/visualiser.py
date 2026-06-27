# File: visualiser.py
# Task C.3: All chart-drawing code lives here so that main.py stays clean and
# the visualisation logic can be tested or swapped independently.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf


# ── Internal helper ──────────────────────────────────────────────────────────

def _aggregate_n_days(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """
    Merge every n_days consecutive *trading-day* rows into a single OHLCV row.

    We use positional grouping (np.arange // n_days) rather than calendar
    resampling (df.resample('5D')) because stock data only exists on trading
    days; a calendar resample would create empty weekend/holiday gaps that
    confuse mplfinance.
    """
    if n_days <= 1:
        return df

    groups = np.arange(len(df)) // n_days   # e.g. [0,0,0,1,1,1,2,…] for n=3

    agg_rules = {}
    for col in df.columns:
        low = col.lower()
        if   low == 'open':   agg_rules[col] = 'first'
        elif low == 'high':   agg_rules[col] = 'max'
        elif low == 'low':    agg_rules[col] = 'min'
        elif low == 'close':  agg_rules[col] = 'last'
        elif low == 'volume': agg_rules[col] = 'sum'
        else:                 agg_rules[col] = 'last'

    agg = df.groupby(groups).agg(agg_rules)
    # Use the *last* date of each group so the x-axis shows real calendar dates.
    last_dates = df.index.to_series().groupby(groups).last().values
    agg.index  = pd.DatetimeIndex(last_dates)
    return agg


# ── Public chart functions ───────────────────────────────────────────────────

def plot_candlestick_chart(
        df: pd.DataFrame,
        n_days: int = 1,
        title: str = 'Candlestick Chart',
        show: bool = True
    ):
    """
    Draw a candlestick chart where each candle covers n_days trading days.

    df must have Open/High/Low/Close columns and a DatetimeIndex.
    If Volume is present it is shown in a sub-panel below the candles.
    """
    if n_days < 1:
        raise ValueError("n_days must be >= 1.")

    required = {'Open', 'High', 'Low', 'Close'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    plot_df    = _aggregate_n_days(df, n_days)
    has_volume = 'Volume' in plot_df.columns

    # mpf.plot arguments:
    #   type='candle'  – classic candlestick (green body = up-day, red = down-day)
    #   style='charles'– a clean built-in colour theme
    #   volume=True    – add a volume bar panel underneath when data is available
    mpf.plot(
        plot_df,
        type='candle',
        style='charles',
        volume=has_volume,
        title=title,
        ylabel='Price',
        returnfig=True
    )
    if show:
        plt.show()


def plot_boxplot_chart(
        df: pd.DataFrame,
        column: str = 'Close',
        n_days: int = 20,
        step: int | None = None,
        title: str = 'Boxplot Chart',
        show: bool = True
    ):
    """
    Draw one box-and-whisker plot per rolling n_days window of `column`.

    step controls how far the window slides between boxes:
      step=n_days (default) → non-overlapping windows
      step=1                → fully sliding window (many overlapping boxes)

    Each box shows: median (orange line), IQR (box), 1.5×IQR whiskers, outliers.
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found; available: {list(df.columns)}")
    if n_days < 1:
        raise ValueError("n_days must be >= 1.")
    if step is None:
        step = n_days   # non-overlapping by default

    series  = df[column]
    windows, labels = [], []

    for start in range(0, len(series) - n_days + 1, step):
        window = series.iloc[start:start + n_days]
        windows.append(window.values)
        labels.append(window.index[-1].strftime('%Y-%m-%d'))

    if not windows:
        raise ValueError(f"Not enough rows ({len(series)}) for an {n_days}-day window.")

    fig, ax = plt.subplots(figsize=(max(8, len(windows) * 0.4), 6))
    ax.boxplot(windows, labels=labels)
    ax.set_title(f'{title}: {column} — {n_days}-day windows (step={step})')
    ax.set_xlabel(f'Window end date (each box = {n_days} trading days)')
    ax.set_ylabel(column)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── Prediction result plot ───────────────────────────────────────────────────

def plot_predictions(
        actual: np.ndarray,
        predicted: np.ndarray,
        company: str,
        show: bool = True
    ):
    """Plot actual vs predicted closing prices on the same axes."""
    plt.figure(figsize=(12, 5))
    plt.plot(actual,    color='black', label=f'Actual {company} Price')
    plt.plot(predicted, color='green', label=f'Predicted {company} Price')
    plt.title(f'{company} Share Price — Actual vs Predicted')
    plt.xlabel('Trading Days (test set)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


def plot_multistep_forecast(
        history: np.ndarray,
        forecast: np.ndarray,
        company: str,
        show: bool = True
    ):
    """
    Plot the recent actual closing prices followed by the k-step forecast that
    extends past the end of the history (Task C.5 multistep).

    history  : 1-D array of recent actual closing prices (USD).
    forecast : 1-D array of the next k predicted closing prices (USD).
    """
    history  = np.asarray(history).flatten()
    forecast = np.asarray(forecast).flatten()

    # x-positions: history occupies [0 .. len(history)-1]; the forecast continues
    # straight after it. We repeat the last history point so the two lines join.
    hist_x = np.arange(len(history))
    fc_x   = np.arange(len(history) - 1, len(history) + len(forecast))
    fc_y   = np.concatenate([history[-1:], forecast])

    plt.figure(figsize=(12, 5))
    plt.plot(hist_x, history, color='black', label=f'Recent actual {company} Close')
    plt.plot(fc_x, fc_y, color='red', marker='o',
             label=f'{len(forecast)}-day forecast')
    plt.title(f'{company} — {len(forecast)}-Day Closing Price Forecast')
    plt.xlabel('Trading Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()


# ── CLI helpers ──────────────────────────────────────────────────────────────

def ask_int(prompt: str, default: int) -> int:
    """Prompt for a whole number >= 1; blank input returns default."""
    while True:
        raw = input(prompt).strip()
        if not raw:
            return default
        try:
            value = int(raw)
            if value >= 1:
                return value
        except ValueError:
            pass
        print("Please enter a whole number >= 1.")


def ask_date(prompt: str, is_end: bool):
    """
    Accept a year ('2021'), year-month ('2021-03'), or full date ('2021-03-15').
    Returns a Timestamp (start or end of the period) or None for blank input.

    For an END bound we take the last moment of the period so '2021' covers all
    of 2021. For a START bound we take the first moment.
    """
    while True:
        raw = input(prompt).strip()
        if not raw:
            return None
        try:
            period = pd.Period(raw)
            return period.end_time if is_end else period.start_time
        except ValueError:
            print("Use YYYY, YYYY-MM, or YYYY-MM-DD.")


def slice_range(df: pd.DataFrame, start, end) -> pd.DataFrame:
    """Trim df to [start, end]; a None bound means 'no limit on that side'."""
    if start is not None:
        df = df[df.index >= start]
    if end is not None:
        df = df[df.index <= end]
    return df
