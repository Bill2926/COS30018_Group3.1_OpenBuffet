# OpenBuffet: Stock Price Prediction System

**Course**: COS30018 - Intelligent Systems
**Project Name**: OpenBuffet
**Group**: HN - Group 3.1
**Option Selection**: Option C - Stock Price Prediction System

## Lecturer and Team Members
* **Lecturer**: Nguyen Manh Toan
* **Nguyen Duc Manh** (Project Lead)
* **Do Trinh Thuan Minh**
* **Nguyen Hong Minh**
* **Do Minh Thanh**

## Project Description
OpenBuffet is a stock price prediction system built around two forecasting
pipelines that share the same feature-engineering backbone:

1. **Pure DL** - an LSTM/GRU/RNN trained end-to-end on engineered log-return
   features.
2. **Hybrid** - a VARMAX linear component that captures the autocorrelated
   structure of the target series, plus a GRU/LSTM/RNN residual model that
   learns the non-linear part VARMAX misses (`Y_hat_final = Y_hat_linear +
   e_hat`).

Both pipelines can optionally be enriched with cross-asset features (QQQ),
FinBERT-derived news sentiment, a rolling Topological Data Analysis
(persistent entropy) feature, and the CBOE VIX index, and both are evaluated
with the same suite of log-space and price-space metrics (RMSE, MAE, MAPE,
directional accuracy) plus loss curves and prediction plots.

An `ablation.py` automation script drives `main.py` programmatically across
9 pre-configured benchmark/ablation scenarios and consolidates the results
into a single comparison CSV - see [Running the Ablation Study](#running-the-ablation-study-ablationpy).

## Technical Stack

* **Language**: Python **3.11** (see `.python-version`)
* **Package/environment manager**: [uv](https://docs.astral.sh/uv/) (recommended - see [Setup](#setup))

### Key libraries (pinned versions, from `pyproject.toml` / `uv.lock`)

| Library | Version | Used for |
|---|---|---|
| `tensorflow` | 2.21.0 | LSTM/GRU/RNN models (`models/dl_model.py`) |
| `keras` | 3.14.1 | TensorFlow's model/layer API |
| `torch` | >=2.13.0 | Backend for the FinBERT sentiment model |
| `transformers` | >=5.14.1 | FinBERT (`ProsusAI/finbert`) sentiment scoring (`sentiments/finBERT.py`) |
| `statsmodels` | >=0.14.6 | VARMAX (Hybrid linear component) and its SARIMAX single-asset fallback (`models/stats_model.py`) |
| `ripser` | 0.6.15 | Persistent homology / TDA rolling-entropy feature (`tda.py`) |
| `scikit-learn` | 1.8.0 | `MinMaxScaler` feature scaling (`data.py`) |
| `numpy` | 2.4.6 | Numerical arrays throughout |
| `pandas` | 3.0.3 | Data wrangling / feature engineering |
| `yfinance` | 1.4.1 | Historical OHLCV download (`data.py`) |
| `matplotlib` / `mplfinance` | 3.10.9 / >=0.12.10b0 | Plots (loss curves, predictions, candlesticks) |

> `torch` and `transformers` are only exercised if you regenerate the
> sentiment CSV via `sentiments/finBERT.py`; `main.py`/`ablation.py`
> normally just read the pre-computed
> `sentiments/aapl_sen_23-07-2012_27-01-2020.csv`.

## Project Structure

```
OpenBuffet/
├── main.py                  # Pipeline entry point (Pure DL or Hybrid), all config toggles live here
├── ablation.py               # Runs main.py across 9 scenarios, isolates outputs, exports a summary CSV
├── data.py                   # DataDownloader (yfinance + cache) and DataHandler (clean/engineer/split/scale/window)
├── tda.py                    # Rolling Persistent Entropy (TDA) feature extraction (ripser)
├── pure_dl_validation.py     # Evaluation suite for the Pure DL pipeline (metrics, loss curve, prediction plots)
├── hybrid_validation.py      # Evaluation suite for the Hybrid pipeline (metrics, loss curve, decomposition plots)
├── visualizer.py             # Standalone candlestick / boxplot chart renderer from an OHLCV CSV
├── test.py                   # Ad-hoc TDA/persistent-entropy experimentation script
│
├── models/                   # Model architectures (gitignored - not saved model weights, just source)
│   ├── dl_model.py           #   DLModel ABC, LSTM/GRU/RNN models, HybridModel, ModelFactory
│   └── stats_model.py        #   StatsModel ABC, ARIMAModel, VARIMAXModel (VARMAX + SARIMAX fallback)
│
├── sentiments/                # News-sentiment inputs/outputs
│   ├── aapl_news_yahoo.csv    #   Raw scraped news headlines
│   ├── aapl_sen_...csv        #   Pre-computed daily FinBERT sentiment scores (consumed by main.py)
│   └── finBERT.py             #   Script that (re)generates the sentiment CSV from raw news via FinBERT
│
├── resource/                  # Project spec / reference material
│
├── cache/                     # Cached yfinance OHLCV downloads, keyed by ticker + date range (gitignored)
├── trained_models/            # Saved .keras models (+ VARMAX pickles for Hybrid), one file per architecture (gitignored)
├── history/                   # Per-model training history JSON (loss/val_loss per epoch, gitignored)
├── results/                   # Default results.json output of a single `python main.py` run (gitignored)
├── plots/                     # Default plots/ output of a single `python main.py` run (gitignored)
├── experiments_output/        # ablation.py output: one isolated subfolder per scenario + the summary CSV (gitignored)
│
├── pyproject.toml / uv.lock   # uv-managed dependency declaration/lockfile
├── requirements.txt           # pip-style dependency list (alternative to uv)
└── .python-version             # Pins the project to Python 3.11
```

## Setup

This project targets **Python 3.11** and is managed with **uv**. Install uv
first if you don't have it ([instructions](https://docs.astral.sh/uv/getting-started/installation/)),
then from the project root:

```bash
# uv reads .python-version and pyproject.toml, and creates/updates .venv
uv sync
```

That installs every pinned dependency (TensorFlow, PyTorch, statsmodels,
ripser, etc.) into a local `.venv`. You don't need to manually activate the
venv - just prefix commands with `uv run`, e.g. `uv run main.py`. If you
prefer an activated shell:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Windows (Git Bash) / macOS / Linux
source .venv/Scripts/activate   # or .venv/bin/activate on macOS/Linux
```

<details>
<summary>Alternative: plain pip</summary>

```bash
python3.11 -m venv .venv
# activate as above, then:
pip install -r requirements.txt
```
</details>

## Running the Main Pipeline (`main.py`)

All configuration (ticker, date range, model architecture, feature toggles,
epochs, etc.) lives as module-level constants near the top of `main.py` -
edit them directly to change a run's setup. The only thing passed on the
command line is which pipeline to run:

```bash
# Pure DL pipeline (default, --mode 1): trains DL_TYPE (LSTM/GRU/RNN) directly on features
uv run main.py --mode 1

# Hybrid pipeline (--mode 2): VARMAX linear component + DL residual model
uv run main.py --mode 2
```

Key toggles in `main.py` you'll typically adjust:

| Constant | Meaning |
|---|---|
| `DL_TYPE` | `"LSTM"`, `"GRU"`, or `"RNN"` |
| `INCLUDE_CROSS_ASSET` | Add QQQ as an input feature alongside AAPL |
| `PREDICT_QQQ_PRICE` | Also predict QQQ's own next-day return (multi-target output) |
| `USE_SENTIMENT` | Add the FinBERT daily sentiment score as a feature |
| `USE_TDA` | Add the rolling Persistent Entropy (TDA) feature |
| `USE_VIX` | Add the CBOE VIX daily close as a feature |
| `EPOCHS`, `BATCH_SIZE` | DL training hyperparameters |
| `VARIMAX_ORDER` | `(p, q)` order for the Hybrid pipeline's VARMAX component |

Each run downloads (or reuses cached) OHLCV data into `cache/`, trains the
selected model, saves it to `trained_models/`, saves training history to
`history/`, then runs the corresponding validation suite
(`PureDLValidation` or `HybridValidation`), which writes metrics to
`results/` and plots (loss curve, actual-vs-predicted, and for Hybrid also
linear/residual decomposition) to `plots/`.

> Note: with `INCLUDE_CROSS_ASSET=False` and `PREDICT_QQQ_PRICE=False` (a
> single-asset target), the Hybrid pipeline automatically falls back from
> multivariate VARMAX to a single-series SARIMAX with the same `(p, q)`
> order internally, since VARMAX itself is undefined for one variable - no
> extra configuration needed.

## Running the Ablation Study (`ablation.py`)

`ablation.py` automates the full benchmark/ablation workflow: it imports
`main.py` and calls `run_pure_dl()` / `run_hybrid()` directly (no
subprocesses), overriding the relevant config toggles for each of 9
predefined scenarios, and isolating every run's outputs so nothing gets
overwritten between runs.

```bash
uv run ablation.py
```

**Scenarios executed** (in order):

| # | scenario_name | Mode | Notes |
|---|---|---|---|
| 1 | `01_pure_dl_rnn` | Pure DL, RNN | Full feature set (cross-asset, sentiment, TDA, VIX) |
| 2 | `02_pure_dl_lstm` | Pure DL, LSTM | Full feature set |
| 3 | `03_pure_dl_gru` | Pure DL, GRU | Full feature set |
| 4 | `04_hybrid_varmax_gru` | Hybrid (VARMAX+GRU) | Full feature set |
| 5 | `05_ablation_pure_price` | Hybrid (VARMAX+GRU) | Price features only (no sentiment/TDA/VIX) |
| 6 | `06_ablation_with_sentiment` | Hybrid (VARMAX+GRU) | + sentiment only |
| 7 | `07_ablation_with_tda` | Hybrid (VARMAX+GRU) | + TDA only |
| 8 | `08_ablation_with_vix` | Hybrid (VARMAX+GRU) | + VIX only |
| 9 | `09_single_asset_aapl_only` | Hybrid (VARMAX+GRU) | AAPL only, no cross-asset (SARIMAX fallback) |

**Outputs**, isolated per scenario under `experiments_output/`:

```
experiments_output/
├── <scenario_name>/
│   ├── results.json     # convergence stats + per-target (AAPL, QQQ) accuracy metrics
│   └── plots/            # loss curve + prediction/decomposition plots for this scenario only
├── ...                   # one folder per scenario
└── benchmark_summary_results.csv   # consolidated comparison across all 9 scenarios
```

`benchmark_summary_results.csv` has one row per scenario (metrics for the
primary AAPL target) with columns: `scenario_name`, `mode`, `model_type`,
`include_cross_asset`, `use_sentiment`, `use_tda`, `use_vix`, `Log_RMSE`,
`Log_MAE`, `Price_RMSE`, `Price_MAE`, `Price_MAPE_percent`,
`Directional_Accuracy_percent`.

Between scenarios, `ablation.py` calls `tf.keras.backend.clear_session()`
and `gc.collect()` to avoid memory buildup across 9 consecutive training
runs, and prints `[i/9] Running <scenario>...` progress as it goes. A full
9-scenario pass (50 epochs each) takes roughly 15-20 minutes on CPU.

## Other Scripts

* `visualizer.py` - renders candlestick and moving-window boxplot charts
  from any OHLCV CSV (independent of the prediction pipelines).
* `sentiments/finBERT.py` - regenerates the sentiment CSV from
  `sentiments/aapl_news_yahoo.csv` using FinBERT; only needed if you want
  to refresh sentiment data (the pre-computed CSV is already checked in).
* `test.py` - scratch script for experimenting with the TDA/persistent
  entropy computation in isolation.
