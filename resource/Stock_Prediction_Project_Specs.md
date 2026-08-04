# Stock Price Prediction Project - Context & Requirements Manual (Tasks C.1 - C.7)

## Overview
This document serves as the primary system context and specification manual for an end-to-end AI/ML Stock Price Forecasting and Sentiment Analysis project built incrementally across seven structured milestone tasks (C.1 to C.7).

---

## Task C.1: Setup & Initial Codebase Evaluation
* **Goal**: Establish the local development environment, set up version control, and evaluate the initial codebase baseline (`v0.1`) against a reference GitHub repository (`P1`).

### Detailed Requirements
1. **Virtual Environment & Dependencies**:
   * Set up a clean Python virtual environment shared between `v0.1` (`stock-prediction.py`) and reference project `P1`.
   * Maintain a fully structured `requirements.txt` file detailing all mandatory packages and exact versions.
2. **Repository & Documentation Setup**:
   * Initialize a GitHub repository for the project and commit the baseline `v0.1` codebase.
   * Configure the project GitHub Wiki to act as the central documentation hub for all weekly progress reports and technical briefs.
3. **Execution & Comparative Evaluation**:
   * Run and train prediction models across both `v0.1` and `P1` environments.
   * Formulate formal evaluation metrics and criteria defining what constitutes a "better" stock price prediction (e.g., RMSE, MAE, directional accuracy, convergence speed).
   * Evaluate and compare performance metrics between `v0.1` and `P1`.

---

## Task C.2: Advanced Data Processing Pipeline (`v0.2`)
* **Goal**: Overhaul the naive data processing pipeline from `v0.1` by introducing robust feature engineering, scaling, missing data handling, and configurable splitting mechanisms inspired by `P1`.

### Detailed Requirements
1. **Multi-Feature Data Ingestion**:
   * Build a modular data loader supporting arbitrary historical date ranges via explicit `start_date` and `end_date` parameters.
   * Extract and process full multi-feature sets (e.g., `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`).
2. **Data Cleaning & Local Persistence**:
   * Implement automated NaN detection, interpolation, and handling strategies.
   * Provide caching mechanisms to save downloaded raw/processed datasets locally to reduce API bandwidth and latency on subsequent runs.
3. **Feature Scaling & Management**:
   * Implement configurable feature scalers (e.g., `MinMaxScaler`, `StandardScaler`).
   * Store stateful scaler objects inside a dedicated data structure to ensure proper reverse transformation (inverse scaling) during post-processing and evaluation.
4. **Flexible Train/Test Splitting**:
   * Support multiple train/test splitting criteria: custom ratio-based split, chronological date-based split, and randomized split.
5. **Code Annotations**:
   * Fully comment and explain complex code logic, including underlying mathematical operations and pandas transformations.

---

## Task C.3: Financial Data Visualization (`v0.3`)
* **Goal**: Integrate financial data visualization tools into the pipeline using `mplfinance` or equivalent libraries to enable technical chart analysis.

### Detailed Requirements
1. **Custom Candlestick Chart Function**:
   * Develop a utility function to render candlestick charts from financial dataframes.
   * Include a dynamic aggregation parameter $n$ allowing each candlestick to represent $n$ consecutive trading days ($n \ge 1$).
2. **Moving Window Boxplot Function**:
   * Implement a boxplot visualization function designed to analyze feature distributions over a moving window of $n$ consecutive trading days.
3. **Parameter Documentation**:
   * Provide full inline documentation explaining function arguments, rendering options, and design decisions.

---

## Task C.4: Dynamic Deep Learning Architecture (`v0.4`)
* **Goal**: Replace manually hardcoded neural network definitions with a dynamic model factory supporting multiple deep learning architectures and hyperparameter tuning.

### Detailed Requirements
1. **Dynamic Model Factory**:
   * Construct a dynamic model-building function accepting architectural hyperparameters, including:
     * Number of layers.
     * Hidden unit size per layer.
     * Layer type specifications (e.g., `LSTM`, `RNN`, `GRU`).
     * Dropout rates and activation function overrides.
2. **Hyperparameter Experimentation**:
   * Conduct systematic experiments across architectural configurations (`LSTM` vs. `RNN` vs. `GRU`) and hyperparameter sets (layer depth, hidden dimensions, learning rate, batch size, epochs).
   * Document loss curves, convergence rates, and predictive accuracy for all experimental runs.

---

## Task C.5: Multivariate & Multistep Forecasting (`v0.5`)
* **Goal**: Extend time-series forecasting capability beyond single-variable single-day prediction to multi-variable and multi-horizon forecasting.

### Detailed Requirements
1. **Multistep Forecasting**:
   * Implement a multi-step sequence prediction function predicting price sequences over $k$ future trading days ($k > 1$).
2. **Multivariate Forecasting**:
   * Implement a multi-variable prediction function leveraging multiple input features (`Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`) to predict future target values.
3. **Combined Multivariate Multistep Architecture**:
   * Unify the two approaches to build an integrated multivariate, multi-step forecasting model predicting $k$-day future stock trajectories based on multi-dimensional historical inputs.

---

## Task C.6: Hybrid Ensemble Modeling
* **Goal**: Develop an ensemble architecture combining statistical time-series models with deep learning and machine learning algorithms to enhance predictive performance.

### Detailed Requirements
1. **Hybrid Architecture Construction**:
   * Build an ensemble framework integrating classical statistical models (e.g., `ARIMA` / `SARIMA`) with deep learning architectures (e.g., `LSTM`, `GRU`) or tree ensembles (e.g., `Random Forest`).
2. **Ensemble Weighting & Fusion Strategies**:
   * Implement model aggregation schemes (e.g., simple average, weighted blending, meta-learner stacking).
3. **Comparative Analysis**:
   * Benchmark the ensemble against standalone single-model baselines across multiple evaluation metrics.

---

## Task C.7: Extension - Sentiment-Driven Movement Classification
* **Goal**: Transform the system into an end-to-end multi-modal binary classification pipeline predicting whether next-day stock closing prices will rise or fall by integrating textual sentiment data.

### Detailed Requirements
1. **Data Collection & Alignment**:
   * Ingest external financial text data (e.g., news headlines, financial media, social media feeds) via APIs or scrapers.
   * Clean textual data and temporally align text timestamps with daily stock trading calendars.
2. **Sentiment Extraction**:
   * Apply NLP sentiment tools (e.g., `VADER`, `FinBERT`, transformer pipelines) to compute daily sentiment scores.
   * Aggregate sentiment scores at the daily level to match numerical stock price frequencies.
3. **Feature Engineering & Binary Classification**:
   * Construct hybrid feature vectors combining price technical indicators, lagged returns, and daily sentiment features.
   * Train a binary classification model (e.g., predicting 1 for Price Rise, 0 for Price Fall).
4. **Performance Evaluation**:
   * Evaluate the model using `Accuracy`, `Precision`, `Recall`, `F1-Score`, and a `Confusion Matrix`.
   * Perform an ablation study comparing the multimodal sentiment-enabled model against a baseline model operating strictly on numerical market data.
5. **Independent Research Enhancement**:
   * Integrate at least one advanced enhancement (e.g., domain-specific financial NLP models, alternative data feeds, attention-based multimodal fusion architectures).