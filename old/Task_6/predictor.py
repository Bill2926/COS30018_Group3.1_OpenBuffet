# File: predictor.py
# Encapsulates the test-prediction and next-day-prediction pipelines.
# Separated from main.py so the maths can be unit-tested or reused without
# re-running the full training workflow.

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_test_inputs(
        train_prices: pd.Series,
        test_prices: pd.Series,
        scaler: MinMaxScaler,
        prediction_days: int
    ) -> np.ndarray:
    """
    Construct the scaled input array for the test period.

    We must prepend the last `prediction_days` rows of the training period to
    the test-period prices so the model has a full look-back window from the
    very first test day.  Without this, prediction would only start on test day
    number `prediction_days`, skipping the beginning of the test set.

    The scaler is fitted on training data only (see data_loader.py) and is
    re-used here so the normalisation is consistent.

    Parameters
    ----------
    train_prices    : Closing prices from the training split.
    test_prices     : Closing prices from the test split.
    scaler          : Already-fitted MinMaxScaler (fit on train data).
    prediction_days : Look-back window length (must match model input shape).

    Returns
    -------
    model_inputs : np.ndarray of shape (len(test)+prediction_days, 1), scaled.
    """
    total = pd.concat([train_prices, test_prices], axis=0)
    # Slice the tail of the combined series: we need exactly
    # (len(test) + prediction_days) rows to produce len(test) predictions.
    raw = total.values[len(total) - len(test_prices) - prediction_days:]
    raw = raw.reshape(-1, 1)
    # Transform using the TRAINING scaler — do NOT refit here (data leakage).
    return scaler.transform(raw)


def make_predictions(model, model_inputs: np.ndarray, prediction_days: int) -> np.ndarray:
    """
    Slide a window of length `prediction_days` over `model_inputs` and run
    the model on each window to produce one predicted scaled value per step.

    Returns
    -------
    x_test : np.ndarray of shape (n_predictions, prediction_days, 1).
    """
    x_test = []
    for i in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[i - prediction_days:i, 0])
    x_test = np.array(x_test)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    return x_test


def predict_next_day(model, model_inputs: np.ndarray, prediction_days: int,
                     scaler: MinMaxScaler) -> float:
    """
    Use the final `prediction_days` scaled values to forecast the next price.

    Returns the predicted price in original (non-scaled) units.
    """
    window = model_inputs[len(model_inputs) - prediction_days:, 0]
    window = window.reshape(1, prediction_days, 1)
    scaled_pred = model.predict(window, verbose=0)
    price = scaler.inverse_transform(scaled_pred)
    return float(price[0, 0])


# ── Task C.5: multivariate / multistep helpers ───────────────────────────────
# These generalise the three helpers above to >1 input feature and/or >1 output
# step. They share the same idea (slide a look-back window, invert the scaling)
# but keep every feature column instead of just Close.

def build_test_inputs_mv(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: list[str],
        scalers: dict,
        prediction_days: int
    ) -> np.ndarray:
    """
    Multivariate version of build_test_inputs().

    Prepend the last `prediction_days` rows of the training period to the test
    period (so the first test day has a full look-back window), then scale every
    feature column with its OWN fitted scaler (from data_loader's `scalers` dict,
    all fitted on training data only — no leakage).

    Returns a 2-D array (len(test)+prediction_days, n_features) of scaled values,
    with columns in the order of `feature_cols`.
    """
    combined = pd.concat([train_df[feature_cols], test_df[feature_cols]], axis=0)
    tail = combined.iloc[len(combined) - len(test_df) - prediction_days:]

    scaled = np.empty(tail.shape, dtype=float)
    for j, col in enumerate(feature_cols):
        # transform (never fit) so the test data uses the training statistics.
        scaled[:, j] = scalers[col].transform(tail[[col]].values).flatten()
    return scaled


def make_windows(model_inputs: np.ndarray, prediction_days: int) -> np.ndarray:
    """
    Multivariate version of make_predictions(): slide a look-back window over a
    2-D scaled array but keep ALL feature columns in each window.

    Returns x_test of shape (n_windows, prediction_days, n_features).
    """
    x_test = []
    for i in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[i - prediction_days:i, :])   # all features
    return np.array(x_test)


def inverse_close(scaled_pred: np.ndarray, close_scaler: MinMaxScaler) -> np.ndarray:
    """
    Invert MinMax scaling on Close predictions back to USD.

    Works for single-step (shape (n, 1)) and multistep (shape (n, k)) outputs by
    flattening to a column, inverting with the Close scaler, then restoring shape.
    """
    scaled_pred = np.asarray(scaled_pred)
    flat = scaled_pred.reshape(-1, 1)
    prices = close_scaler.inverse_transform(flat)
    return prices.reshape(scaled_pred.shape)


def forecast_future(model, last_window: np.ndarray, close_scaler: MinMaxScaler,
                    k_steps: int) -> np.ndarray:
    """
    Forecast the next `k_steps` closing prices in one shot (Task C.5 multistep).

    A model built with dense_units=k outputs all k future values from a single
    look-back window, so we just feed the most recent window and invert scaling.

    Parameters
    ----------
    last_window : scaled array (prediction_days, n_features) — the latest window.
    close_scaler: scaler fitted on the Close column (for inverse-transform).
    k_steps     : number of future days predicted (must match the model output).

    Returns
    -------
    np.ndarray of shape (k_steps,) — predicted closing prices in USD.
    """
    pred_days, n_features = last_window.shape
    window = last_window.reshape(1, pred_days, n_features)
    scaled_pred = model.predict(window, verbose=0)        # shape (1, k_steps)
    return inverse_close(scaled_pred, close_scaler).flatten()
