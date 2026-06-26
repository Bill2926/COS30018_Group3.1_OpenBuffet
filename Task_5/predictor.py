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
