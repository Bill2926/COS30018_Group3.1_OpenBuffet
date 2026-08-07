# File: test.py
# Task C.4 evaluation: loads a trained Pure DL model (+ its saved training history)
# and evaluates it against a held-out test set.
# Handles log-return predictions by inverse-transforming them to actual price values
# (C_{t+1} = C_t * exp(y_t)) for price-level evaluation and plotting.

import json
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

PLOTS_DIR = "plots"
RESULTS_DIR = "results"
HISTORY_DIR = "history"


class Validation:
    """
    Evaluates an already-trained Keras model against a test set.

    Responsibilities:
      1. Loss curves: plot training/validation loss per epoch.
      2. Convergence rate: summarize loss reduction speed and plateau epoch.
      3. Predictive accuracy: calculate log-space metrics, price-space metrics,
         and directional accuracy.
      4. Visualizations: plot actual vs predicted prices on the test set.
    """

    def __init__(self, model_path: str, history_path: Optional[str] = None,
                 plots_dir: str = PLOTS_DIR, results_dir: str = RESULTS_DIR):
        self.model_path = model_path
        self.model = load_model(model_path)

        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        if history_path is None:
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            history_path = os.path.join(HISTORY_DIR, f"{model_name}_history.json")
        self.history = self._load_history(history_path)

    @staticmethod
    def _load_history(history_path: str) -> dict:
        if os.path.exists(history_path):
            with open(history_path) as f:
                return json.load(f)
        print(f"[Test] No history file found at {history_path}; "
              "loss-curve / convergence stats will be skipped.")
        return {}

    def plot_loss_curve(self, filename: str = "loss_curve.png") -> Optional[str]:
        """Plot training (and validation, if present) loss per epoch."""
        if "loss" not in self.history:
            print("[Test] No training history available; skipping loss curve.")
            return None

        loss = self.history["loss"]
        val_loss = self.history.get("val_loss")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(loss, label="Training Loss")
        if val_loss:
            ax.plot(val_loss, label="Validation Loss")
        ax.set_title("Loss Curve (Log Return Space)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[Test] Loss curve saved to {save_path}")
        return save_path

    def convergence_stats(self, tolerance: float = 0.05) -> dict:
        """
        Summarize how the loss converged during training.
        """
        if "loss" not in self.history:
            print("[Test] No training history available; skipping convergence stats.")
            return {}

        loss = np.array(self.history["loss"])
        initial, final = float(loss[0]), float(loss[-1])
        improvement_rate = (initial - final) / initial if initial != 0 else 0.0

        threshold = final + tolerance * abs(final)
        converged_epoch = next((i + 1 for i, l in enumerate(loss) if l <= threshold), len(loss))

        stats = {
            "initial_loss": initial,
            "final_loss": final,
            "improvement_rate": improvement_rate,
            "epochs_to_converge": converged_epoch,
            "total_epochs": len(loss),
        }
        print("[Test] Convergence stats:")
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        return stats

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Generate model predictions on sequence inputs."""
        return self.model.predict(X_test)

    def _reconstruct_price(self, y_log: np.ndarray, raw_close_today: np.ndarray) -> np.ndarray:
        """
        Reconstruct actual price from log return: C_{t+1} = C_t * exp(y_t).
        """
        return raw_close_today * np.exp(y_log)

    def accuracy_metrics(self, X_test: np.ndarray, y_test: np.ndarray,
                         raw_close_today: np.ndarray | None = None) -> Tuple[dict, np.ndarray | None, np.ndarray | None]:
        """
        Evaluate model accuracy on both log returns and reconstructed prices.

        Parameters:
            X_test: Input sequence windows.
            y_test: Actual target log returns.
            raw_close_today: Unscaled Close price at time t (last step in input window).

        Returns:
            Tuple of (metrics dict, reconstructed actual prices, reconstructed predicted prices).
        """
        y_pred_log = self.predict(X_test)
        y_true_log = np.asarray(y_test)
        y_pred_log = y_pred_log.reshape(y_true_log.shape)

        # 1. Log-space metrics
        log_errors = y_pred_log - y_true_log
        log_rmse = float(np.sqrt(np.mean(log_errors ** 2)))
        log_mae = float(np.mean(np.abs(log_errors)))

        # 2. Directional Accuracy (Sign matching on log return)
        # y_t > 0 means price increase, y_t < 0 means price decrease
        first_step_true = y_true_log[:, 0] if y_true_log.ndim > 1 else y_true_log
        first_step_pred = y_pred_log[:, 0] if y_pred_log.ndim > 1 else y_pred_log
        correct_direction = np.sign(first_step_pred) == np.sign(first_step_true)
        directional_accuracy = float(np.mean(correct_direction)) * 100

        metrics = {
            "Log_RMSE": log_rmse,
            "Log_MAE": log_mae,
            "Directional Accuracy (%)": directional_accuracy,
        }

        # 3. Price-space metrics (if raw_close_today is provided)
        c_true_next, c_pred_next = None, None
        if raw_close_today is not None:
            raw_close_today = raw_close_today[:len(first_step_true)]
            c_true_next = self._reconstruct_price(first_step_true, raw_close_today)
            c_pred_next = self._reconstruct_price(first_step_pred, raw_close_today)

            price_errors = c_pred_next - c_true_next
            price_rmse = float(np.sqrt(np.mean(price_errors ** 2)))
            price_mae = float(np.mean(np.abs(price_errors)))
            nonzero = c_true_next != 0
            price_mape = float(np.mean(np.abs(price_errors[nonzero] / c_true_next[nonzero])) * 100) if nonzero.any() else float("nan")

            metrics.update({
                "Price_RMSE": price_rmse,
                "Price_MAE": price_mae,
                "Price_MAPE (%)": price_mape,
            })

        print("[Test] Predictive accuracy:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return metrics, c_true_next, c_pred_next

    def plot_predictions(self, c_true: np.ndarray, c_pred: np.ndarray,
                         filename: str = "predictions.png") -> str:
        """Plot actual vs predicted prices over the test set."""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(c_true, color="black", label="Actual Price")
        ax.plot(c_pred, color="green", label="Predicted Price")
        ax.set_title("Actual vs Predicted Price (Test Set)")
        ax.set_xlabel("Test Sample")
        ax.set_ylabel("Price")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[Test] Prediction plot saved to {save_path}")
        return save_path

    def save_results(self, results: dict, filename: Optional[str] = None) -> str:
        """Save evaluation results as timestamped JSON."""
        if filename is None:
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.json"
        save_path = os.path.join(self.results_dir, filename)

        payload = {
            "model_path": self.model_path,
            "timestamp": datetime.now().isoformat(),
            **results,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Test] Results saved to {save_path}")
        return save_path

    def run(self, X_test: np.ndarray, y_test: np.ndarray,
            raw_close_today: np.ndarray | None = None) -> dict:
        """Run full evaluation suite."""
        self.plot_loss_curve()
        convergence = self.convergence_stats()
        metrics, c_true, c_pred = self.accuracy_metrics(X_test, y_test, raw_close_today)

        if c_true is not None and c_pred is not None:
            self.plot_predictions(c_true, c_pred)

        results = {"convergence": convergence, "accuracy": metrics}
        self.save_results(results)
        return results


if __name__ == "__main__":
    # Quick manual test execution
    from data import DataDownloader, DataHandler

    TICKER = "AAPL"
    PREDICTION_DAYS = 60

    downloader = DataDownloader(TICKER)
    csv_path = downloader.download()

    handler = DataHandler(csv_path)
    (_, _), (X_test, y_test) = handler.get_train_test(
        window=PREDICTION_DAYS, k=1, split_method="ratio", split_param=0.8,
        use_log_features=True
    )

    # Retrieve raw close price array for price reconstruction
    raw_close_test = handler.raw_close_test

    tester = Validation(model_path=os.path.join("models", "LSTMModel.keras"))
    tester.run(X_test, y_test, raw_close_today=raw_close_test)