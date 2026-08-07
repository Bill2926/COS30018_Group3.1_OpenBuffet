# File: validation.py
# Task C.4 evaluation: loads a trained Pure DL model (+ its saved training history)
# and evaluates it against a held-out test set.
# Handles log-return predictions by inverse-transforming them to actual price values
# (C_{t+1} = C_t * exp(y_t)) for price-level evaluation and plotting.

import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

PLOTS_DIR = "plots"
RESULTS_DIR = "results"
HISTORY_DIR = "history"


class PureDLValidation:
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

    def accuracy_metrics(
        self, X_test: np.ndarray, y_test: np.ndarray,
        target_names: Optional[list] = None,
        raw_close_map: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, dict], Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]]]:
        """
        Evaluate model accuracy on both log returns and reconstructed prices,
        reported separately per prediction target (e.g. AAPL vs QQQ) instead
        of blended together.

        Parameters:
            X_test: Input sequence windows.
            y_test: Actual target log returns; 1-D (single target) or 2-D
                    (n_samples, n_targets) when predicting multiple assets.
            target_names: Label for each column of y_test, e.g. ["AAPL", "QQQ"].
                          Defaults to ["Target"] for a single target.
            raw_close_map: {target_name: unscaled Close price at time t},
                           used to reconstruct actual prices for that target.
                           A target with no entry only gets log-space metrics.

        Returns:
            (per_target metrics dict keyed by target name,
             per_target (c_true, c_pred) reconstructed price arrays)
        """
        y_pred_log = self.predict(X_test)
        y_true_log = np.asarray(y_test)
        y_pred_log = y_pred_log.reshape(y_true_log.shape)

        if y_true_log.ndim == 1:
            y_true_log = y_true_log[:, None]
            y_pred_log = y_pred_log[:, None]
        n_targets = y_true_log.shape[-1]

        if not target_names or len(target_names) != n_targets:
            target_names = ["Target"] if n_targets == 1 else [f"Target_{i}" for i in range(n_targets)]

        raw_close_map = raw_close_map or {}
        per_target_metrics: Dict[str, dict] = {}
        per_target_prices: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

        for i, name in enumerate(target_names):
            true_i = y_true_log[:, i]
            pred_i = y_pred_log[:, i]

            log_errors = pred_i - true_i
            log_rmse = float(np.sqrt(np.mean(log_errors ** 2)))
            log_mae = float(np.mean(np.abs(log_errors)))

            correct_direction = np.sign(pred_i) == np.sign(true_i)
            directional_accuracy = float(np.mean(correct_direction)) * 100

            metrics = {
                "Log_RMSE": log_rmse,
                "Log_MAE": log_mae,
                "Directional Accuracy (%)": directional_accuracy,
            }

            c_true, c_pred = None, None
            raw_close_today = raw_close_map.get(name)
            if raw_close_today is not None:
                raw_close_today = raw_close_today[:len(true_i)]
                c_true = self._reconstruct_price(true_i, raw_close_today)
                c_pred = self._reconstruct_price(pred_i, raw_close_today)

                price_errors = c_pred - c_true
                price_rmse = float(np.sqrt(np.mean(price_errors ** 2)))
                price_mae = float(np.mean(np.abs(price_errors)))
                nonzero = c_true != 0
                price_mape = float(np.mean(np.abs(price_errors[nonzero] / c_true[nonzero])) * 100) if nonzero.any() else float("nan")

                metrics.update({
                    "Price_RMSE": price_rmse,
                    "Price_MAE": price_mae,
                    "Price_MAPE (%)": price_mape,
                })

            print(f"[Test] Predictive accuracy ({name}):")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            per_target_metrics[name] = metrics
            per_target_prices[name] = (c_true, c_pred)

        return per_target_metrics, per_target_prices

    def plot_predictions(self, c_true: np.ndarray, c_pred: np.ndarray,
                         stock_name: str = "", filename: str = "predictions.png") -> str:
        """Plot actual vs predicted prices for one target over the test set."""
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(c_true, color="black", label="Actual Price")
        ax.plot(c_pred, color="green", label="Predicted Price")
        title = f"Actual vs Predicted Price - {stock_name} (Test Set)" if stock_name \
            else "Actual vs Predicted Price (Test Set)"
        ax.set_title(title)
        ax.set_xlabel("Test Sample")
        ax.set_ylabel("Price")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[Test] Prediction plot for {stock_name or 'target'} saved to {save_path}")
        return save_path

    def save_results(self, results: dict, filename: Optional[str] = None,
                      model_info: Optional[dict] = None) -> str:
        """Save evaluation results as timestamped JSON."""
        if filename is None:
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.json"
        save_path = os.path.join(self.results_dir, filename)

        payload = {
            "model_path": self.model_path,
            "timestamp": datetime.now().isoformat(),
            **(model_info or {}),
            **results,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Test] Results saved to {save_path}")
        return save_path

    def run(self, X_test: np.ndarray, y_test: np.ndarray,
            raw_close_today: np.ndarray | None = None,
            model_info: Optional[dict] = None,
            target_names: Optional[list] = None,
            raw_close_map: Optional[Dict[str, np.ndarray]] = None) -> dict:
        """
        Run full evaluation suite, reporting metrics and plots separately per
        prediction target (e.g. AAPL vs QQQ) when the model predicts more than one.

        model_info : optional extra fields to log alongside the results, e.g.
                     {"input_dim": (window, n_features), "output_dim": k_steps}.
        target_names : label per target column of y_test, e.g. ["AAPL", "QQQ"].
                       Defaults to ["Target"] for a single-target model.
        raw_close_map : {target_name: raw close array}, used for price
                        reconstruction/plots. raw_close_today is a shorthand
                        for the single-target case and is folded into this
                        map under the first target_names entry if given.
        """
        self.plot_loss_curve()
        convergence = self.convergence_stats()

        if raw_close_map is None and raw_close_today is not None:
            primary_name = (target_names or ["Target"])[0]
            raw_close_map = {primary_name: raw_close_today}

        per_target_metrics, per_target_prices = self.accuracy_metrics(
            X_test, y_test, target_names=target_names, raw_close_map=raw_close_map
        )

        for name, (c_true, c_pred) in per_target_prices.items():
            if c_true is not None and c_pred is not None:
                safe_name = name.replace(" ", "_")
                self.plot_predictions(c_true, c_pred, stock_name=name,
                                       filename=f"predictions_{safe_name}.png")

        results = {"convergence": convergence, "accuracy": per_target_metrics}
        self.save_results(results, model_info=model_info)
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

    tester = PureDLValidation(model_path=os.path.join("models", "LSTMModel.keras"))
    tester.run(X_test, y_test, raw_close_today=raw_close_test)