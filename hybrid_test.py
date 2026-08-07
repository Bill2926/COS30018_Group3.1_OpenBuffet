# File: hybrid_test.py
# Task C.6 evaluation: loads a trained HybridModel (ARIMA + DL residual) and
# evaluates it against a held-out test set: loss curve (of the residual DL
# sub-model), convergence rate, and predictive accuracy of the reconstructed
# price Y_hat_final = Y_hat_linear + e_hat. Mirrors test.py's Test class, but
# adapted for Hybrid's raw-series train/evaluate signature and ARIMA + DL
# composition (so it isn't a plain Keras .predict(X) like pure DL models).

import json
import os
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from models.dl_model import HybridModel

PLOTS_DIR = "plots"
RESULTS_DIR = "results"
HISTORY_DIR = "history"


class HybridTest:
    """
    Evaluates an already-trained HybridModel against a test set.

    Responsibilities (mirrors Test in test.py, adapted for Hybrid):
      1. Loss curve         - training/validation loss of the DL residual
                               sub-model, from its saved history JSON.
      2. Convergence rate    - how fast/how much that loss dropped.
      3. Predictive accuracy - RMSE, MAE, MAPE, directional accuracy on the
                               reconstructed price Y_hat_final, plus plots.
    """

    def __init__(self, model_path: str, input_shape: Tuple[int, int],
                 arima_order: Tuple[int, int, int] = (5, 1, 0), dl_type: str = "lstm",
                 history_path: Optional[str] = None, plots_dir: str = PLOTS_DIR,
                 results_dir: str = RESULTS_DIR):
        self.model_path = model_path
        self.hybrid = HybridModel(input_shape=input_shape, output_dim=1,
                                   arima_order=arima_order, dl_type=dl_type)
        self.hybrid.load(model_path)

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
        print(f"[HybridTest] No history file found at {history_path}; "
              "loss-curve / convergence stats will be skipped.")
        return {}

    def plot_loss_curve(self, filename: str = "hybrid_loss_curve.png") -> Optional[str]:
        """Plot training (and validation, if present) residual-loss per epoch."""
        if "loss" not in self.history:
            print("[HybridTest] No training history available; skipping loss curve.")
            return None

        loss = self.history["loss"]
        val_loss = self.history.get("val_loss")

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(loss, label="Training Loss (residual)")
        if val_loss:
            ax.plot(val_loss, label="Validation Loss (residual)")
        ax.set_title("Hybrid Model - Residual Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE on ARIMA residual)")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[HybridTest] Loss curve saved to {save_path}")
        return save_path

    def convergence_stats(self, tolerance: float = 0.05) -> dict:
        """Same convergence summary as Test.convergence_stats, applied to the residual DL loss."""
        if "loss" not in self.history:
            print("[HybridTest] No training history available; skipping convergence stats.")
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
        print("[HybridTest] Convergence stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return stats

    def accuracy_metrics(self, y_test: np.ndarray, exog_test: Optional[np.ndarray] = None) -> dict:
        """
        Reconstructs Y_hat_final = Y_hat_linear + e_hat via HybridModel.evaluate()
        and returns its full result dict (metrics + aligned arrays for plotting).
        """
        eval_result = self.hybrid.evaluate(y_test, exog_test)
        return eval_result

    def plot_predictions(self, eval_result: dict, filename: str = "hybrid_predictions.png") -> str:
        """Plot actual vs reconstructed predicted price over the test set."""
        actual = eval_result["y_true"]
        predicted = eval_result["y_hat_final"]

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(actual, color="black", label="Actual")
        ax.plot(predicted, color="green", label="Predicted (ARIMA + Residual DL)")
        ax.set_title("Hybrid Model - Actual vs Predicted (Test Set)")
        ax.set_xlabel("Test Sample")
        ax.set_ylabel("Value (scaled)")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[HybridTest] Prediction plot saved to {save_path}")
        return save_path

    def plot_components(self, eval_result: dict, filename: str = "hybrid_components.png") -> str:
        """Plot the linear (ARIMA) and residual (DL) components alongside the final reconstruction."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        axes[0].plot(eval_result["y_true"], color="black", label="Actual")
        axes[0].plot(eval_result["y_hat_final"], color="green", label="Y_hat_final = Linear + Residual")
        axes[0].plot(eval_result["y_hat_linear"], color="blue", linestyle="--", label="Y_hat_linear (ARIMA)")
        axes[0].set_title("Hybrid Decomposition: Actual vs Linear vs Final")
        axes[0].legend()

        axes[1].plot(eval_result["e_hat"], color="orange", label="e_hat (predicted residual)")
        axes[1].axhline(0, color="grey", linewidth=0.8)
        axes[1].set_title("Predicted Residual (DL component)")
        axes[1].set_xlabel("Test Sample")
        axes[1].legend()

        fig.tight_layout()
        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[HybridTest] Component plot saved to {save_path}")
        return save_path

    def save_results(self, results: dict, filename: Optional[str] = None) -> str:
        """Save convergence/accuracy results to results/ as timestamped JSON, mirroring Test.save_results."""
        if filename is None:
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.json"
        save_path = os.path.join(self.results_dir, filename)

        payload = {
            "model_path": self.model_path,
            "model_type": "Hybrid",
            "components": {
                "linear": f"ARIMA{self.hybrid.arima_order}",
                "residual": type(self.hybrid.dl_model).__name__,
            },
            "timestamp": datetime.now().isoformat(),
            **results,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[HybridTest] Results saved to {save_path}")
        return save_path

    def run(self, y_test: np.ndarray, exog_test: Optional[np.ndarray] = None) -> dict:
        """Run the full evaluation suite, save results to results/, and return them."""
        self.plot_loss_curve()
        convergence = self.convergence_stats()
        eval_result = self.accuracy_metrics(y_test, exog_test)
        self.plot_predictions(eval_result)
        self.plot_components(eval_result)

        results = {"convergence": convergence, "accuracy": eval_result["metrics"]}
        self.save_results(results)
        return results


if __name__ == "__main__":
    # Quick manual test: python hybrid_test.py
    # Rebuilds the same test set main.py's run_hybrid() used, then evaluates
    # the saved Hybrid model fresh from disk.
    from data import DataDownloader, DataHandler

    TICKER = "AAPL"
    FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
    TARGET_COLUMN = "Close"
    PREDICTION_DAYS = 60
    ARIMA_ORDER = (5, 1, 0)
    DL_TYPE = "gru"  # must match the dl_type used when the model was trained

    downloader = DataDownloader(TICKER)
    csv_path = downloader.download()

    handler = DataHandler(csv_path, feature_columns=FEATURE_COLUMNS, target_column=TARGET_COLUMN)
    handler.clean()
    train_df, test_df = handler.split(split_method="ratio", split_param=0.8)
    train_df, test_df = handler.scale(train_df, test_df)

    exog_columns = [c for c in FEATURE_COLUMNS if c != TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN].values
    exog_test = test_df[exog_columns].values if exog_columns else None
    n_features = 1 + len(exog_columns)

    tester = HybridTest(
        model_path=os.path.join("trained_models", "HybridModel.keras"),
        input_shape=(PREDICTION_DAYS, n_features),
        arima_order=ARIMA_ORDER, dl_type=DL_TYPE,
    )
    tester.run(y_test, exog_test)
