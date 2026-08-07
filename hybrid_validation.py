# File: hybrid_test.py
# Evaluation module for HybridModel (VARIMAX + DL residual)
# Evaluates multi-target reconstructed predictions against a held-out test set,
# computes per-target accuracy metrics (Log-space & Price-space), loss curves,
# convergence statistics, and generates component decomposition plots.

import json
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from models.dl_model import HybridModel

PLOTS_DIR = "plots"
RESULTS_DIR = "results"
HISTORY_DIR = "history"


class HybridValidation:
    """
    Evaluates an already-trained HybridModel against a test set.

    Responsibilities (mirrors PureDLValidation in validation.py, adapted for Hybrid):
      1. Loss curve         - Training/validation loss of the DL residual sub-model.
      2. Convergence rate   - Loss reduction speed and plateau epoch summary.
      3. Predictive accuracy - Multi-target Log-space metrics, Price-space metrics,
                               and Directional Accuracy on reconstructed output.
      4. Visualizations      - Plots for actual vs predicted, plus linear/residual decomposition.
    """

    def __init__(self, model_path: str, input_shape: Tuple[int, int], output_dim: int = 1,
                 varimax_order: Tuple[int, int] = (1, 1), dl_type: str = "lstm",
                 history_path: Optional[str] = None, plots_dir: str = PLOTS_DIR,
                 results_dir: str = RESULTS_DIR):
        self.model_path = model_path
        self.hybrid = HybridModel(
            input_shape=input_shape,
            output_dim=output_dim,
            varimax_order=varimax_order,
            dl_type=dl_type
        )
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
        ax.plot(loss, label="Training Loss (Residual DL)")
        if val_loss:
            ax.plot(val_loss, label="Validation Loss (Residual DL)")
        ax.set_title("Hybrid Model - Residual Loss Curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MSE)")
        ax.legend()
        fig.tight_layout()

        save_path = os.path.join(self.plots_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)
        print(f"[HybridTest] Loss curve saved to {save_path}")
        return save_path

    def convergence_stats(self, tolerance: float = 0.05) -> dict:
        """Summarize how the residual DL sub-model loss converged during training."""
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
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
        return stats

    def _reconstruct_price(self, y_log: np.ndarray, raw_close_today: np.ndarray) -> np.ndarray:
        """Reconstruct actual price from log return: C_{t+1} = C_t * exp(y_t)."""
        return raw_close_today * np.exp(y_log)

    def accuracy_metrics(
        self,
        y_test: np.ndarray,
        exog_test: Optional[np.ndarray] = None,
        target_names: Optional[list] = None,
        raw_close_map: Optional[Dict[str, np.ndarray]] = None
    ) -> Tuple[Dict[str, dict], dict]:
        """
        Reconstructs Y_hat_final = Y_hat_linear + e_hat via HybridModel.evaluate()
        and computes per-target evaluation metrics (Log-space and Price-space).
        """
        eval_result = self.hybrid.evaluate(y_test, exog_test)
        
        y_true = eval_result["y_true"]        # (n_samples, n_targets)
        y_hat_final = eval_result["y_hat_final"] # (n_samples, n_targets)

        if y_true.ndim == 1:
            y_true = y_true[:, None]
            y_hat_final = y_hat_final[:, None]
            
        n_targets = y_true.shape[-1]
        if not target_names or len(target_names) != n_targets:
            target_names = ["Target"] if n_targets == 1 else [f"Target_{i}" for i in range(n_targets)]

        raw_close_map = raw_close_map or {}
        per_target_metrics: Dict[str, dict] = {}

        for i, name in enumerate(target_names):
            true_i = y_true[:, i]
            pred_i = y_hat_final[:, i]

            # 1. Log / Return Space Metrics
            log_errors = pred_i - true_i
            log_rmse = float(np.sqrt(np.mean(log_errors ** 2)))
            log_mae = float(np.mean(np.abs(log_errors)))

            # Directional Accuracy
            if len(true_i) > 1:
                actual_dir = np.sign(np.diff(true_i))
                pred_dir = np.sign(pred_i[1:] - true_i[:-1])
                directional_accuracy = float(np.mean(actual_dir == pred_dir)) * 100
            else:
                directional_accuracy = float("nan")

            metrics = {
                "Log_RMSE": log_rmse,
                "Log_MAE": log_mae,
                "Directional Accuracy (%)": directional_accuracy,
            }

            # 2. Price Space Metrics (If raw close price maps are provided)
            raw_close_today = raw_close_map.get(name)
            if raw_close_today is not None:
                # Align raw close to match window-shifted true_i length
                raw_close_today = raw_close_today[-len(true_i):]
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

            print(f"[HybridTest] Predictive accuracy ({name}):")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

            per_target_metrics[name] = metrics

        return per_target_metrics, eval_result

    def plot_predictions(self, eval_result: dict, target_names: Optional[list] = None) -> list:
        """Plot actual vs reconstructed predictions for each target."""
        y_true = eval_result["y_true"]
        y_hat_final = eval_result["y_hat_final"]

        if y_true.ndim == 1:
            y_true = y_true[:, None]
            y_hat_final = y_hat_final[:, None]

        n_targets = y_true.shape[-1]
        if not target_names or len(target_names) != n_targets:
            target_names = ["Target"] if n_targets == 1 else [f"Target_{i}" for i in range(n_targets)]

        saved_paths = []
        for i, name in enumerate(target_names):
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(y_true[:, i], color="black", label="Actual")
            ax.plot(y_hat_final[:, i], color="green", label="Predicted (VARIMAX + Residual DL)")
            ax.set_title(f"Hybrid Model - {name} Actual vs Predicted (Test Set)")
            ax.set_xlabel("Test Sample")
            ax.set_ylabel("Value")
            ax.legend()
            fig.tight_layout()

            safe_name = name.replace(" ", "_")
            save_path = os.path.join(self.plots_dir, f"hybrid_predictions_{safe_name}.png")
            fig.savefig(save_path)
            plt.close(fig)
            print(f"[HybridTest] Prediction plot saved to {save_path}")
            saved_paths.append(save_path)

        return saved_paths

    def plot_components(self, eval_result: dict, target_names: Optional[list] = None) -> list:
        """Plot the linear (VARIMAX) and residual (DL) components for each target."""
        y_true = eval_result["y_true"]
        y_hat_final = eval_result["y_hat_final"]
        y_hat_linear = eval_result["y_hat_linear"]
        e_hat = eval_result["e_hat"]

        if y_true.ndim == 1:
            y_true = y_true[:, None]
            y_hat_final = y_hat_final[:, None]
            y_hat_linear = y_hat_linear[:, None]
            e_hat = e_hat[:, None]

        n_targets = y_true.shape[-1]
        if not target_names or len(target_names) != n_targets:
            target_names = ["Target"] if n_targets == 1 else [f"Target_{i}" for i in range(n_targets)]

        saved_paths = []
        for i, name in enumerate(target_names):
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            axes[0].plot(y_true[:, i], color="black", label="Actual")
            axes[0].plot(y_hat_final[:, i], color="green", label="Y_hat_final = Linear + Residual")
            axes[0].plot(y_hat_linear[:, i], color="blue", linestyle="--", label="Y_hat_linear (VARIMAX)")
            axes[0].set_title(f"Hybrid Decomposition ({name}): Actual vs Linear vs Final")
            axes[0].legend()

            axes[1].plot(e_hat[:, i], color="orange", label="e_hat (Predicted Residual DL)")
            axes[1].axhline(0, color="grey", linewidth=0.8)
            axes[1].set_title(f"Predicted Residual ({name})")
            axes[1].set_xlabel("Test Sample")
            axes[1].legend()

            fig.tight_layout()
            safe_name = name.replace(" ", "_")
            save_path = os.path.join(self.plots_dir, f"hybrid_components_{safe_name}.png")
            fig.savefig(save_path)
            plt.close(fig)
            print(f"[HybridTest] Component plot saved to {save_path}")
            saved_paths.append(save_path)

        return saved_paths

    def save_results(self, results: dict, filename: Optional[str] = None,
                      model_info: Optional[dict] = None) -> str:
        """Save convergence and accuracy results to results/ as timestamped JSON."""
        if filename is None:
            model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.json"
        save_path = os.path.join(self.results_dir, filename)

        payload = {
            "model_path": self.model_path,
            "model_type": "Hybrid",
            "components": {
                "linear": f"VARIMAX{self.hybrid.varimax_order}",
                "residual": type(self.hybrid.dl_model).__name__,
            },
            "timestamp": datetime.now().isoformat(),
            **(model_info or {}),
            **results,
        }
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[HybridTest] Results saved to {save_path}")
        return save_path

    def run(self, y_test: np.ndarray, exog_test: Optional[np.ndarray] = None,
            target_names: Optional[list] = None,
            raw_close_map: Optional[Dict[str, np.ndarray]] = None,
            model_info: Optional[dict] = None) -> dict:
        """
        Run the complete evaluation suite across all targets, log metrics,
        save results, and generate decomposition plots.
        """
        self.plot_loss_curve()
        convergence = self.convergence_stats()
        
        per_target_metrics, eval_result = self.accuracy_metrics(
            y_test, exog_test=exog_test, target_names=target_names, raw_close_map=raw_close_map
        )
        
        self.plot_predictions(eval_result, target_names=target_names)
        self.plot_components(eval_result, target_names=target_names)

        results = {"convergence": convergence, "accuracy": per_target_metrics}
        self.save_results(results, model_info=model_info)
        return results


if __name__ == "__main__":
    # Smoke test simulation for Multi-Target Hybrid Model Evaluation
    print("\nExecuting hybrid_test.py multi-target evaluation run...")
    
    WINDOW = 10
    N_TARGETS = 2
    N_EXOG = 2
    TEST_LEN = 100

    # 1. Generate Dummy Test Data
    np.random.seed(42)
    y_test_dummy = np.random.randn(TEST_LEN, N_TARGETS) * 0.01
    exog_test_dummy = np.random.randn(TEST_LEN, N_EXOG)
    
    raw_close_aapl = 150.0 * np.exp(np.cumsum(y_test_dummy[:, 0]))
    raw_close_qqq = 380.0 * np.exp(np.cumsum(y_test_dummy[:, 1]))

    # 2. Mock Evaluator Setup
    # Note: Assumes a trained HybridModel exists or saved mock weights are present
    model_file = os.path.join("trained_models", "HybridModel.keras")
    
    if os.path.exists(model_file):
        tester = HybridValidation(
            model_path=model_file,
            input_shape=(WINDOW, N_TARGETS + N_EXOG),
            output_dim=N_TARGETS,
            varimax_order=(1, 1),
            dl_type="lstm"
        )
        
        tester.run(
            y_test=y_test_dummy,
            exog_test=exog_test_dummy,
            target_names=["AAPL", "QQQ"],
            raw_close_map={"AAPL": raw_close_aapl, "QQQ": raw_close_qqq}
        )
    else:
        print(f"[HybridTest] Model file not found at {model_file}. Run training script first.")