# File: ablation.py
# Standalone automation script for the Ablation Studies & Model Benchmark.
# Drives main.py's run_pure_dl()/run_hybrid() across all 9 configured
# scenarios, isolating each run's outputs under experiments_output/<scenario>/
# and consolidating the results into a single benchmark CSV.

import csv
import gc
import json
import os
import time
import traceback

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

import main

OUTPUT_ROOT = "experiments_output"
SUMMARY_CSV = os.path.join(OUTPUT_ROOT, "benchmark_summary_results.csv")

CSV_COLUMNS = [
    "scenario_name", "mode", "model_type", "include_cross_asset",
    "use_sentiment", "use_tda", "use_vix",
    "Log_RMSE", "Log_MAE", "Price_RMSE", "Price_MAE",
    "Price_MAPE_percent", "Directional_Accuracy_percent",
]

SCENARIOS = [
    # --- Group 1: Architecture comparison ---
    {"scenario_name": "01_pure_dl_rnn", "run_mode": "pure_dl", "DL_TYPE": "RNN",
     "INCLUDE_CROSS_ASSET": True, "PREDICT_QQQ_PRICE": True,
     "USE_SENTIMENT": True, "USE_TDA": True, "USE_VIX": True},
    {"scenario_name": "02_pure_dl_lstm", "run_mode": "pure_dl", "DL_TYPE": "LSTM",
     "INCLUDE_CROSS_ASSET": True, "PREDICT_QQQ_PRICE": True,
     "USE_SENTIMENT": True, "USE_TDA": True, "USE_VIX": True},
    {"scenario_name": "03_pure_dl_gru", "run_mode": "pure_dl", "DL_TYPE": "GRU",
     "INCLUDE_CROSS_ASSET": True, "PREDICT_QQQ_PRICE": True,
     "USE_SENTIMENT": True, "USE_TDA": True, "USE_VIX": True},
    {"scenario_name": "04_hybrid_varmax_gru", "run_mode": "hybrid", "DL_TYPE": "GRU",
     "INCLUDE_CROSS_ASSET": True, "PREDICT_QQQ_PRICE": True,
     "USE_SENTIMENT": True, "USE_TDA": True, "USE_VIX": True},

    # --- Group 2: Ablation on exogenous features (Hybrid model) ---
    {"scenario_name": "05_ablation_pure_price", "run_mode": "hybrid", "DL_TYPE": "GRU",
     "INCLUDE_CROSS_ASSET": True, "PREDICT_QQQ_PRICE": True,
     "USE_SENTIMENT": False, "USE_TDA": False, "USE_VIX": False},
    {"scenario_name": "06_ablation_with_sentiment", "run_mode": "hybrid", "DL_TYPE": "GRU",
     "INCLUDE_CROSS_ASSET": True, "PREDICT_QQQ_PRICE": True,
     "USE_SENTIMENT": True, "USE_TDA": False, "USE_VIX": False},
    {"scenario_name": "07_ablation_with_tda", "run_mode": "hybrid", "DL_TYPE": "GRU",
     "INCLUDE_CROSS_ASSET": True, "PREDICT_QQQ_PRICE": True,
     "USE_SENTIMENT": False, "USE_TDA": True, "USE_VIX": False},
    {"scenario_name": "08_ablation_with_vix", "run_mode": "hybrid", "DL_TYPE": "GRU",
     "INCLUDE_CROSS_ASSET": True, "PREDICT_QQQ_PRICE": True,
     "USE_SENTIMENT": False, "USE_TDA": False, "USE_VIX": True},

    # --- Group 3: Cross-asset impact ---
    {"scenario_name": "09_single_asset_aapl_only", "run_mode": "hybrid", "DL_TYPE": "GRU",
     "INCLUDE_CROSS_ASSET": False, "PREDICT_QQQ_PRICE": False,
     "USE_SENTIMENT": True, "USE_TDA": True, "USE_VIX": True},
]


def apply_config(scenario: dict) -> None:
    """Override main.py's module-level toggles for this scenario."""
    main.DL_TYPE = scenario["DL_TYPE"]
    main.INCLUDE_CROSS_ASSET = scenario["INCLUDE_CROSS_ASSET"]
    main.PREDICT_QQQ_PRICE = scenario["PREDICT_QQQ_PRICE"]
    main.USE_SENTIMENT = scenario["USE_SENTIMENT"]
    main.USE_TDA = scenario["USE_TDA"]
    main.USE_VIX = scenario["USE_VIX"]


def run_scenario(scenario: dict) -> dict:
    apply_config(scenario)

    scenario_dir = os.path.join(OUTPUT_ROOT, scenario["scenario_name"])
    plots_dir = os.path.join(scenario_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    run_fn = main.run_pure_dl if scenario["run_mode"] == "pure_dl" else main.run_hybrid
    return run_fn(plots_dir=plots_dir, results_dir=scenario_dir, results_filename="results.json")


def build_summary_csv() -> str:
    rows = []
    for scenario in SCENARIOS:
        name = scenario["scenario_name"]
        results_path = os.path.join(OUTPUT_ROOT, name, "results.json")
        if not os.path.exists(results_path):
            print(f"[Summary] Skipping {name}: no results.json found.")
            continue

        with open(results_path) as f:
            data = json.load(f)

        accuracy = data.get("accuracy", {})
        primary = accuracy.get(main.TICKER) or next(iter(accuracy.values()), {})

        rows.append({
            "scenario_name": name,
            "mode": data.get("mode", ""),
            "model_type": data.get("model_type", ""),
            "include_cross_asset": data.get("include_cross_asset", ""),
            "use_sentiment": data.get("use_sentiment", ""),
            "use_tda": data.get("use_tda", ""),
            "use_vix": data.get("use_vix", ""),
            "Log_RMSE": primary.get("Log_RMSE", ""),
            "Log_MAE": primary.get("Log_MAE", ""),
            "Price_RMSE": primary.get("Price_RMSE", ""),
            "Price_MAE": primary.get("Price_MAE", ""),
            "Price_MAPE_percent": primary.get("Price_MAPE (%)", ""),
            "Directional_Accuracy_percent": primary.get("Directional Accuracy (%)", ""),
        })

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Summary] Consolidated CSV written to {SUMMARY_CSV}\n")
    return SUMMARY_CSV


def print_csv_table(csv_path: str) -> None:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        print("[Summary] CSV is empty.")
        return
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)))


def main_loop() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    total = len(SCENARIOS)
    failures = []
    overall_start = time.time()

    for i, scenario in enumerate(SCENARIOS, start=1):
        name = scenario["scenario_name"]
        print(f"\n{'=' * 70}\n[{i}/{total}] Running {name}...\n{'=' * 70}")
        start = time.time()
        try:
            run_scenario(scenario)
            elapsed = time.time() - start
            print(f"[{i}/{total}] Finished {name} in {elapsed / 60:.1f} min.")
        except Exception:
            print(f"[{i}/{total}] FAILED {name}:")
            traceback.print_exc()
            failures.append(name)
        finally:
            tf.keras.backend.clear_session()
            gc.collect()

    total_elapsed = time.time() - overall_start
    print(f"\nAll scenarios processed in {total_elapsed / 60:.1f} min. "
          f"{total - len(failures)}/{total} succeeded.")
    if failures:
        print(f"Failed scenarios: {failures}")

    csv_path = build_summary_csv()
    print_csv_table(csv_path)


if __name__ == "__main__":
    main_loop()
