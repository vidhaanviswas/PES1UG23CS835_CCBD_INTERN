"""
Mitigation Policy Simulation

Applies a simple action policy based on predicted skew probability and
estimates impact using runtime stats as a proxy for job duration.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_model import load_models
from predict_job import predict_from_dataframe
from feature_engineering import prepare_features_for_training
from splitters import time_based_split
from logger import setup_logging, close_logging


def simulate_policy(df: pd.DataFrame, model_name: str = "logistic_regression",
                    threshold: float = 0.7,
                    skew_reduction: float = 0.3,
                    overhead: float = 0.05):
    """
    Simulate mitigation policy outcomes.

    - If predicted skew >= threshold, apply mitigation.
    - If truly skewed: duration reduced by skew_reduction.
    - If not skewed: duration increased by overhead.
    """
    preds = predict_from_dataframe(df, model_name=model_name)
    preds["apply_mitigation"] = (preds["skew_probability"] >= threshold).astype(int)

    # Use max_task_runtime as a proxy for duration.
    if "max_task_runtime" not in preds.columns:
        raise ValueError("max_task_runtime missing; required for mitigation impact simulation")

    baseline = preds["max_task_runtime"].astype(float)
    adjusted = baseline.copy()

    skewed_mask = preds["is_skewed"] == 1
    mitigated_mask = preds["apply_mitigation"] == 1

    adjusted[skewed_mask & mitigated_mask] = baseline[skewed_mask & mitigated_mask] * (1 - skew_reduction)
    adjusted[(~skewed_mask) & mitigated_mask] = baseline[(~skewed_mask) & mitigated_mask] * (1 + overhead)

    results = {
        "jobs": len(preds),
        "mitigations": int(mitigated_mask.sum()),
        "true_skewed": int(skewed_mask.sum()),
        "true_mitigated": int((skewed_mask & mitigated_mask).sum()),
        "false_mitigated": int(((~skewed_mask) & mitigated_mask).sum()),
        "baseline_total": float(baseline.sum()),
        "adjusted_total": float(adjusted.sum()),
        "relative_improvement": float((baseline.sum() - adjusted.sum()) / baseline.sum()) if baseline.sum() else 0.0,
    }

    return results


def main():
    df = pd.read_csv("data/processed/job_level_data.csv")
    train_df, test_df = time_based_split(df)

    # Evaluate mitigation on test set only.
    results = simulate_policy(test_df, model_name="logistic_regression", threshold=0.7)

    print("=" * 80)
    print("MITIGATION POLICY SIMULATION")
    print("=" * 80)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mitigation policy simulation")
    parser.add_argument("--save-log", action="store_true", 
                        help="Save terminal output to log file in outputs/ directory")
    args = parser.parse_args()
    
    logger = None
    if args.save_log:
        logger = setup_logging("mitigation_simulation")
    
    try:
        run()
    finally:
        if logger:
            close_logging(logger)
