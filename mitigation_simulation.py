"""
Mitigation Policy Simulation

Applies a simple action policy based on predicted skew probability and
estimates impact using runtime stats as a proxy for job duration.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_model import load_models
from predict_job import predict_from_dataframe, get_default_threshold
from feature_engineering import prepare_features_for_training
from splitters import time_based_split
from logger import setup_logging, close_logging


def simulate_policy(df: pd.DataFrame, model_name: str = "logistic_regression",
                    threshold: Optional[float] = None,
                    skew_reduction: float = 0.3,
                    overhead: float = 0.05):
    """
    Simulate mitigation policy outcomes.

    - If predicted skew >= threshold, apply mitigation.
    - If truly skewed: duration reduced by skew_reduction.
    - If not skewed: duration increased by overhead.
    """
    threshold = get_default_threshold(model_name) if threshold is None else float(threshold)

    preds = predict_from_dataframe(df, model_name=model_name, threshold=threshold)
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


def compare_models(
    df: pd.DataFrame,
    model_names: list,
    skew_reduction: float = 0.3,
    overhead: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for model_name in model_names:
        threshold = get_default_threshold(model_name)
        results = simulate_policy(
            df,
            model_name=model_name,
            threshold=threshold,
            skew_reduction=skew_reduction,
            overhead=overhead,
        )
        results["model"] = model_name
        results["threshold"] = float(threshold)
        rows.append(results)

    out = pd.DataFrame(rows)
    out["mitigation_rate"] = out["mitigations"] / out["jobs"]
    out["precision_mitigation"] = out["true_mitigated"] / out["mitigations"].where(out["mitigations"] > 0, 1)
    out["recall_on_skew"] = out["true_mitigated"] / out["true_skewed"].where(out["true_skewed"] > 0, 1)
    out = out.sort_values("relative_improvement", ascending=False)
    return out


def main(model_name: str = "logistic_regression",
         all_models: bool = False,
         output_csv: Optional[str] = None,
         skew_reduction: float = 0.3,
         overhead: float = 0.05):
    df = pd.read_csv("data/processed/job_level_data.csv")
    train_df, test_df = time_based_split(df)

    if all_models:
        model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
        comparison = compare_models(
            test_df,
            model_names=model_names,
            skew_reduction=skew_reduction,
            overhead=overhead,
        )
        print("=" * 110)
        print("MITIGATION POLICY SIMULATION - ALL MODELS")
        print("=" * 110)
        print(comparison.to_string(index=False))
        if output_csv:
            comparison.to_csv(output_csv, index=False)
            print(f"\nSaved comparison to {output_csv}")
        return

    # Evaluate mitigation on test set only for the selected model.
    results = simulate_policy(
        test_df,
        model_name=model_name,
        threshold=None,
        skew_reduction=skew_reduction,
        overhead=overhead,
    )

    print("=" * 80)
    print(f"MITIGATION POLICY SIMULATION ({model_name})")
    print("=" * 80)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mitigation policy simulation")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        help="Model to use for single-model simulation",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run simulation for all models using their default thresholds",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save all-model comparison CSV",
    )
    parser.add_argument(
        "--skew-reduction",
        type=float,
        default=0.3,
        help="Fractional runtime reduction when mitigation is correctly applied (default: 0.3)",
    )
    parser.add_argument(
        "--overhead",
        type=float,
        default=0.05,
        help="Fractional runtime overhead when mitigation is applied to non-skewed jobs (default: 0.05)",
    )
    parser.add_argument("--save-log", action="store_true", 
                        help="Save terminal output to log file in outputs/ directory")
    args = parser.parse_args()
    
    logger = None
    if args.save_log:
        logger = setup_logging("mitigation_simulation")
    
    try:
        main(
            model_name=args.model,
            all_models=args.all_models,
            output_csv=args.output_csv,
            skew_reduction=args.skew_reduction,
            overhead=args.overhead,
        )
    finally:
        if logger:
            close_logging(logger)
