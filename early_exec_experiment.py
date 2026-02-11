"""
Early-Execution Prediction Experiment

Evaluates how early we can predict skew using the first k% of tasks.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_task_events
from preprocessing import clean_task_events, extract_task_runtimes
from feature_engineering import extract_pre_execution_features, encode_categorical_features, prepare_features_for_training, get_feature_columns
from skew_labeling import label_jobs_from_task_runtimes
from early_execution import compute_early_runtime_features
from splitters import add_template_id, time_based_split
from train_model import train_all_models
from evaluate_model import evaluate_all_models, print_comparison_table
from logger import setup_logging, close_logging


def run_for_k(k: float):
    print(f"\n=== Early-exec experiment: k={k:.2f} ===")

    df = load_task_events()
    df_clean = clean_task_events(df)
    runtimes = extract_task_runtimes(df_clean)

    pre_exec = extract_pre_execution_features(df_clean)
    pre_exec = encode_categorical_features(pre_exec)

    labels = label_jobs_from_task_runtimes(runtimes)
    early = compute_early_runtime_features(runtimes, k=k)

    merged = pre_exec.merge(early, on="job_id", how="inner")
    merged = merged.merge(labels, on="job_id", how="inner")
    merged = add_template_id(merged)

    train_df, test_df = time_based_split(merged)
    X_train, y_train = prepare_features_for_training(train_df, mode="early_exec")
    X_test, y_test = prepare_features_for_training(test_df, mode="early_exec")

    models, scalers = train_all_models(X_train, y_train, use_smote=True, calibrate=True,
                                       save_path=f"models/trained_models_early_{int(k*100)}.pkl")
    results = evaluate_all_models(models, scalers, X_test, y_test, get_feature_columns(mode="early_exec"))
    print_comparison_table(results)

    return results


def main():
    ks = [0.05, 0.10, 0.20]
    for k in ks:
        run_for_k(k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run early-execution prediction experiment")
    parser.add_argument("--save-log", action="store_true", 
                        help="Save terminal output to log file in outputs/ directory")
    args = parser.parse_args()
    
    logger = None
    if args.save_log:
        logger = setup_logging("early_exec_experiment")
    
    try:
        main()
    finally:
        if logger:
            close_logging(logger)
