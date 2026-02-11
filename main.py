"""
Main Pipeline Script

This script runs the complete ML pipeline for predicting data skew in cloud-based big data jobs.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_task_events
from preprocessing import clean_task_events, extract_task_runtimes
from feature_engineering import extract_pre_execution_features, encode_categorical_features, prepare_features_for_training, get_feature_columns
from skew_labeling import label_jobs_from_task_runtimes, get_skew_statistics
from train_model import train_all_models
from evaluate_model import evaluate_all_models, print_comparison_table
from baseline import evaluate_baseline, plot_baseline_confusion_matrix, compare_with_ml_models
from splitters import add_template_id, time_based_split, template_based_split
from logger import setup_logging, close_logging
import pandas as pd


def save_processed_data(df: pd.DataFrame, save_path: str = "data/processed/job_level_data.csv"):
    """
    Save processed job-level data to CSV.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Job-level dataframe
    save_path : str
        Path to save the CSV file
    """
    # Convert to absolute path if relative
    if not os.path.isabs(save_path):
        project_root = Path(__file__).parent
        save_path = project_root / save_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df.to_csv(save_path, index=False)
    print(f"\nProcessed job-level data saved to {save_path}")
    print(f"Total jobs: {len(df):,}")


def main():
    """
    Main pipeline execution function.
    """
    print("="*80)
    print("Early Prediction of Data Skew in Cloud-Based Big Data Jobs")
    print("Using Lightweight Machine Learning Models")
    print("="*80)
    
    try:
        # Step 1: Load data
        print("\n[Step 1/8] Loading dataset...")
        df = load_task_events()
        
        # Step 2: Preprocess data
        print("\n[Step 2/8] Preprocessing data...")
        df_clean = clean_task_events(df)
        df_runtimes = extract_task_runtimes(df_clean)

        # Step 3: Feature engineering (pre-execution only)
        print("\n[Step 3/8] Engineering pre-execution features...")
        pre_exec_features = extract_pre_execution_features(df_clean)
        pre_exec_features = encode_categorical_features(pre_exec_features)

        # Step 4: Label skewed jobs from runtimes
        print("\n[Step 4/8] Labeling skewed jobs...")
        labels = label_jobs_from_task_runtimes(df_runtimes)

        # Merge features + labels
        job_labeled = pre_exec_features.merge(labels, on="job_id", how="inner")
        job_labeled = add_template_id(job_labeled)
        
        # Print skew statistics
        stats = get_skew_statistics(job_labeled)
        print("\nSkew Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value:,}")
        
        # Save processed data
        save_processed_data(job_labeled)
        
        # Step 5: Prepare features for training
        print("\n[Step 5/8] Preparing features for training...")
        X, y = prepare_features_for_training(job_labeled, mode="pre_exec")

        # Step 6: Train ML models (time-based split used for saved models)
        print("\n[Step 6/8] Training ML models...")
        train_time_df, test_time_df = time_based_split(job_labeled)
        X_train, y_train = prepare_features_for_training(train_time_df, mode="pre_exec")
        X_test, y_test = prepare_features_for_training(test_time_df, mode="pre_exec")
        models, scalers = train_all_models(X_train, y_train, use_smote=True, calibrate=True)

        # Step 7: Evaluate ML models (time-based)
        print("\n[Step 7/8] Evaluating ML models (time-based split)...")
        feature_names = get_feature_columns(mode="pre_exec")
        ml_results_time = evaluate_all_models(models, scalers, X_test, y_test, feature_names)
        print_comparison_table(ml_results_time)

        # Step 8: Evaluate baseline and compare (time-based)
        print("\n[Step 8/8] Evaluating baseline model (time-based split)...")
        baseline_metrics, y_pred_baseline = evaluate_baseline(
            test_time_df, y_test, feature="num_tasks"
        )
        plot_baseline_confusion_matrix(y_test, y_pred_baseline,
                                     "models/confusion_matrix_baseline.png")
        compare_with_ml_models(baseline_metrics, ml_results_time)

        # Additional evaluation: template-based split
        print("\n[Extra] Evaluating ML models (template-based split)...")
        train_tpl_df, test_tpl_df = template_based_split(job_labeled)
        X_train_tpl, y_train_tpl = prepare_features_for_training(train_tpl_df, mode="pre_exec")
        X_test_tpl, y_test_tpl = prepare_features_for_training(test_tpl_df, mode="pre_exec")
        models_tpl, _ = train_all_models(X_train_tpl, y_train_tpl, use_smote=True, calibrate=True,
                                         save_path="models/trained_models_template.pkl")
        ml_results_tpl = evaluate_all_models(models_tpl, {}, X_test_tpl, y_test_tpl, feature_names)
        print_comparison_table(ml_results_tpl)
        
        print("\n" + "="*80)
        print("Pipeline execution completed successfully!")
        print("="*80)
        print("\nOutputs generated:")
        print("  - data/processed/job_level_data.csv")
        print("  - models/trained_models.pkl")
        print("  - models/confusion_matrix_*.png")
        print("  - models/feature_importance_*.png")
        print("="*80)
        
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease ensure that task_events.csv is placed in the data/raw/ directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data skew prediction pipeline")
    parser.add_argument("--save-log", action="store_true", 
                        help="Save terminal output to log file in outputs/ directory")
    args = parser.parse_args()
    
    logger = None
    if args.save_log:
        logger = setup_logging("main")
    
    try:
        main()
    finally:
        if logger:
            close_logging(logger)
