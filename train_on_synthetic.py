"""
Train Models on Synthetic Data

Quick training script for synthetic datasets that already have
features and labels prepared.

Usage:
    python train_on_synthetic.py
    python train_on_synthetic.py --data data/processed/synthetic_jobs.csv
    python train_on_synthetic.py --save-log
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
from feature_engineering import prepare_features_for_training, get_feature_columns
from train_model import train_all_models
from evaluate_model import evaluate_all_models, print_comparison_table
from baseline import evaluate_baseline, compare_with_ml_models
from splitters import time_based_split, template_based_split
from logger import setup_logging, close_logging


def main(data_path="data/processed/synthetic_jobs.csv"):
    """
    Train and evaluate models on synthetic data.
    """
    print("=" * 80)
    print("TRAINING ON SYNTHETIC DATA")
    print("=" * 80)
    
    # Load synthetic data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df):,} jobs")
    print(f"  Skewed: {df['skewed'].sum():,} ({df['skewed'].mean()*100:.2f}%)")
    print(f"  Non-skewed: {(df['skewed']==0).sum():,} ({(df['skewed']==0).mean()*100:.2f}%)")
    
    # Check if required columns exist
    required_cols = ['job_id', 'num_tasks', 'scheduling_class', 'priority', 'skewed']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        print("Make sure you're loading a properly formatted synthetic dataset.")
        return
    
    print("\n" + "=" * 80)
    print("TIME-BASED SPLIT EVALUATION")
    print("=" * 80)
    
    # Time-based split
    if 'submit_time' not in df.columns:
        print("\nWarning: 'submit_time' not found. Using 80/20 random split instead.")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                             stratify=df['skewed'])
    else:
        train_df, test_df = time_based_split(df, test_size=0.2)
    
    print(f"\nTrain set: {len(train_df):,} jobs")
    print(f"Test set: {len(test_df):,} jobs")
    
    # Rename column for consistency with feature engineering code
    train_df = train_df.rename(columns={'skewed': 'is_skewed'})
    test_df = test_df.rename(columns={'skewed': 'is_skewed'})
    
    # Prepare features
    X_train, y_train = prepare_features_for_training(train_df, mode="pre_exec")
    X_test, y_test = prepare_features_for_training(test_df, mode="pre_exec")
    
    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Training samples: {X_train.shape[0]:,}")
    print(f"Testing samples: {X_test.shape[0]:,}")
    
    # Train models
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    
    models, scalers = train_all_models(
        X_train, y_train, 
        use_smote=True, 
        calibrate=True,
        save_path="models/trained_models_synthetic.pkl"
    )
    
    print(f"\nTrained {len(models)} models")
    
    # Evaluate models
    print("\n" + "=" * 80)
    print("EVALUATING MODELS")
    print("=" * 80)
    
    results = evaluate_all_models(
        models, scalers, 
        X_test, y_test, 
        get_feature_columns(mode="pre_exec")
    )
    
    print_comparison_table(results)
    
    # Baseline comparison
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)
    
    baseline_results, _ = evaluate_baseline(test_df, y_test, feature="num_tasks")
    compare_with_ml_models(baseline_results, results)
    
    # Template-based split if available
    if 'template_id' in df.columns:
        print("\n" + "=" * 80)
        print("TEMPLATE-BASED SPLIT EVALUATION")
        print("=" * 80)
        
        train_df_tpl, test_df_tpl = template_based_split(df, test_size=0.2)
        
        print(f"\nTrain set: {len(train_df_tpl):,} jobs")
        print(f"Test set: {len(test_df_tpl):,} jobs")
        print(f"Train templates: {train_df_tpl['template_id'].nunique()}")
        print(f"Test templates: {test_df_tpl['template_id'].nunique()}")
        
        # Rename column for consistency
        train_df_tpl = train_df_tpl.rename(columns={'skewed': 'is_skewed'})
        test_df_tpl = test_df_tpl.rename(columns={'skewed': 'is_skewed'})
        
        X_train_tpl, y_train_tpl = prepare_features_for_training(train_df_tpl, mode="pre_exec")
        X_test_tpl, y_test_tpl = prepare_features_for_training(test_df_tpl, mode="pre_exec")
        
        # Train on template split
        models_tpl, scalers_tpl = train_all_models(
            X_train_tpl, y_train_tpl,
            use_smote=True,
            calibrate=True,
            save_path="models/trained_models_synthetic_template.pkl"
        )
        
        # Evaluate on template split
        results_tpl = evaluate_all_models(
            models_tpl, scalers_tpl,
            X_test_tpl, y_test_tpl,
            get_feature_columns(mode="pre_exec")
        )
        
        print_comparison_table(results_tpl)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print("\nModels saved:")
    print("  - models/trained_models_synthetic.pkl")
    if 'template_id' in df.columns:
        print("  - models/trained_models_synthetic_template.pkl")
    print("\nConfusion matrices and plots saved to models/ directory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train models on synthetic data"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/synthetic_jobs.csv",
        help="Path to synthetic data CSV (default: data/processed/synthetic_jobs.csv)"
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        help="Save terminal output to log file in outputs/ directory"
    )
    
    args = parser.parse_args()
    
    logger = None
    if args.save_log:
        logger = setup_logging("train_on_synthetic")
    
    try:
        main(args.data)
    finally:
        if logger:
            close_logging(logger)
