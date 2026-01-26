"""
Main Pipeline Script

This script runs the complete ML pipeline for predicting data skew in cloud-based big data jobs.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_task_events
from preprocessing import clean_task_events, extract_task_runtimes, prepare_for_aggregation
from feature_engineering import aggregate_to_job_level, encode_categorical_features, prepare_features_for_training, get_feature_columns
from skew_labeling import label_skewed_jobs, get_skew_statistics
from train_model import split_data, train_all_models
from evaluate_model import evaluate_all_models, print_comparison_table
from baseline import evaluate_baseline, plot_baseline_confusion_matrix, compare_with_ml_models
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
        df_prep = prepare_for_aggregation(df_runtimes)
        
        # Step 3: Feature engineering
        print("\n[Step 3/8] Engineering features...")
        job_features = aggregate_to_job_level(df_prep)
        job_features = encode_categorical_features(job_features)
        
        # Step 4: Label skewed jobs
        print("\n[Step 4/8] Labeling skewed jobs...")
        job_labeled = label_skewed_jobs(job_features)
        
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
        X, y = prepare_features_for_training(job_labeled)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Step 6: Train ML models
        print("\n[Step 6/8] Training ML models...")
        models, scalers = train_all_models(X_train, y_train, X_test)
        
        # Step 7: Evaluate ML models
        print("\n[Step 7/8] Evaluating ML models...")
        feature_names = get_feature_columns()
        ml_results = evaluate_all_models(models, scalers, X_test, y_test, feature_names)
        print_comparison_table(ml_results)
        
        # Step 8: Evaluate baseline and compare
        print("\n[Step 8/8] Evaluating baseline model...")
        baseline_metrics, y_pred_baseline = evaluate_baseline(
            job_labeled.loc[X_test.index], y_test
        )
        plot_baseline_confusion_matrix(y_test, y_pred_baseline,
                                     "models/confusion_matrix_baseline.png")
        
        # Final comparison
        compare_with_ml_models(baseline_metrics, ml_results)
        
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
    main()
