"""
Baseline Comparison Module

This module implements a simple rule-based baseline for comparison with ML models.
Baseline: If max_task_runtime > threshold → skewed
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def baseline_predict(df: pd.DataFrame, threshold: float = None) -> pd.Series:
    """
    Predict skewed jobs using a simple rule-based baseline.
    
    Baseline rule: If max_task_runtime > threshold → skewed (1), else non-skewed (0)
    If threshold is None, uses median of max_task_runtime as threshold.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Job-level dataframe with max_task_runtime
    threshold : float, optional
        Threshold for max_task_runtime. If None, uses median.
        
    Returns:
    --------
    pd.Series
        Predicted labels (0 or 1)
    """
    if threshold is None:
        threshold = df['max_task_runtime'].median()
        print(f"Using median max_task_runtime as threshold: {threshold:.2f}")
    
    predictions = (df['max_task_runtime'] > threshold).astype(int)
    
    return predictions


def find_optimal_threshold(df: pd.DataFrame, y_true: pd.Series) -> float:
    """
    Find the optimal threshold for the baseline by maximizing F1-score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Job-level dataframe with max_task_runtime
    y_true : pd.Series
        True labels
        
    Returns:
    --------
    float
        Optimal threshold value
    """
    max_runtime = df['max_task_runtime']
    thresholds = np.linspace(max_runtime.min(), max_runtime.max(), 100)
    
    best_threshold = None
    best_f1 = -1
    
    for threshold in thresholds:
        y_pred = (max_runtime > threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.2f} (F1-score: {best_f1:.4f})")
    
    return best_threshold


def evaluate_baseline(df: pd.DataFrame, y_true: pd.Series, 
                     threshold: float = None) -> dict:
    """
    Evaluate the baseline model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Job-level dataframe with max_task_runtime
    y_true : pd.Series
        True labels
    threshold : float, optional
        Threshold for prediction. If None, uses optimal threshold.
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    if threshold is None:
        threshold = find_optimal_threshold(df, y_true)
    
    y_pred = baseline_predict(df, threshold)
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold
    }
    
    print(f"\n{'='*60}")
    print("Baseline Model Evaluation Results")
    print(f"{'='*60}")
    print(f"Threshold: {metrics['threshold']:.2f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics, y_pred


def plot_baseline_confusion_matrix(y_true: pd.Series, y_pred: pd.Series,
                                  save_path: str = None):
    """
    Plot confusion matrix for baseline model.
    
    Parameters:
    -----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    save_path : str, optional
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Skewed', 'Skewed'],
                yticklabels=['Non-Skewed', 'Skewed'])
    plt.title('Confusion Matrix - Baseline Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        # Convert to absolute path if relative
        if not os.path.isabs(save_path):
            project_root = Path(__file__).parent.parent
            save_path = project_root / save_path
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Baseline confusion matrix saved to {save_path}")
    
    plt.show()


def compare_with_ml_models(baseline_metrics: dict, ml_results: dict):
    """
    Compare baseline results with ML model results.
    
    Parameters:
    -----------
    baseline_metrics : dict
        Baseline evaluation metrics
    ml_results : dict
        Dictionary of ML model evaluation results
    """
    print("\n" + "="*80)
    print("BASELINE vs ML MODELS COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    # Print baseline
    print(f"{'Baseline (Rule-based)':<25} {baseline_metrics['accuracy']:<12.4f} "
          f"{baseline_metrics['precision']:<12.4f} {baseline_metrics['recall']:<12.4f} "
          f"{baseline_metrics['f1_score']:<12.4f}")
    
    # Print ML models
    for model_name, metrics in ml_results.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    from data_loader import load_task_events
    from preprocessing import clean_task_events, extract_task_runtimes, prepare_for_aggregation
    from feature_engineering import aggregate_to_job_level, encode_categorical_features, prepare_features_for_training
    from skew_labeling import label_skewed_jobs
    from train_model import split_data, load_models
    from evaluate_model import evaluate_all_models
    from feature_engineering import get_feature_columns
    
    # Test baseline
    try:
        print("Loading and preprocessing data...")
        df = load_task_events()
        df_clean = clean_task_events(df)
        df_runtimes = extract_task_runtimes(df_clean)
        df_prep = prepare_for_aggregation(df_runtimes)
        job_features = aggregate_to_job_level(df_prep)
        job_features = encode_categorical_features(job_features)
        job_labeled = label_skewed_jobs(job_features)
        
        X, y = prepare_features_for_training(job_labeled)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Evaluate baseline
        baseline_metrics, y_pred_baseline = evaluate_baseline(
            job_labeled.loc[X_test.index], y_test
        )
        plot_baseline_confusion_matrix(y_test, y_pred_baseline,
                                     "models/confusion_matrix_baseline.png")
        
        # Load and evaluate ML models
        models, scalers = load_models()
        feature_names = get_feature_columns()
        ml_results = evaluate_all_models(models, scalers, X_test, y_test, feature_names)
        
        # Compare
        compare_with_ml_models(baseline_metrics, ml_results)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
