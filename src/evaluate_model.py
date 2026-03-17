"""
Model Evaluation Module

This module evaluates trained ML models using various metrics and visualizations.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — prevents tkinter errors in CLI runs
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def _show_plot_if_interactive() -> None:
    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    plt.close()


def evaluate_model(y_true: pd.Series, y_pred: pd.Series, model_name: str = "Model") -> dict:
    """
    Evaluate a model using multiple metrics.
    
    Parameters:
    -----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    model_name : str
        Name of the model for display
        
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results for {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


def evaluate_model_proba(y_true: pd.Series, y_proba: np.ndarray,
                         model_name: str = "Model") -> dict:
    """
    Evaluate probability quality metrics.
    """
    pr_auc = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)

    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"Brier:    {brier:.4f}")

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
    }


def expected_calibration_error(y_true: pd.Series, y_proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_proba, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        acc = y_true[mask].mean()
        conf = y_proba[mask].mean()
        ece += (mask.mean()) * abs(acc - conf)
    return float(ece)


def plot_confusion_matrix(y_true: pd.Series, y_pred: pd.Series, 
                         model_name: str = "Model",
                         save_path: str = None):
    """
    Plot and save confusion matrix.
    
    Parameters:
    -----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Skewed', 'Skewed'],
                yticklabels=['Non-Skewed', 'Skewed'])
    plt.title(f'Confusion Matrix - {model_name}')
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
        print(f"Confusion matrix saved to {save_path}")
    
    _show_plot_if_interactive()


def plot_feature_importance(model, feature_names: list, model_name: str = "Model",
                           save_path: str = None):
    """
    Plot feature importance for Random Forest model.
    
    Parameters:
    -----------
    model
        Trained model (must have feature_importances_ attribute)
    feature_names : list
        List of feature names
    model_name : str
        Name of the model
    save_path : str, optional
        Path to save the plot
    """
    base = model
    if hasattr(model, "base_estimator"):
        base = model.base_estimator
    elif hasattr(model, "estimators_") and model.estimators_:
        base = model.estimators_[0]

    if not hasattr(base, 'feature_importances_'):
        print(f"{model_name} does not support feature importance visualization")
        return

    importances = base.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Feature Importance - {model_name}')
    plt.bar(range(len(feature_names)), importances[indices])
    plt.xticks(range(len(feature_names)), 
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.tight_layout()
    
    if save_path:
        # Convert to absolute path if relative
        if not os.path.isabs(save_path):
            project_root = Path(__file__).parent.parent
            save_path = project_root / save_path
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    _show_plot_if_interactive()


def evaluate_all_models(models: dict, scalers: dict, X_test: pd.DataFrame,
                       y_test: pd.Series, feature_names: list,
                       skip_plots: bool = False):
    """
    Evaluate all trained models and (optionally) generate visualizations.

    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    scalers : dict
        Dictionary of scalers (currently unused; kept for API compatibility)
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    feature_names : list
        List of feature names
    skip_plots : bool
        When True, skip confusion-matrix and feature-importance plots.
        Use this during rolling-eval loops to avoid generating dozens of images.
    """
    results = {}
    
    for model_name, model in models.items():
        # Predict
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred.astype(float)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, model_name)
        proba_metrics = evaluate_model_proba(y_test, y_proba, model_name)
        ece = expected_calibration_error(y_test.to_numpy(), y_proba)
        metrics.update(proba_metrics)
        metrics["ece"] = ece
        results[model_name] = metrics
        
        if not skip_plots:
            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred, model_name,
                                f"models/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")

            # Plot feature importance (for Random Forest)
            if model_name == 'random_forest':
                plot_feature_importance(model, feature_names, model_name,
                                      f"models/feature_importance_{model_name.lower().replace(' ', '_')}.png")

    return results


def print_rolling_summary(fold_results: list) -> dict:
    """
    Aggregate per-fold evaluation results and print a mean ± std summary table.

    Parameters
    ----------
    fold_results : list[dict]
        Each element is the return value of ``evaluate_all_models`` for one fold,
        i.e. ``{model_name: metrics_dict}``.

    Returns
    -------
    dict
        ``{model_name: {metric: {'mean': float, 'std': float}}}``
    """
    accum = defaultdict(lambda: defaultdict(list))
    for fold_res in fold_results:
        for model_name, metrics in fold_res.items():
            for metric, value in metrics.items():
                accum[model_name][metric].append(value)

    agg = {
        model_name: {
            k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in metric_lists.items()
        }
        for model_name, metric_lists in accum.items()
    }

    n_folds = len(fold_results)
    width = 80
    print(f"\n{'='*width}")
    print(f"ROLLING TIME-SPLIT SUMMARY  ({n_folds} folds, mean \u00b1 std)")
    print(f"{'='*width}")
    print(f"{'Model':<22} {'PR-AUC':<19} {'Recall':<19} {'F1-Score':<19}")
    print("-" * width)

    def _fmt(m, key):
        if key not in m:
            return "N/A"
        return f"{m[key]['mean']:.4f} \u00b1 {m[key]['std']:.4f}"

    for model_name, m in agg.items():
        print(f"{model_name:<22} {_fmt(m,'pr_auc'):<19} {_fmt(m,'recall'):<19} "
              f"{_fmt(m,'f1_score'):<19}")

    print("=" * width + "\n")
    return agg


def print_comparison_table(results: dict):
    """
    Print a comparison table of all model results.
    
    Parameters:
    -----------
    results : dict
        Dictionary of evaluation results for each model
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'PR-AUC':<10} {'ECE':<10}")
    print("-"*80)
    
    for model_name, metrics in results.items():
          print(f"{model_name:<25} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f} {metrics.get('pr_auc', 0.0):<10.4f} "
              f"{metrics.get('ece', 0.0):<10.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    from data_loader import load_task_events
    from preprocessing import clean_task_events, extract_task_runtimes
    from feature_engineering import extract_pre_execution_features, encode_categorical_features, prepare_features_for_training, get_feature_columns
    from skew_labeling import label_jobs_from_task_runtimes
    from train_model import split_data, load_models
    
    # Test model evaluation
    try:
        print("Loading and preprocessing data...")
        df = load_task_events()
        df_clean = clean_task_events(df)
        df_runtimes = extract_task_runtimes(df_clean)
        pre_exec = extract_pre_execution_features(df_clean)
        pre_exec = encode_categorical_features(pre_exec)
        labels = label_jobs_from_task_runtimes(df_runtimes)
        job_labeled = pre_exec.merge(labels, on="job_id", how="inner")
        
        X, y = prepare_features_for_training(job_labeled, mode="pre_exec")
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Load models
        models, scalers = load_models()
        
        # Evaluate
        feature_names = get_feature_columns(mode="pre_exec")
        results = evaluate_all_models(models, scalers, X_test, y_test, feature_names)
        print_comparison_table(results)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
