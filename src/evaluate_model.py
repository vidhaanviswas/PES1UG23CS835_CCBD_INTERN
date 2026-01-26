"""
Model Evaluation Module

This module evaluates trained ML models using various metrics and visualizations.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


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
    
    plt.show()


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
    if not hasattr(model, 'feature_importances_'):
        print(f"{model_name} does not support feature importance visualization")
        return
    
    importances = model.feature_importances_
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
    
    plt.show()


def evaluate_all_models(models: dict, scalers: dict, X_test: pd.DataFrame, 
                       y_test: pd.Series, feature_names: list):
    """
    Evaluate all trained models and generate visualizations.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    scalers : dict
        Dictionary of scalers
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    feature_names : list
        List of feature names
    """
    results = {}
    
    for model_name, model in models.items():
        # Prepare features
        if model_name == 'logistic_regression':
            scaler = scalers.get('logistic_regression')
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, model_name)
        results[model_name] = metrics
        
        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, model_name,
                            f"models/confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
        
        # Plot feature importance (for Random Forest)
        if model_name == 'random_forest':
            plot_feature_importance(model, feature_names, model_name,
                                  f"models/feature_importance_{model_name.lower().replace(' ', '_')}.png")
    
    return results


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
    print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['accuracy']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1_score']:<12.4f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    from data_loader import load_task_events
    from preprocessing import clean_task_events, extract_task_runtimes, prepare_for_aggregation
    from feature_engineering import aggregate_to_job_level, encode_categorical_features, prepare_features_for_training, get_feature_columns
    from skew_labeling import label_skewed_jobs
    from train_model import split_data, load_models
    
    # Test model evaluation
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
        
        # Load models
        models, scalers = load_models()
        
        # Evaluate
        feature_names = get_feature_columns()
        results = evaluate_all_models(models, scalers, X_test, y_test, feature_names)
        print_comparison_table(results)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
