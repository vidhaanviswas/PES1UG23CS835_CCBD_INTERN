"""
Advanced Visualization Script

Creates ROC curves, Precision-Recall curves, and other advanced visualizations
for your model evaluation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_model import load_models, split_data
from feature_engineering import prepare_features_for_training, get_feature_columns
from evaluate_model import evaluate_all_models

def create_roc_curves(models, scalers, X_test, y_test):
    """Create ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models.items():
        if model_name == 'logistic_regression':
            scaler = scalers.get('logistic_regression')
            X_test_scaled = scaler.transform(X_test)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    # Baseline (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.500)', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = Path(__file__).parent / "models" / "roc_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()

def create_pr_curves(models, scalers, X_test, y_test):
    """Create Precision-Recall curves (better for imbalanced data)."""
    plt.figure(figsize=(10, 8))
    
    for model_name, model in models.items():
        if model_name == 'logistic_regression':
            scaler = scalers.get('logistic_regression')
            X_test_scaled = scaler.transform(X_test)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model.predict_proba(X_test)[:, 1]
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        avg_precision = average_precision_score(y_test, y_proba)
        
        plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})', linewidth=2)
    
    # Baseline (random classifier)
    baseline_precision = y_test.mean()
    plt.axhline(y=baseline_precision, color='k', linestyle='--', 
                label=f'Random Classifier (AP = {baseline_precision:.3f})', linewidth=1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    save_path = Path(__file__).parent / "models" / "precision_recall_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Precision-Recall curves saved to {save_path}")
    plt.close()

def create_feature_comparison_plot():
    """Create comparison plots for skewed vs non-skewed jobs."""
    try:
        df = pd.read_csv("data/processed/job_level_data.csv")
    except FileNotFoundError:
        print("ERROR: job_level_data.csv not found. Run main_sample.py first.")
        return
    
    feature_cols = ['num_tasks', 'avg_task_runtime', 'max_task_runtime', 
                    'std_task_runtime', 'scheduling_class', 'priority']
    
    skewed = df[df['is_skewed'] == 1]
    non_skewed = df[df['is_skewed'] == 0]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, feature in enumerate(feature_cols):
        if feature in df.columns:
            ax = axes[idx]
            
            # Box plot
            data_to_plot = [non_skewed[feature].dropna(), skewed[feature].dropna()]
            bp = ax.boxplot(data_to_plot, tick_labels=['Non-Skewed', 'Skewed'], patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Value', fontsize=9)
            ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Feature Comparison: Skewed vs Non-Skewed Jobs', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    save_path = Path(__file__).parent / "models" / "feature_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature comparison plot saved to {save_path}")
    plt.close()

def create_metrics_comparison_bar():
    """Create bar chart comparing all models."""
    # These are your results - update if you re-run
    results = {
        'Model': ['Baseline', 'Logistic Regression', 'Random Forest'],
        'Accuracy': [0.2562, 0.9963, 0.9926],
        'Precision': [0.0195, 0.9231, 0.8333],
        'Recall': [0.8571, 0.8571, 0.7143],
        'F1-Score': [0.0382, 0.8889, 0.7692]
    }
    
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(df_results['Model'], df_results[metric], color=colors[idx], alpha=0.7)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric, fontsize=10)
        ax.set_ylim([0, 1.1])
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = Path(__file__).parent / "models" / "metrics_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics comparison plot saved to {save_path}")
    plt.close()

def main():
    """Generate all advanced visualizations."""
    print("="*80)
    print("Creating Advanced Visualizations")
    print("="*80)
    
    # Load data and models
    print("\n[1] Loading data and models...")
    try:
        df = pd.read_csv("data/processed/job_level_data.csv")
        X, y = prepare_features_for_training(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        models, scalers = load_models()
        print("   ✓ Data and models loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading data/models: {e}")
        print("   Make sure you've run main_sample.py first!")
        return
    
    # Create visualizations
    print("\n[2] Creating ROC curves...")
    create_roc_curves(models, scalers, X_test, y_test)
    
    print("\n[3] Creating Precision-Recall curves...")
    create_pr_curves(models, scalers, X_test, y_test)
    
    print("\n[4] Creating feature comparison plots...")
    create_feature_comparison_plot()
    
    print("\n[5] Creating metrics comparison chart...")
    create_metrics_comparison_bar()
    
    print("\n" + "="*80)
    print("All visualizations created successfully!")
    print("="*80)
    print("\nGenerated files:")
    print("  - models/roc_curves.png")
    print("  - models/precision_recall_curves.png")
    print("  - models/feature_comparison.png")
    print("  - models/metrics_comparison.png")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
