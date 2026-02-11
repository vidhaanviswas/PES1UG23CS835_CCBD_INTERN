"""
Quick Results Analysis Script

This script loads and analyzes the results from your pipeline execution.
Run this after main_sample.py to get insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle

def analyze_results():
    """Analyze pipeline results and generate insights."""
    
    print("="*80)
    print("RESULTS ANALYSIS")
    print("="*80)
    
    # 1. Load processed data
    print("\n[1] Loading processed job-level data...")
    try:
        df = pd.read_csv("data/processed/job_level_data.csv")
        print(f"   Loaded {len(df):,} jobs")
    except FileNotFoundError:
        print("   ERROR: job_level_data.csv not found. Run main_sample.py first.")
        return
    
    # 2. Basic statistics
    print("\n[2] Dataset Statistics:")
    print(f"   Total jobs: {len(df):,}")
    print(f"   Skewed jobs: {df['is_skewed'].sum():,} ({df['is_skewed'].mean()*100:.2f}%)")
    print(f"   Non-skewed jobs: {(df['is_skewed']==0).sum():,} ({(df['is_skewed']==0).mean()*100:.2f}%)")
    
    # 3. Feature statistics
    print("\n[3] Feature Statistics:")
    feature_cols = [
        'num_tasks',
        'scheduling_class',
        'priority',
        'cpu_request_mean',
        'cpu_request_std',
        'memory_request_mean',
        'memory_request_std',
        'disk_space_request_mean',
        'disk_space_request_std',
        'different_machine_constraint_mean',
        'avg_task_runtime',
        'max_task_runtime',
        'std_task_runtime',
    ]
    
    for col in feature_cols:
        if col in df.columns:
            print(f"\n   {col}:")
            print(f"     Mean: {df[col].mean():.2f}")
            print(f"     Median: {df[col].median():.2f}")
            print(f"     Std: {df[col].std():.2f}")
            print(f"     Min: {df[col].min():.2f}")
            print(f"     Max: {df[col].max():.2f}")
    
    # 4. Compare skewed vs non-skewed
    print("\n[4] Skewed vs Non-Skewed Job Comparison:")
    skewed = df[df['is_skewed'] == 1]
    non_skewed = df[df['is_skewed'] == 0]
    
    print("\n   Average values:")
    for col in feature_cols:
        if col in df.columns:
            print(f"     {col}:")
            print(f"       Skewed: {skewed[col].mean():.2f}")
            print(f"       Non-skewed: {non_skewed[col].mean():.2f}")
            if non_skewed[col].mean() > 0:
                ratio = skewed[col].mean() / non_skewed[col].mean()
                print(f"       Ratio: {ratio:.2f}x")
    
    # 5. Feature correlations
    print("\n[5] Feature Correlations with Skew Label:")
    if 'is_skewed' in df.columns:
        correlations = df[feature_cols + ['is_skewed']].corr()['is_skewed'].sort_values(ascending=False)
        print("\n   Correlation with is_skewed:")
        for feature, corr in correlations.items():
            if feature != 'is_skewed':
                print(f"     {feature}: {corr:.4f}")
    
    # 6. Load model results if available
    print("\n[6] Model Information:")
    try:
        with open("models/trained_models.pkl", 'rb') as f:
            model_data = pickle.load(f)
            models = model_data['models']
            print(f"   Trained models: {list(models.keys())}")
            
            # Feature importance for Random Forest
            if 'random_forest' in models:
                rf = models['random_forest']
                base = rf
                if hasattr(rf, 'base_estimator'):
                    base = rf.base_estimator
                elif hasattr(rf, 'estimators_') and rf.estimators_:
                    base = rf.estimators_[0]

                if hasattr(base, 'feature_importances_'):
                    print("\n   Random Forest Feature Importance:")
                    importances = base.feature_importances_
                    feature_names = [c for c in feature_cols if c in df.columns]
                    for feat, imp in sorted(zip(feature_names, importances), 
                                          key=lambda x: x[1], reverse=True):
                        print(f"     {feat}: {imp:.4f}")
    except FileNotFoundError:
        print("   Models not found. Run main_sample.py first.")
    
    # 7. Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    skew_ratio = df['is_skewed'].mean()
    if skew_ratio < 0.05:
        print("\n⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print(f"   Only {skew_ratio*100:.2f}% of jobs are skewed.")
        print("   Recommendations:")
        print("   1. Use class_weight='balanced' in model training")
        print("   2. Try SMOTE for oversampling")
        print("   3. Focus on Precision and Recall, not just Accuracy")
        print("   4. Consider using F1-score as primary metric")
    
    print("\n✅ Next Steps:")
    print("   1. Review confusion matrices in models/ folder")
    print("   2. Try class_weight='balanced' in train_model.py")
    print("   3. Experiment with different features")
    print("   4. Add ROC and Precision-Recall curves")
    print("   5. See NEXT_STEPS.md for detailed guidance")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyze_results()
