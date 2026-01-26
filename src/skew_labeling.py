"""
Skew Labeling Module

This module labels jobs as skewed or non-skewed based on the definition:
A job is skewed if max_task_runtime >= 2 * average_task_runtime
"""

import pandas as pd
import numpy as np


def label_skewed_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label jobs as skewed (1) or non-skewed (0).
    
    Definition: A job is labeled as skewed if:
        max_task_runtime >= 2 * average_task_runtime
    
    Parameters:
    -----------
    df : pd.DataFrame
        Job-level dataframe with avg_task_runtime and max_task_runtime
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with 'is_skewed' column added
    """
    print("Labeling skewed jobs...")
    
    df_labeled = df.copy()
    
    # Check required columns
    required_cols = ['avg_task_runtime', 'max_task_runtime']
    missing_cols = [col for col in required_cols if col not in df_labeled.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Compute skew condition: max_task_runtime >= 2 * avg_task_runtime
    df_labeled['is_skewed'] = (
        df_labeled['max_task_runtime'] >= 2 * df_labeled['avg_task_runtime']
    ).astype(int)
    
    # Print label distribution
    skew_counts = df_labeled['is_skewed'].value_counts()
    print(f"\nSkew label distribution:")
    print(f"  Non-skewed (0): {skew_counts.get(0, 0):,} jobs "
          f"({skew_counts.get(0, 0)/len(df_labeled)*100:.2f}%)")
    print(f"  Skewed (1): {skew_counts.get(1, 0):,} jobs "
          f"({skew_counts.get(1, 0)/len(df_labeled)*100:.2f}%)")
    
    return df_labeled


def get_skew_statistics(df: pd.DataFrame) -> dict:
    """
    Get statistics about data skew in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Job-level dataframe with 'is_skewed' column
        
    Returns:
    --------
    dict
        Dictionary with skew statistics
    """
    if 'is_skewed' not in df.columns:
        raise ValueError("Dataframe must have 'is_skewed' column")
    
    stats = {
        'total_jobs': len(df),
        'skewed_jobs': df['is_skewed'].sum(),
        'non_skewed_jobs': (df['is_skewed'] == 0).sum(),
        'skew_ratio': df['is_skewed'].mean(),
        'avg_max_runtime_skewed': df[df['is_skewed'] == 1]['max_task_runtime'].mean() 
                                  if (df['is_skewed'] == 1).any() else 0,
        'avg_max_runtime_non_skewed': df[df['is_skewed'] == 0]['max_task_runtime'].mean() 
                                      if (df['is_skewed'] == 0).any() else 0,
    }
    
    return stats


if __name__ == "__main__":
    from data_loader import load_task_events
    from preprocessing import clean_task_events, extract_task_runtimes, prepare_for_aggregation
    from feature_engineering import aggregate_to_job_level, encode_categorical_features
    
    # Test skew labeling
    try:
        df = load_task_events()
        df_clean = clean_task_events(df)
        df_runtimes = extract_task_runtimes(df_clean)
        df_prep = prepare_for_aggregation(df_runtimes)
        job_features = aggregate_to_job_level(df_prep)
        job_features = encode_categorical_features(job_features)
        job_labeled = label_skewed_jobs(job_features)
        
        stats = get_skew_statistics(job_labeled)
        print("\nSkew statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
