"""
Feature Engineering Module

This module aggregates task-level data to job-level features.
All features are pre-execution features that can be computed before job execution.
"""

import pandas as pd
import numpy as np


def aggregate_to_job_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate task-level data to job-level features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with task runtimes and metadata
        
    Returns:
    --------
    pd.DataFrame
        Job-level dataframe with aggregated features
    """
    print("Aggregating task-level data to job-level...")
    
    # Check which columns are available
    available_cols = df.columns.tolist()
    
    # Required columns
    if 'job_id' not in available_cols:
        raise ValueError("'job_id' column is required for aggregation")
    if 'runtime' not in available_cols:
        raise ValueError("'runtime' column is required for aggregation")
    
    # Count tasks per job first (separate to avoid MultiIndex issues)
    num_tasks = df.groupby('job_id').size().reset_index(name='num_tasks')
    
    # Ensure runtime is numeric before aggregation
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df = df[df['runtime'].notna()]  # Remove any non-numeric runtimes
    
    # Build aggregation dictionary for runtime statistics
    agg_dict = {
        'runtime': ['mean', 'max', 'std']
    }
    
    # Add optional columns if available (ensure they're numeric)
    if 'scheduling_class' in available_cols:
        df['scheduling_class'] = pd.to_numeric(df['scheduling_class'], errors='coerce').fillna(0)
        agg_dict['scheduling_class'] = 'first'
    if 'priority' in available_cols:
        df['priority'] = pd.to_numeric(df['priority'], errors='coerce').fillna(0)
        agg_dict['priority'] = 'mean'
    
    # Perform aggregation
    job_features = df.groupby('job_id').agg(agg_dict).reset_index()
    
    # Flatten MultiIndex columns if needed
    if isinstance(job_features.columns, pd.MultiIndex):
        # Build column names based on aggregation
        col_names = ['job_id']
        
        # Add runtime statistics (mean, max, std)
        col_names.extend(['avg_task_runtime', 'max_task_runtime', 'std_task_runtime'])
        
        # Add optional columns
        if 'scheduling_class' in agg_dict:
            col_names.append('scheduling_class')
        if 'priority' in agg_dict:
            col_names.append('priority')
        
        # Flatten columns
        job_features.columns = col_names
    else:
        # If columns are already named correctly, ensure they match expected names
        # This shouldn't happen with MultiIndex, but handle it anyway
        pass
    
    # Merge num_tasks
    job_features = job_features.merge(num_tasks, on='job_id', how='left')
    
    # Reorder columns to have num_tasks after job_id
    col_order = ['job_id', 'num_tasks', 'avg_task_runtime', 'max_task_runtime', 'std_task_runtime']
    if 'scheduling_class' in job_features.columns:
        col_order.append('scheduling_class')
    if 'priority' in job_features.columns:
        col_order.append('priority')
    
    job_features = job_features[col_order]
    
    # Ensure required columns exist
    required_cols = ['job_id', 'num_tasks', 'avg_task_runtime', 'max_task_runtime', 'std_task_runtime']
    missing_cols = [col for col in required_cols if col not in job_features.columns]
    if missing_cols:
        raise ValueError(f"Failed to create required columns: {missing_cols}")
    
    # Add missing optional columns with defaults
    if 'scheduling_class' not in job_features.columns:
        job_features['scheduling_class'] = 0
    if 'priority' not in job_features.columns:
        job_features['priority'] = 0
    
    # Handle NaN values in std_task_runtime (occurs when job has only 1 task)
    job_features['std_task_runtime'] = job_features['std_task_runtime'].fillna(0)
    
    # Ensure numeric types
    numeric_cols = ['num_tasks', 'avg_task_runtime', 'max_task_runtime', 
                    'std_task_runtime', 'scheduling_class', 'priority']
    for col in numeric_cols:
        job_features[col] = pd.to_numeric(job_features[col], errors='coerce')
    
    # Remove jobs with invalid values
    job_features = job_features[job_features['num_tasks'] > 0]
    job_features = job_features[job_features['avg_task_runtime'] > 0]
    job_features = job_features[job_features['max_task_runtime'] > 0]
    
    print(f"Aggregated to {len(job_features):,} jobs")
    
    return job_features


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features for ML models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Job-level dataframe with features
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with encoded categorical features
    """
    df_encoded = df.copy()
    
    # Encode scheduling_class (if it's categorical, use label encoding)
    # In Google Cluster Traces, scheduling_class is already numeric (0-11)
    # But we'll ensure it's treated as a feature
    if 'scheduling_class' in df_encoded.columns:
        df_encoded['scheduling_class'] = pd.to_numeric(
            df_encoded['scheduling_class'], errors='coerce'
        ).fillna(0)
    
    return df_encoded


def get_feature_columns() -> list:
    """
    Get the list of feature column names for ML models.
    
    Returns:
    --------
    list
        List of feature column names
    """
    return [
        'num_tasks',
        'avg_task_runtime',
        'max_task_runtime',
        'std_task_runtime',
        'scheduling_class',
        'priority'
    ]


def prepare_features_for_training(df: pd.DataFrame) -> tuple:
    """
    Prepare features and labels for ML training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Job-level dataframe with features and labels
        
    Returns:
    --------
    tuple
        (X, y) where X is feature matrix and y is labels
    """
    feature_cols = get_feature_columns()
    
    # Check if all feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")
    
    X = df[feature_cols].copy()
    y = df['is_skewed'].copy() if 'is_skewed' in df.columns else None
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    return X, y


if __name__ == "__main__":
    from data_loader import load_task_events
    from preprocessing import clean_task_events, extract_task_runtimes, prepare_for_aggregation
    from skew_labeling import label_skewed_jobs
    
    # Test feature engineering
    try:
        df = load_task_events()
        df_clean = clean_task_events(df)
        df_runtimes = extract_task_runtimes(df_clean)
        df_prep = prepare_for_aggregation(df_runtimes)
        job_features = aggregate_to_job_level(df_prep)
        job_features = encode_categorical_features(job_features)
        job_features = label_skewed_jobs(job_features)
        
        X, y = prepare_features_for_training(job_features)
        
        print("\nFeature summary:")
        print(X.describe())
        print(f"\nLabel distribution:")
        print(y.value_counts())
    except Exception as e:
        print(f"Error: {e}")
