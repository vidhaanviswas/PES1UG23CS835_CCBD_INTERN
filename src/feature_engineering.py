"""
Feature Engineering Module

This module builds job-level features for training and inference.
It now separates leakage-free pre-execution features from runtime-based labels.
"""

import pandas as pd
import numpy as np
import ast


PRE_EXEC_FEATURES = [
    "num_tasks",
    "scheduling_class",
    "priority",
    "cpu_request_mean",
    "cpu_request_std",
    "memory_request_mean",
    "memory_request_std",
    "disk_space_request_mean",
    "disk_space_request_std",
    "different_machine_constraint_mean",
]

EARLY_EXEC_FEATURES = [
    "early_num_tasks",
    "early_avg_task_runtime",
    "early_max_task_runtime",
    "early_std_task_runtime",
    "num_tasks",
    "scheduling_class",
    "priority",
    "cpu_request_mean",
    "memory_request_mean",
    "disk_space_request_mean",
]


def _coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _mode_or_first(series: pd.Series):
    if series.empty:
        return 0
    try:
        return series.mode().iloc[0]
    except Exception:
        return series.iloc[0]


def _parse_resource_request_value(value):
    """Parse a resource_request entry into (cpu, memory, disk)."""
    if pd.isna(value):
        return (np.nan, np.nan, np.nan)

    data = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return (np.nan, np.nan, np.nan)
        try:
            data = ast.literal_eval(text)
        except Exception:
            return (np.nan, np.nan, np.nan)

    if not isinstance(data, dict):
        return (np.nan, np.nan, np.nan)

    cpu = data.get("cpus", data.get("cpu", np.nan))
    memory = data.get("memory", data.get("mem", np.nan))
    disk = data.get("disk", data.get("disk_space", np.nan))
    return (cpu, memory, disk)


def _parse_constraint_value(value):
    """Convert constraint field to a numeric proxy (0 if empty, else 1)."""
    if pd.isna(value):
        return np.nan

    parsed = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0.0
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return 0.0

    if isinstance(parsed, (list, tuple, set)):
        return 1.0 if len(parsed) > 0 else 0.0

    if isinstance(parsed, dict):
        return 1.0 if len(parsed) > 0 else 0.0

    return 0.0


def extract_pre_execution_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract leakage-free job-level features from pre-execution information only.
    Uses submit events (event_type == 0) when available.
    """
    df_prep = df.copy()

    if "collection_id" in df_prep.columns and "job_id" not in df_prep.columns:
        df_prep["job_id"] = df_prep["collection_id"]
    if "instance_index" in df_prep.columns and "task_index" not in df_prep.columns:
        df_prep["task_index"] = df_prep["instance_index"]
    if "user" in df_prep.columns and "user_name" not in df_prep.columns:
        df_prep["user_name"] = df_prep["user"]
    if "time" in df_prep.columns and "timestamp" not in df_prep.columns:
        df_prep["timestamp"] = df_prep["time"]

    # Direct-format traces may store resources in a nested string dict.
    # Parse this once into numeric columns expected by the pipeline.
    if "resource_request" in df_prep.columns and (
        "cpu_request" not in df_prep.columns or "memory_request" not in df_prep.columns
    ):
        parsed = df_prep["resource_request"].map(_parse_resource_request_value)
        parsed_df = pd.DataFrame(parsed.tolist(), columns=["cpu_request", "memory_request", "disk_space_request"],
                                 index=df_prep.index)
        for col in ["cpu_request", "memory_request", "disk_space_request"]:
            if col not in df_prep.columns:
                df_prep[col] = parsed_df[col]

    if "constraint" in df_prep.columns and "different_machine_constraint" not in df_prep.columns:
        df_prep["different_machine_constraint"] = df_prep["constraint"].map(_parse_constraint_value)

    if "event_type" in df_prep.columns:
        submit_events = df_prep[df_prep["event_type"] == 0].copy()
    else:
        submit_events = df_prep.copy()

    if submit_events.empty:
        raise ValueError("No submit events found to build pre-execution features.")

    numeric_cols = [
        "cpu_request",
        "memory_request",
        "disk_space_request",
        "priority",
        "scheduling_class",
        "different_machine_constraint",
    ]
    submit_events = _coerce_numeric(submit_events, numeric_cols)

    # Derive submit time for time-based splits.
    if "timestamp" in submit_events.columns:
        submit_time = submit_events.groupby("job_id")["timestamp"].min().rename("submit_time")
    elif "start_time" in submit_events.columns:
        submit_time = submit_events.groupby("job_id")["start_time"].min().rename("submit_time")
    else:
        submit_time = pd.Series(dtype=float, name="submit_time")

    if "task_index" in submit_events.columns:
        num_tasks = submit_events.groupby("job_id")["task_index"].nunique().rename("num_tasks")
    else:
        num_tasks = submit_events.groupby("job_id").size().rename("num_tasks")

    agg = {}
    for base_col in ["cpu_request", "memory_request", "disk_space_request"]:
        if base_col in submit_events.columns:
            agg[base_col] = ["mean", "std"]

    if "different_machine_constraint" in submit_events.columns:
        agg["different_machine_constraint"] = "mean"

    if "priority" in submit_events.columns:
        agg["priority"] = "mean"
    if "scheduling_class" in submit_events.columns:
        agg["scheduling_class"] = _mode_or_first
    if "user_name" in submit_events.columns:
        agg["user_name"] = _mode_or_first

    if agg:
        features = submit_events.groupby("job_id").agg(agg).reset_index()
        if isinstance(features.columns, pd.MultiIndex):
            flat_cols = ["job_id"]
            for name, stat in features.columns.tolist()[1:]:
                if stat:
                    flat_cols.append(f"{name}_{stat}")
                else:
                    flat_cols.append(str(name))
            features.columns = flat_cols
    else:
        features = submit_events[["job_id"]].drop_duplicates()

    features = features.merge(num_tasks.reset_index(), on="job_id", how="left")

    if not submit_time.empty:
        features = features.merge(submit_time.reset_index(), on="job_id", how="left")

    # Normalize column names for expected feature naming.
    features = features.rename(
        columns={
            "priority_mean": "priority",
            "scheduling_class__mode_or_first": "scheduling_class",
            "scheduling_class_<lambda>": "scheduling_class",
            "user_name__mode_or_first": "user_name",
            "user_name_<lambda>": "user_name",
        }
    )

    # Fill missing optional columns with defaults.
    for col in PRE_EXEC_FEATURES:
        if col not in features.columns:
            features[col] = 0

    # Ensure numeric types.
    features = _coerce_numeric(features, PRE_EXEC_FEATURES + ["submit_time"])
    for col in PRE_EXEC_FEATURES:
        if col in features.columns:
            features[col] = features[col].fillna(0)
    features["num_tasks"] = features["num_tasks"].fillna(0).astype(int)

    return features


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
    
    # Ensure scheduling_class is numeric if present.
    if 'scheduling_class' in df_encoded.columns:
        df_encoded['scheduling_class'] = pd.to_numeric(
            df_encoded['scheduling_class'], errors='coerce'
        ).fillna(0)
    
    return df_encoded


def get_feature_columns(mode: str = "pre_exec") -> list:
    """Get feature column names for the requested mode."""
    if mode == "pre_exec":
        return PRE_EXEC_FEATURES
    if mode == "early_exec":
        return EARLY_EXEC_FEATURES
    raise ValueError(f"Unknown feature mode: {mode}")


def prepare_features_for_training(df: pd.DataFrame, mode: str = "pre_exec") -> tuple:
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
    feature_cols = get_feature_columns(mode=mode)
    
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
    from preprocessing import clean_task_events, extract_task_runtimes
    from skew_labeling import label_jobs_from_task_runtimes
    
    # Test feature engineering
    try:
        df = load_task_events()
        df_clean = clean_task_events(df)
        df_runtimes = extract_task_runtimes(df_clean)
        pre_exec = extract_pre_execution_features(df_clean)
        pre_exec = encode_categorical_features(pre_exec)
        labels = label_jobs_from_task_runtimes(df_runtimes)
        job_features = pre_exec.merge(labels, on="job_id", how="inner")

        X, y = prepare_features_for_training(job_features, mode="pre_exec")
        
        print("\nFeature summary:")
        print(X.describe())
        print(f"\nLabel distribution:")
        print(y.value_counts())
    except Exception as e:
        print(f"Error: {e}")
