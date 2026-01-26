"""
Preprocessing Module

This module handles data cleaning, validation, and preparation for feature engineering.
"""

import pandas as pd
import numpy as np


def clean_task_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the task events dataframe.
    Supports both event-based and direct format datasets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw task events dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("Cleaning task events data...")
    original_size = len(df)
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Debug: Print available columns
    print(f"Available columns: {list(df_clean.columns)}")
    
    # Detect format: event-based (has event_type) or direct (has start_time/end_time/duration)
    is_event_based = 'event_type' in df_clean.columns
    is_direct_format = 'start_time' in df_clean.columns or 'end_time' in df_clean.columns or 'duration' in df_clean.columns
    
    print(f"Format detection: is_event_based={is_event_based}, is_direct_format={is_direct_format}")
    
    if is_event_based:
        # Event-based format cleaning
        # Remove rows with missing critical information
        if 'missing_info' in df_clean.columns:
            df_clean = df_clean[df_clean['missing_info'] == 0]
        
        # Remove rows with invalid job_id or task_index
        df_clean = df_clean[df_clean['job_id'].notna()]
        df_clean = df_clean[df_clean['task_index'].notna()]
        
        # Remove rows with negative values for critical fields
        df_clean = df_clean[df_clean['job_id'] >= 0]
        df_clean = df_clean[df_clean['task_index'] >= 0]
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean[df_clean['timestamp'] >= 0]
        
        # Handle missing values in numeric columns
        numeric_cols = ['cpu_request', 'memory_request', 'disk_space_request', 
                        'priority', 'scheduling_class']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                if df_clean[col].notna().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                else:
                    df_clean[col] = df_clean[col].fillna(0)
        
        # Ensure event_type is valid (0-8)
        if 'event_type' in df_clean.columns:
            df_clean = df_clean[df_clean['event_type'].isin(range(9))]
    
    elif is_direct_format:
        # Direct format cleaning (borg_traces_data.csv style)
        print("Applying direct format cleaning...")
        initial_size = len(df_clean)
        
        # Map alternative column names for Google 2019 Cluster Sample
        # collection_id -> job_id, instance_index -> task_index
        if 'collection_id' in df_clean.columns and 'job_id' not in df_clean.columns:
            df_clean['job_id'] = df_clean['collection_id']
        if 'instance_index' in df_clean.columns and 'task_index' not in df_clean.columns:
            df_clean['task_index'] = df_clean['instance_index']
        
        # Remove rows with invalid job_id (must exist and be non-null)
        # Check both job_id and collection_id
        job_id_col = None
        if 'job_id' in df_clean.columns:
            job_id_col = 'job_id'
        elif 'collection_id' in df_clean.columns:
            job_id_col = 'collection_id'
        
        if job_id_col:
            before = len(df_clean)
            df_clean = df_clean[df_clean[job_id_col].notna()]
            print(f"  Removed {before - len(df_clean)} rows with null {job_id_col}")
            
            # If we used collection_id, also create job_id column
            if job_id_col == 'collection_id':
                df_clean['job_id'] = df_clean['collection_id']
            
            # Try to convert to numeric, but don't filter if conversion fails
            # (job_id might be string identifiers like "job_123")
            try:
                numeric_job_id = pd.to_numeric(df_clean['job_id'], errors='coerce')
                # Only apply numeric filtering if most values converted successfully
                if numeric_job_id.notna().sum() > len(df_clean) * 0.5:  # More than 50% converted
                    # Keep numeric ones and convert them
                    df_clean.loc[numeric_job_id.notna(), 'job_id'] = numeric_job_id[numeric_job_id.notna()]
                    # Only filter numeric ones by >= 0
                    mask = numeric_job_id.notna() & (numeric_job_id >= 0)
                    mask = mask | numeric_job_id.isna()  # Keep non-numeric ones too
                    df_clean = df_clean[mask]
                    print(f"  Applied numeric filtering for job_id")
            except Exception as e:
                print(f"  Could not convert job_id to numeric (keeping as-is): {e}")
        else:
            print("  Warning: No job_id or collection_id column found")
        
        # Handle task_index/instance_index if present (optional - don't require it)
        task_idx_col = None
        if 'task_index' in df_clean.columns:
            task_idx_col = 'task_index'
        elif 'instance_index' in df_clean.columns:
            task_idx_col = 'instance_index'
        
        if task_idx_col:
            if task_idx_col == 'instance_index':
                df_clean['task_index'] = df_clean['instance_index']
        
        print(f"  Direct format cleaning: {initial_size:,} -> {len(df_clean):,} rows")
    
    else:
        # Unknown format - try basic cleaning (minimal filtering)
        print("Warning: Unknown format, applying basic cleaning...")
        initial_size = len(df_clean)
        if 'job_id' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[df_clean['job_id'].notna()]
            print(f"  Removed {before - len(df_clean)} rows with null job_id")
            # Try numeric conversion but don't filter strictly
            try:
                numeric_job_id = pd.to_numeric(df_clean['job_id'], errors='coerce')
                if numeric_job_id.notna().sum() > len(df_clean) * 0.5:
                    df_clean.loc[numeric_job_id.notna(), 'job_id'] = numeric_job_id[numeric_job_id.notna()]
            except:
                pass
        print(f"  Basic cleaning: {initial_size:,} -> {len(df_clean):,} rows")
    
    print(f"Cleaned data: {original_size:,} -> {len(df_clean):,} rows "
          f"({(1 - len(df_clean)/original_size)*100:.2f}% removed)")
    
    return df_clean


def extract_task_runtimes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract task start and end times to compute runtime.
    Supports both event-based and direct format datasets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned task events dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with task runtimes computed
    """
    print("Extracting task runtimes...")
    
    # Map column names for Google 2019 Cluster Sample format
    # collection_id -> job_id, instance_index -> task_index
    df_mapped = df.copy()
    if 'collection_id' in df_mapped.columns and 'job_id' not in df_mapped.columns:
        df_mapped['job_id'] = df_mapped['collection_id']
    if 'instance_index' in df_mapped.columns and 'task_index' not in df_mapped.columns:
        df_mapped['task_index'] = df_mapped['instance_index']
    
    # Check if this is direct format (has duration or start_time/end_time)
    if 'duration' in df_mapped.columns:
        # Direct format with duration already computed
        print("Using existing 'duration' column")
        df_runtimes = df_mapped.copy()
        df_runtimes['runtime'] = pd.to_numeric(df_runtimes['duration'], errors='coerce')
        # Convert from microseconds to seconds if needed (if values are very large)
        if len(df_runtimes) > 0 and df_runtimes['runtime'].notna().sum() > 0:
            max_runtime = df_runtimes['runtime'].max()
            if pd.notna(max_runtime) and max_runtime > 1e10:
                df_runtimes['runtime'] = df_runtimes['runtime'] / 1e6  # microseconds to seconds
                print("Converted duration from microseconds to seconds")
        
    elif 'start_time' in df_mapped.columns and 'end_time' in df_mapped.columns:
        # Direct format with start_time and end_time
        print("Computing duration from start_time and end_time")
        df_runtimes = df_mapped.copy()
        df_runtimes['start_time'] = pd.to_numeric(df_runtimes['start_time'], errors='coerce')
        df_runtimes['end_time'] = pd.to_numeric(df_runtimes['end_time'], errors='coerce')
        df_runtimes['runtime'] = df_runtimes['end_time'] - df_runtimes['start_time']
        # Convert from microseconds to seconds if needed
        if len(df_runtimes) > 0 and df_runtimes['runtime'].notna().sum() > 0:
            max_runtime = df_runtimes['runtime'].max()
            if pd.notna(max_runtime) and max_runtime > 1e10:
                df_runtimes['runtime'] = df_runtimes['runtime'] / 1e6
                print("Converted runtime from microseconds to seconds")
    
    elif 'event_type' in df_mapped.columns:
        # Event-based format - extract from submit/finish events
        print("Extracting from event-based format...")
        submit_events = df_mapped[df_mapped['event_type'] == 0].copy()
        finish_events = df_mapped[df_mapped['event_type'] == 4].copy()
        
        # Create unique task identifier
        submit_events['task_id'] = (submit_events['job_id'].astype(str) + '_' + 
                                    submit_events['task_index'].astype(str))
        finish_events['task_id'] = (finish_events['job_id'].astype(str) + '_' + 
                                    finish_events['task_index'].astype(str))
        
        # Get start times from submit events
        cols_to_keep = ['task_id', 'job_id', 'task_index', 'timestamp']
        if 'scheduling_class' in submit_events.columns:
            cols_to_keep.append('scheduling_class')
        if 'priority' in submit_events.columns:
            cols_to_keep.append('priority')
        
        start_times = submit_events[cols_to_keep].copy()
        start_times.rename(columns={'timestamp': 'start_time'}, inplace=True)
        
        # Get end times from finish events
        end_times = finish_events[['task_id', 'timestamp']].copy()
        end_times.rename(columns={'timestamp': 'end_time'}, inplace=True)
        
        # Merge start and end times
        task_runtimes = start_times.merge(end_times, on='task_id', how='inner')
        
        # Compute runtime
        task_runtimes['runtime'] = task_runtimes['end_time'] - task_runtimes['start_time']
        df_runtimes = task_runtimes
    
    else:
        raise ValueError("Cannot extract runtime: dataset must have 'duration', 'start_time/end_time', or 'event_type' column")
    
    # Ensure runtime is numeric
    df_runtimes['runtime'] = pd.to_numeric(df_runtimes['runtime'], errors='coerce')
    
    # Filter out invalid runtimes (negative, zero, or NaN)
    df_runtimes = df_runtimes[df_runtimes['runtime'].notna()]
    df_runtimes = df_runtimes[df_runtimes['runtime'] > 0]
    
    print(f"Extracted runtimes for {len(df_runtimes):,} tasks")
    
    return df_runtimes


def prepare_for_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare cleaned data for job-level aggregation.
    Adds missing optional columns with default values.
    Maps alternative column names (collection_id -> job_id, instance_index -> task_index).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with task runtimes
        
    Returns:
    --------
    pd.DataFrame
        Dataframe ready for aggregation
    """
    df_prep = df.copy()
    
    # Map alternative column names to standard names
    # Google 2019 Cluster Sample uses: collection_id -> job_id, instance_index -> task_index
    if 'collection_id' in df_prep.columns and 'job_id' not in df_prep.columns:
        print("Mapping 'collection_id' -> 'job_id'")
        df_prep['job_id'] = df_prep['collection_id']
    
    if 'instance_index' in df_prep.columns and 'task_index' not in df_prep.columns:
        print("Mapping 'instance_index' -> 'task_index'")
        df_prep['task_index'] = df_prep['instance_index']
    
    # Required columns
    required_cols = ['job_id', 'runtime']
    missing_required = [col for col in required_cols if col not in df_prep.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}. Available columns: {list(df_prep.columns)}")
    
    # Add task_index if missing (create sequential index)
    if 'task_index' not in df_prep.columns:
        print("Warning: 'task_index' not found, creating sequential index per job")
        df_prep['task_index'] = df_prep.groupby('job_id').cumcount()
    
    # Add optional columns with default values if missing
    if 'scheduling_class' not in df_prep.columns:
        print("Warning: 'scheduling_class' not found, using default value 0")
        df_prep['scheduling_class'] = 0
    
    if 'priority' not in df_prep.columns:
        print("Warning: 'priority' not found, using default value 0")
        df_prep['priority'] = 0
    
    # Ensure numeric types
    df_prep['scheduling_class'] = pd.to_numeric(df_prep['scheduling_class'], errors='coerce').fillna(0)
    df_prep['priority'] = pd.to_numeric(df_prep['priority'], errors='coerce').fillna(0)
    
    # Ensure job_id is numeric or can be used for grouping
    # Try to convert to numeric, but keep as-is if it fails (might be string IDs)
    try:
        numeric_job_id = pd.to_numeric(df_prep['job_id'], errors='coerce')
        if numeric_job_id.notna().sum() > len(df_prep) * 0.5:
            df_prep.loc[numeric_job_id.notna(), 'job_id'] = numeric_job_id[numeric_job_id.notna()]
    except:
        pass  # Keep job_id as-is if conversion fails
    
    # Remove any remaining invalid values
    df_prep = df_prep[df_prep['runtime'].notna()].copy()
    df_prep = df_prep[df_prep['runtime'] > 0]
    
    return df_prep


if __name__ == "__main__":
    from data_loader import load_task_events
    
    # Test preprocessing
    try:
        df = load_task_events()
        df_clean = clean_task_events(df)
        df_runtimes = extract_task_runtimes(df_clean)
        df_prep = prepare_for_aggregation(df_runtimes)
        
        print("\nPreprocessed data summary:")
        print(df_prep.describe())
        print(f"\nNumber of unique jobs: {df_prep['job_id'].nunique():,}")
    except Exception as e:
        print(f"Error: {e}")
