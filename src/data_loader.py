"""
Data Loader Module

This module handles loading the Google Cluster Workload Traces dataset.
Specifically, it loads task_events.csv and provides utilities for data access.
"""

import pandas as pd
import os
from pathlib import Path


def load_task_events(data_path: str = "data/raw/task_events.csv") -> pd.DataFrame:
    """
    Load the task_events.csv file from the Google Cluster Workload Traces dataset.
    Supports both formats:
    1. Event-based format (task_events.csv with event_type)
    2. Direct format (borg_traces_data.csv with start_time/end_time or duration)
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing task events data
    """
    # Convert to absolute path if relative
    if not os.path.isabs(data_path):
        project_root = Path(__file__).parent.parent
        data_path = project_root / data_path
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please ensure the CSV file is placed in the data/raw/ directory."
        )
    
    print(f"Loading dataset from {data_path}...")
    
    # Try to load with headers first (for borg_traces_data.csv format)
    try:
        df = pd.read_csv(data_path, low_memory=False)
        print(f"Loaded dataset with headers: {list(df.columns)}")
        
        # Check if this is the direct format (has start_time/end_time or duration)
        if 'start_time' in df.columns or 'duration' in df.columns or 'end_time' in df.columns:
            print("Detected direct format (borg_traces_data.csv style)")
            return df
        
        # Otherwise, assume it's event-based format with headers
        print("Detected event-based format with headers")
        return df
        
    except Exception as e1:
        # If that fails, try without headers (original task_events.csv format)
        print("Trying event-based format without headers...")
        try:
            column_names = [
                'timestamp',
                'missing_info',
                'job_id',
                'task_index',
                'machine_id',
                'event_type',
                'user_name',
                'scheduling_class',
                'priority',
                'cpu_request',
                'memory_request',
                'disk_space_request',
                'different_machine_constraint'
            ]
            df = pd.read_csv(data_path, header=None, names=column_names, low_memory=False)
            print(f"Successfully loaded {len(df):,} task events (event-based format)")
            return df
        except Exception as e2:
            raise Exception(f"Error loading dataset. Tried both formats.\nError 1: {str(e1)}\nError 2: {str(e2)}")


def get_sample_data(data_path: str = "data/raw/task_events.csv", 
                   n_rows: int = None) -> pd.DataFrame:
    """
    Load a sample of the dataset (useful for testing).
    Auto-detects format like load_task_events.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV file
    n_rows : int, optional
        Number of rows to load. If None, loads entire dataset.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing sample task events data
    """
    if n_rows is None:
        return load_task_events(data_path)
    
    # Convert to absolute path if relative
    if not os.path.isabs(data_path):
        project_root = Path(__file__).parent.parent
        data_path = project_root / data_path
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}. "
            f"Please ensure the CSV file is placed in the data/raw/ directory."
        )
    
    print(f"Loading sample dataset from {data_path}...")
    
    # Try to load with headers first (for borg_traces_data.csv format)
    try:
        df = pd.read_csv(data_path, nrows=n_rows, low_memory=False)
        print(f"Loaded dataset with headers: {list(df.columns)}")
        
        # Check if this is the direct format (has start_time/end_time or duration)
        if 'start_time' in df.columns or 'duration' in df.columns or 'end_time' in df.columns:
            print("Detected direct format (borg_traces_data.csv style)")
            print(f"Loaded sample of {len(df):,} task events")
            return df
        
        # Otherwise, assume it's event-based format with headers
        print("Detected event-based format with headers")
        print(f"Loaded sample of {len(df):,} task events")
        return df
        
    except Exception as e1:
        # If that fails, try without headers (original task_events.csv format)
        print("Trying event-based format without headers...")
        try:
            column_names = [
                'timestamp',
                'missing_info',
                'job_id',
                'task_index',
                'machine_id',
                'event_type',
                'user_name',
                'scheduling_class',
                'priority',
                'cpu_request',
                'memory_request',
                'disk_space_request',
                'different_machine_constraint'
            ]
            df = pd.read_csv(data_path, header=None, names=column_names, 
                           nrows=n_rows, low_memory=False)
            print(f"Loaded sample of {len(df):,} task events (event-based format)")
            return df
        except Exception as e2:
            raise Exception(f"Error loading dataset. Tried both formats.\nError 1: {str(e1)}\nError 2: {str(e2)}")


if __name__ == "__main__":
    # Test the data loader
    try:
        df = load_task_events()
        print("\nDataset Info:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        print("\nDataset shape:", df.shape)
    except FileNotFoundError as e:
        print(f"Error: {e}")
