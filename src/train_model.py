"""
Model Training Module

This module trains machine learning models for predicting data skew.
Models: Logistic Regression and Random Forest Classifier
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os
from pathlib import Path


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Labels
    test_size : float
        Proportion of data to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Data split:")
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Testing set: {len(X_test):,} samples")
    
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series,
                              X_test: pd.DataFrame = None) -> tuple:
    """
    Train a Logistic Regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame, optional
        Test features for scaling consistency
        
    Returns:
    --------
    tuple
        (model, scaler) - trained model and fitted scaler
    """
    print("\nTraining Logistic Regression model...")
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model.fit(X_train_scaled, y_train)
    
    print("Logistic Regression training completed")
    
    return model, scaler


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       n_estimators: int = 100, max_depth: int = 10,
                       random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    n_estimators : int
        Number of trees in the forest
    max_depth : int
        Maximum depth of trees
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    RandomForestClassifier
        Trained Random Forest model
    """
    print("\nTraining Random Forest Classifier model...")
    
    # Random Forest doesn't require scaling
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    print("Random Forest training completed")
    
    return model


def save_models(models: dict, scalers: dict, save_path: str = "models/trained_models.pkl"):
    """
    Save trained models and scalers to disk.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models (e.g., {'lr': model, 'rf': model})
    scalers : dict
        Dictionary of scalers (e.g., {'lr': scaler})
    save_path : str
        Path to save the models
    """
    # Convert to absolute path if relative
    if not os.path.isabs(save_path):
        project_root = Path(__file__).parent.parent
        save_path = project_root / save_path
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save models and scalers
    model_data = {
        'models': models,
        'scalers': scalers
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nModels saved to {save_path}")


def load_models(load_path: str = "models/trained_models.pkl") -> tuple:
    """
    Load trained models and scalers from disk.
    
    Parameters:
    -----------
    load_path : str
        Path to load the models from
        
    Returns:
    --------
    tuple
        (models, scalers) dictionaries
    """
    # Convert to absolute path if relative
    if not os.path.isabs(load_path):
        project_root = Path(__file__).parent.parent
        load_path = project_root / load_path
    
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Models not found at {load_path}")
    
    with open(load_path, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data['models'], model_data['scalers']


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame = None,
                    save_path: str = "models/trained_models.pkl") -> tuple:
    """
    Train all ML models and save them.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_test : pd.DataFrame, optional
        Test features for scaling consistency
    save_path : str
        Path to save the models
        
    Returns:
    --------
    tuple
        (models, scalers) dictionaries
    """
    models = {}
    scalers = {}
    
    # Train Logistic Regression
    lr_model, lr_scaler = train_logistic_regression(X_train, y_train, X_test)
    models['logistic_regression'] = lr_model
    scalers['logistic_regression'] = lr_scaler
    
    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['random_forest'] = rf_model
    
    # Save models
    save_models(models, scalers, save_path)
    
    return models, scalers


if __name__ == "__main__":
    from data_loader import load_task_events
    from preprocessing import clean_task_events, extract_task_runtimes, prepare_for_aggregation
    from feature_engineering import aggregate_to_job_level, encode_categorical_features, prepare_features_for_training
    from skew_labeling import label_skewed_jobs
    
    # Test model training
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
        
        models, scalers = train_all_models(X_train, y_train, X_test)
        
        print("\nModel training completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
