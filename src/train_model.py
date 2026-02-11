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
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import pickle
import os
from pathlib import Path

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    from imblearn.over_sampling import SMOTE
except Exception:
    SMOTE = None


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
                              class_weight: str = "balanced"):
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
    
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=1000, solver="lbfgs", class_weight=class_weight)),
        ]
    )
    model.fit(X_train, y_train)
    
    print("Logistic Regression training completed")
    
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       n_estimators: int = 200, max_depth: int = 12,
                       random_state: int = 42,
                       class_weight: str = "balanced") -> RandomForestClassifier:
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
        n_jobs=-1,
        class_weight=class_weight,
    )
    
    model.fit(X_train, y_train)
    
    print("Random Forest training completed")
    
    return model


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series,
                  random_state: int = 42) -> object:
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed")

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = float(neg / pos) if pos else 1.0

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series,
                   random_state: int = 42) -> object:
    if LGBMClassifier is None:
        raise ImportError("lightgbm is not installed")

    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=random_state,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model


def apply_smote_if_available(X_train: pd.DataFrame, y_train: pd.Series) -> tuple:
    if SMOTE is None:
        print("SMOTE not available; proceeding without oversampling")
        return X_train, y_train
    sampler = SMOTE(random_state=42)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res


def calibrate_model(model, X_train: pd.DataFrame, y_train: pd.Series,
                    method: str = "isotonic"):
    calibrated = CalibratedClassifierCV(model, method=method, cv=3)
    calibrated.fit(X_train, y_train)
    return calibrated


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
        "models": models,
        "scalers": scalers,
        "metadata": {
            "version": "2.0",
            "calibrated": True,
        },
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
    
    models = model_data.get("models", {})
    scalers = model_data.get("scalers", {})
    return models, scalers


def train_all_models(X_train: pd.DataFrame, y_train: pd.Series,
                    use_smote: bool = False,
                    calibrate: bool = True,
                    calibration_method: str = "isotonic",
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

    if use_smote:
        X_train, y_train = apply_smote_if_available(X_train, y_train)

    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    if calibrate:
        lr_model = calibrate_model(lr_model, X_train, y_train, method=calibration_method)
    models["logistic_regression"] = lr_model

    # Train Random Forest
    rf_model = train_random_forest(X_train, y_train)
    if calibrate:
        rf_model = calibrate_model(rf_model, X_train, y_train, method=calibration_method)
    models["random_forest"] = rf_model

    # Train XGBoost
    try:
        xgb_model = train_xgboost(X_train, y_train)
        if calibrate:
            xgb_model = calibrate_model(xgb_model, X_train, y_train, method=calibration_method)
        models["xgboost"] = xgb_model
    except Exception as e:
        print(f"Skipping XGBoost: {e}")

    # Train LightGBM
    try:
        lgbm_model = train_lightgbm(X_train, y_train)
        if calibrate:
            lgbm_model = calibrate_model(lgbm_model, X_train, y_train, method=calibration_method)
        models["lightgbm"] = lgbm_model
    except Exception as e:
        print(f"Skipping LightGBM: {e}")

    save_models(models, scalers, save_path)
    return models, scalers


if __name__ == "__main__":
    from data_loader import load_task_events
    from preprocessing import clean_task_events, extract_task_runtimes
    from feature_engineering import extract_pre_execution_features, encode_categorical_features, prepare_features_for_training
    from skew_labeling import label_jobs_from_task_runtimes
    
    # Test model training
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
        
        models, scalers = train_all_models(X_train, y_train)
        
        print("\nModel training completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
