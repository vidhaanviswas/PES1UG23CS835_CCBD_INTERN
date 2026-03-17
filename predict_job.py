"""
Production Prediction Script

This script demonstrates how to use the trained model for real-world predictions.
It simulates predicting skew for new jobs before they execute.
"""

import sys
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_model import load_models
from feature_engineering import get_feature_columns


# Model-specific thresholds tuned on held-out real data in hybrid setting.
DEFAULT_INFERENCE_THRESHOLDS = {
    "logistic_regression": 0.10,
    "random_forest": 0.08,
    "xgboost": 0.07,
    "lightgbm": 0.23,
}


def _resolve_threshold(model_name: str, threshold: float = None) -> float:
    """Resolve inference threshold (explicit override wins over model defaults)."""
    if threshold is not None:
        return float(threshold)
    return float(DEFAULT_INFERENCE_THRESHOLDS.get(model_name, 0.5))


def get_default_threshold(model_name: str) -> float:
    """Public helper to fetch default inference threshold for a model."""
    return _resolve_threshold(model_name=model_name, threshold=None)


def predict_job_skew(job_features: dict, model_name: str = 'logistic_regression',
                     threshold: float = None):
    """
    Predict if a job will be skewed based on pre-execution features.
    
    Parameters:
    -----------
    job_features : dict
        Dictionary with job features:
        {
            'num_tasks': int,
            'scheduling_class': int,
            'priority': float,
            'cpu_request_mean': float,
            'cpu_request_std': float,
            'memory_request_mean': float,
            'memory_request_std': float,
            'disk_space_request_mean': float,
            'disk_space_request_std': float,
            'different_machine_constraint_mean': float
        }
    model_name : str
        Model to use (e.g., 'logistic_regression', 'random_forest',
        'xgboost', 'lightgbm')
    threshold : float, optional
        Decision threshold for classifying a job as skewed.
        If None, uses tuned per-model defaults.
        
    Returns:
    --------
    dict
        Prediction results with probability and label
    """
    try:
        # Load trained models
        models, scalers = load_models()
        
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(models.keys())}")
        
        model = models[model_name]
        
        # Get feature columns in correct order
        feature_cols = get_feature_columns(mode="pre_exec")
        
        # Create feature vector
        X = pd.DataFrame([job_features])[feature_cols]
        
        # Handle missing features
        missing = [col for col in feature_cols if col not in X.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Get probability and threshold-based prediction.
        proba = model.predict_proba(X)[0]
        skew_probability = proba[1]  # Probability of class 1 (skewed)
        threshold_used = _resolve_threshold(model_name, threshold)
        prediction = int(skew_probability >= threshold_used)
        
        return {
            'job_id': job_features.get('job_id', 'unknown'),
            'prediction': int(prediction),
            'skew_probability': float(skew_probability),
            'is_skewed': bool(prediction == 1),
            'threshold_used': float(threshold_used),
            'confidence': 'high' if abs(skew_probability - threshold_used) > 0.3 else 'medium' if abs(skew_probability - threshold_used) > 0.15 else 'low',
            'recommendation': get_recommendation(prediction, skew_probability)
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'prediction': None
        }


def get_recommendation(prediction: int, probability: float) -> str:
    """Get actionable recommendation based on prediction."""
    if prediction == 1:  # Skewed
        if probability > 0.8:
            return "HIGH RISK: Job likely to be skewed. Consider increasing parallelism or enabling skew handling."
        else:
            return "MODERATE RISK: Job may be skewed. Monitor closely during execution."
    else:  # Not skewed
        if probability < 0.2:
            return "LOW RISK: Job appears balanced. Normal execution expected."
        else:
            return "LOW-MODERATE RISK: Job likely balanced but monitor for unexpected skew."


def predict_from_dataframe(df: pd.DataFrame, model_name: str = 'logistic_regression',
                           threshold: float = None):
    """
    Predict skew for multiple jobs from a dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with job features
    model_name : str
        Model to use
    threshold : float, optional
        Decision threshold for classifying jobs as skewed.
        If None, uses tuned per-model defaults.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions added
    """
    try:
        models, scalers = load_models()
        model = models[model_name]
        feature_cols = get_feature_columns(mode="pre_exec")
        
        # Prepare features
        X = df[feature_cols].copy()
        
        proba = model.predict_proba(X)
        threshold_used = _resolve_threshold(model_name, threshold)
        predictions = (proba[:, 1] >= threshold_used).astype(int)
        
        # Add predictions to dataframe
        df_result = df.copy()
        df_result['predicted_skew'] = predictions
        df_result['skew_probability'] = proba[:, 1]
        df_result['threshold_used'] = float(threshold_used)
        df_result['prediction_confidence'] = df_result['skew_probability'].apply(
            lambda p: 'high' if abs(p - threshold_used) > 0.3 else 'medium' if abs(p - threshold_used) > 0.15 else 'low'
        )
        
        return df_result
        
    except Exception as e:
        print(f"Error in batch prediction: {e}")
        raise


def main():
    """Demonstrate prediction on sample jobs."""
    print("="*80)
    print("JOB SKEW PREDICTION - Production Demo")
    print("="*80)
    
    # Example 1: Predict single job
    print("\n[Example 1] Predicting skew for a single job...")
    print("-" * 80)
    
    # Simulate a new job with estimated features (from history/metadata)
    new_job = {
        'job_id': 'NEW_JOB_001',
        'num_tasks': 100,
        'scheduling_class': 2,
        'priority': 150,
        'cpu_request_mean': 0.8,
        'cpu_request_std': 0.1,
        'memory_request_mean': 0.6,
        'memory_request_std': 0.2,
        'disk_space_request_mean': 0.4,
        'disk_space_request_std': 0.1,
        'different_machine_constraint_mean': 0.0,
    }
    
    print("Job Features:")
    for key, value in new_job.items():
        print(f"  {key}: {value}")
    
    result = predict_job_skew(new_job, model_name='logistic_regression')
    
    print("\nPrediction Results:")
    print(f"  Will be skewed: {'YES' if result['is_skewed'] else 'NO'}")
    print(f"  Skew probability: {result['skew_probability']:.2%}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Recommendation: {result['recommendation']}")
    
    # Example 2: Batch prediction
    print("\n" + "="*80)
    print("[Example 2] Batch prediction on test data...")
    print("-" * 80)
    
    try:
        # Load processed data
        df = pd.read_csv("data/processed/job_level_data.csv")
        
        # Use a sample of jobs (simulating new jobs)
        df_sample = df.sample(n=10, random_state=42)
        
        # Predict
        df_predictions = predict_from_dataframe(df_sample, model_name='logistic_regression')
        
        print(f"\nPredictions for {len(df_predictions)} jobs:")
        print("\nResults:")
        print(df_predictions[['job_id', 'is_skewed', 'predicted_skew', 
                              'skew_probability', 'prediction_confidence']].to_string(index=False))
        
        # Calculate accuracy
        accuracy = (df_predictions['is_skewed'] == df_predictions['predicted_skew']).mean()
        print(f"\nAccuracy on sample: {accuracy:.2%}")
        
    except FileNotFoundError:
        print("Processed data not found. Run main_sample.py first.")
    
    print("\n" + "="*80)
    print("Prediction demo completed!")
    print("="*80)
    print("\nTo use in production:")
    print("  1. Extract features from job metadata/history")
    print("  2. Call predict_job_skew() with job features")
    print("  3. Take action based on prediction and recommendation")
    print("="*80)


if __name__ == "__main__":
    main()
