"""
Explainability Report

Computes feature importance using SHAP (if available) or permutation importance.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from train_model import load_models, split_data
from feature_engineering import prepare_features_for_training, get_feature_columns
from logger import setup_logging, close_logging


def _unwrap_model(model):
    if hasattr(model, "base_estimator"):
        return model.base_estimator
    if hasattr(model, "estimators_") and model.estimators_:
        return model.estimators_[0]
    return model


def run():
    df = pd.read_csv("data/processed/job_level_data.csv")
    X, y = prepare_features_for_training(df, mode="pre_exec")
    _, X_test, _, y_test = split_data(X, y)

    models, _ = load_models()
    feature_names = get_feature_columns(mode="pre_exec")

    try:
        import shap
        use_shap = True
    except Exception:
        use_shap = False

    for name, model in models.items():
        base = _unwrap_model(model)
        print("=" * 80)
        print(f"EXPLAINABILITY: {name}")
        print("=" * 80)

        if use_shap and hasattr(base, "predict_proba"):
            try:
                explainer = shap.Explainer(base, X_test)
                shap_values = explainer(X_test)
                mean_abs = shap_values.abs.mean(0).values
                ranking = sorted(zip(feature_names, mean_abs), key=lambda x: x[1], reverse=True)
                print("Top features (SHAP mean |value|):")
                for feat, val in ranking[:10]:
                    print(f"  {feat}: {val:.4f}")
                continue
            except Exception as e:
                print(f"SHAP failed for {name}: {e}")

        # Fallback: permutation importance
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            ranking = sorted(zip(feature_names, r.importances_mean), key=lambda x: x[1], reverse=True)
            print("Top features (permutation importance):")
            for feat, val in ranking[:10]:
                print(f"  {feat}: {val:.4f}")
        except Exception as e:
            print(f"Permutation importance failed for {name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate explainability report for trained models")
    parser.add_argument("--save-log", action="store_true", 
                        help="Save terminal output to log file in outputs/ directory")
    args = parser.parse_args()
    
    logger = None
    if args.save_log:
        logger = setup_logging("explainability_report")
    
    try:
        run()
    finally:
        if logger:
            close_logging(logger)
