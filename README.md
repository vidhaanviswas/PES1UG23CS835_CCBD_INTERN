# Early Prediction of Data Skew in Cloud-Based Big Data Jobs

A research-oriented machine learning pipeline for predicting job-level data skew before full execution, using leakage-free pre-execution features.

## 1. What This Project Does

This project evaluates whether data skew can be predicted early enough to support scheduling and mitigation decisions.

Pipeline components:
- End-to-end preprocessing, feature engineering, labeling, training, and evaluation
- Models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Imbalance handling with SMOTE and probability calibration
- Baseline comparison against a simple rule-based method
- Early-execution experiment utilities
- Policy simulation for skew mitigation
- Explainability and validation scripts
- Logging for reproducibility

## 2. Label Definition and Feature Policy

Skew is defined at job level as:

```text
max_task_runtime >= 2 * avg_task_runtime
```

Labels are created from runtime statistics, but default prediction features are pre-execution metadata only (`mode="pre_exec"`) to avoid leakage.

## 3. Current Status (March 2026)

### Real sample trace
- Jobs: ~4,057
- Positive class: ~68 skewed jobs (about 1.68%)
- Strong class imbalance remains the core challenge

### Synthetic-only training
- Performs near-perfect when train/test come from the same synthetic distribution
- Useful for pipeline validation, not sufficient alone for real-world claims

### Hybrid training (real-train + synthetic) tested on held-out real
- Improved recall and PR-AUC over synthetic-only transfer
- Threshold tuning on held-out real set was completed and saved

Saved artifacts from recent runs:
- `models/trained_models_cross_large_to_small.pkl`
- `models/trained_models_hybrid_real_synth.pkl`
- `outputs/threshold_optimization_hybrid_on_real.csv`

## 4. Repository Structure

```text
CCBD_INTERNSHIP/
├── data/
│   ├── raw/                         # Input traces
│   └── processed/                   # Processed and synthetic job-level datasets
├── models/                          # Trained model bundles and plots
├── outputs/                         # Timestamped logs and experiment tables
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── skew_labeling.py
│   ├── splitters.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── baseline.py
│   └── logger.py
├── main.py                          # Full-data pipeline
├── main_sample.py                   # Sample-data pipeline (recommended start)
├── generate_synthetic_data.py       # Synthetic job generator
├── train_on_synthetic.py            # Synthetic train/eval pipeline
├── predict_job.py                   # Inference helpers (single + batch)
├── mitigation_simulation.py         # Mitigation policy simulator
├── explainability_report.py         # Explainability report
├── early_exec_experiment.py         # Early-execution feature experiment
├── validate_model.py                # Validation checks
├── dashboard_app.py                 # Streamlit dashboard
└── README.md
```

## 5. Setup

### Python
- Recommended: Python 3.10+

### Install dependencies
```powershell
pip install -r requirements.txt
```

Windows venv example:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 6. Data Requirements

Required raw input path:
- `data/raw/task_events.csv`

Loader support (`src/data_loader.py`):
- Event-style task traces (with or without headers)
- Direct runtime style input when duration fields exist

## 7. Quick Start

Recommended first run:
```powershell
python main_sample.py --save-log
```

Full pipeline:
```powershell
python main.py --save-log
```

Key generated outputs:
- `data/processed/job_level_data.csv`
- `models/trained_models.pkl`
- `models/trained_models_template.pkl`
- `models/confusion_matrix_*.png`
- `models/feature_importance_*.png`

## 8. Synthetic Workflow

Generate synthetic jobs:
```powershell
python generate_synthetic_data.py --jobs 50000 --skew-ratio 0.15
```

Train on synthetic data:
```powershell
python train_on_synthetic.py --data data/processed/synthetic_jobs.csv --save-log
```

Small synthetic file for quick checks (already generated in this repo context):
- `data/processed/synthetic_jobs_2x_required.csv`

## 9. Inference Thresholds (Updated)

Inference now uses tuned default probability thresholds by model in `predict_job.py`.

Default thresholds:
- logistic_regression: 0.58
- random_forest: 0.93
- xgboost: 0.69
- lightgbm: 0.69

Behavior:
- `predict_job_skew(...)` uses model-specific threshold by default
- `predict_from_dataframe(...)` uses model-specific threshold by default
- Both functions still support manual threshold override
- Prediction output includes `threshold_used`

## 10. Hybrid and Transfer Experiments

Recent experiment outcomes:
- Synthetic -> synthetic (different synthetic split/files): near-perfect metrics
- Synthetic -> real (`data/processed/job_level_data.csv`): weak transfer
- Hybrid train (real-train + synthetic) -> held-out real: meaningful improvement in recall and PR-AUC

Threshold optimization table is available at:
- `outputs/threshold_optimization_hybrid_on_real.csv`

## 11. Explainability, Validation, and Mitigation

Explainability report:
```powershell
python explainability_report.py --save-log
```

Model validation checks:
```powershell
python validate_model.py
```

Mitigation policy simulation:
```powershell
python mitigation_simulation.py --save-log
```

Prediction demo:
```powershell
python predict_job.py
```

## 12. Logging and Reproducibility

Most scripts support:

```text
--save-log
```

Logs and result artifacts are written to `outputs/`.

Reference guides:
- `LOGGING_GUIDE.md`
- `LOGGING_QUICKREF.md`
- `LOGGING_IMPLEMENTATION.md`

## 13. Suggested Run Order

1. `python main_sample.py --save-log`
2. `python generate_synthetic_data.py --jobs 50000 --skew-ratio 0.15`
3. `python train_on_synthetic.py --data data/processed/synthetic_jobs.csv --save-log`
4. `python explainability_report.py --save-log`
5. `python mitigation_simulation.py --save-log`
6. `python early_exec_experiment.py --save-log`

## 14. Known Limitations

- Very small positive class in current real sample setup
- Synthetic performance can significantly overestimate real deployment performance
- Mitigation module is a simulator (not a cluster-integrated scheduler)
- Publication-grade claims require larger/more diverse real traces

## 15. Troubleshooting

### Missing dataset
Ensure this file exists:
- `data/raw/task_events.csv`

### Import/dependency issues
```powershell
pip install -r requirements.txt
```

### Too few positive labels
- Increase sample size
- Test alternative traces (`ALTERNATIVE_DATASETS.md`)
- Use synthetic/hybrid training for methodology development

### Windows venv issues
```powershell
.\.venv\Scripts\Activate.ps1
```

## 16. Additional Documentation

- `PROJECT_REPORT.md`
- `ARCHITECTURE.md`
- `ARCHITECTURE_SUMMARY.md`
- `SOLUTIONS_FOR_LIMITED_DATA.md`
- `ALTERNATIVE_DATASETS.md`
- `QUICK_START_SOLUTIONS.md`
- `SYNTHETIC_DATA_RESULTS.md`
- `RESULTS_COMPARISON.md`
- `PAPER_GUIDE.md`

## 17. Citation

If you use this repository in research, cite the project topic:

```text
Early Prediction of Data Skew in Cloud-Based Big Data Jobs Using Lightweight Machine Learning Models
```

## 18. License and Usage

This repository is currently positioned for research/academic use.

If you plan to publish or redistribute, add an explicit license file and verify third-party dataset terms.
