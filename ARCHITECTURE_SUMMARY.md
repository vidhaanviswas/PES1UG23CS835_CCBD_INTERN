# Architecture Summary - Quick Reference

> 📝 **Output Logging**: Use `--save-log` flag with any script to save terminal output to `outputs/` directory. See [LOGGING_GUIDE.md](LOGGING_GUIDE.md) for details.

## System Overview

```
                    ┌─────────────────────┐
                    │   Raw Dataset        │
                    │  (task_events.csv)   │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Data Processing    │
                    │  Pipeline           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Job-Level Features │
                    │  + Skew Labels      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  ML Training        │
                    │  (LR + RF)          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Trained Models     │
                    │  + Evaluations      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Predictions        │
                    │  + Dashboard        │
                    └─────────────────────┘
```

## Core Pipeline (8 Steps)

```
1. Load Data      → 2. Preprocess  → 3. Engineer Features
                                              │
                                              ▼
4. Label Skew    → 5. Prepare Data → 6. Train Models
                                              │
                                              ▼
7. Evaluate      → 8. Compare Baseline
```

## Key Components

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Data       │───▶│  Feature     │───▶│   ML         │
│  Processing  │    │  Engineering │    │  Training    │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
  Clean & Validate    Aggregate Tasks    Train & Evaluate
```

## Technology Stack

```
Python 3
├── pandas (Data Processing)
├── scikit-learn (ML Models)
├── matplotlib/seaborn (Visualization)
└── streamlit (Dashboard - Optional)
```

## Input/Output

**INPUT:**
- `data/raw/task_events.csv` (Task-level data)

**OUTPUT:**
- `data/processed/job_level_data.csv` (Job-level features + labels)
- `models/trained_models.pkl` (Trained models)
- `models/*.png` (Visualizations)
- Console metrics (Accuracy, Precision, Recall, F1)

## Models

1. **Logistic Regression** (with StandardScaler)
2. **Random Forest** (100 trees, max_depth=10)
3. **Baseline** (Rule-based threshold)

## Features (Pre-Execution)

1. num_tasks
2. scheduling_class
3. priority
4. cpu_request_mean / cpu_request_std
5. memory_request_mean / memory_request_std
6. disk_space_request_mean / disk_space_request_std
7. different_machine_constraint_mean
