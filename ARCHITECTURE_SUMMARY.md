# Architecture Summary - Quick Reference

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
2. avg_task_runtime
3. max_task_runtime
4. std_task_runtime
5. scheduling_class
6. priority
