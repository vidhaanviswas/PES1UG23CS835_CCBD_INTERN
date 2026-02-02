# High-Level Architecture

## Early Prediction of Data Skew in Cloud-Based Big Data Jobs Using Lightweight Machine Learning Models

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RESEARCH PROJECT ARCHITECTURE                │
│         Early Prediction of Data Skew in Big Data Jobs          │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌──────────────┐
│   Raw Data   │────────▶│  Processing  │────────▶│  ML Models  │
│  (CSV Files) │         │   Pipeline   │         │  (Trained)   │
└──────────────┘         └──────────────┘         └──────────────┘
                                                         │
                                                         ▼
                                                  ┌──────────────┐
                                                  │ Predictions  │
                                                  │ & Dashboard  │
                                                  └──────────────┘
```

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION LAYER                           │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Google Cluster Workload Traces (2019 Sample)                    │   │
│  │  - task_events.csv (Raw task-level data)                         │   │
│  │  - Format: Event-based or Direct (auto-detected)                 │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PROCESSING LAYER                            │
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Data       │───▶│ Preprocessing│───▶│   Feature   │               │
│  │   Loader     │    │   & Cleaning │    │ Engineering  │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│       │                     │                     │                     │
│       │                     │                     │                     │
│       ▼                     ▼                     ▼                     │
│  • Load CSV          • Remove invalid    • Aggregate task→job           │
│  • Auto-detect       • Handle missing    • Compute features:            │
│    format            • Extract runtime     - num_tasks                  │
│  • Column mapping    • Validate data       - avg_task_runtime           │
│                       • Type conversion     - max_task_runtime          │
│                                         - std_task_runtime              │
│                                         - scheduling_class              │
│                                         - priority                      │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         SKEW LABELING LAYER                             │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Label Definition: max_task_runtime >= 2 * avg_task_runtime      │   │
│  │  • Skewed (1): Jobs with significant runtime imbalance           │   │
│  │  • Non-skewed (0): All other jobs                                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Processed Job-Level Dataset (job_level_data.csv)                │   │
│  │  - Features + Labels ready for ML training                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        MACHINE LEARNING LAYER                          │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    TRAINING PIPELINE                             │  │
│  │                                                                  │  │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐        │  │
│  │  │   Train/Test │───▶│   Feature    │───▶│   Model      │       │  │
│  │  │   Split      │    │   Scaling    │    │   Training   │        │  │
│  │  │  (80/20)     │    │  (Standard)  │    │              │        │  │
│  │  └──────────────┘    └──────────────┘    └──────────────┘        │  │
│  │                                                                  │  │
│  │  Models Trained:                                                 │  │
│  │  • Logistic Regression (with StandardScaler)                     │  │
│  │  • Random Forest Classifier (100 trees, max_depth=10)            │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    EVALUATION PIPELINE                           │  │
│  │                                                                  │  │
│  │  • Accuracy, Precision, Recall, F1-Score                         │  │
│  │  • Confusion Matrix                                              │  │
│  │  • Feature Importance (Random Forest)                            │  │
│  │  • Baseline Comparison (Rule-based)                              │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Trained Models (trained_models.pkl)                             │  │
│  │  - Logistic Regression model + scaler                            │  │
│  │  - Random Forest model                                           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        PREDICTION & VISUALIZATION LAYER                │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    PRODUCTION PREDICTION                         │  │
│  │                                                                  │  │
│  │  • Single job prediction (predict_job.py)                        │  │
│  │  • Batch prediction (predict_from_dataframe)                     │  │
│  │  • Model validation (validate_model.py)                          │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                         │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    INTERACTIVE DASHBOARD                         │  │
│  │                    (Streamlit - Optional)                        │  │
│  │                                                                  │  │
│  │  • Live job simulation (job_simulator.py)                        │  │
│  │  • Real-time predictions                                         │  │
│  │  • Visualizations (charts, tables)                               │  │
│  │  • Non-technical explanations                                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PROJECT COMPONENTS                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CORE PIPELINE MODULES (src/)                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │ data_loader │  │preprocessing │  │   feature    │                │
│  │    .py      │─▶│     .py      │─▶│ engineering │                │
│  └─────────────┘  └──────────────┘  │     .py      │                │
│                                     └──────────────┘                │
│                                             │                       │
│                                             ▼                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   skew       │  │  train_model │  │  evaluate    │               │
│  │  labeling    │─▶│     .py      │─▶│  _model.py  │               │
│  │    .py       │  └──────────────┘  └──────────────┘               │
│  └──────────────┘                           │                       │
│                                             ▼                       │
│                                    ┌──────────────┐                 │
│                                    │  baseline.py │                 │
│                                    └──────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  EXECUTION SCRIPTS                                                  │
├─────────────────────────────────────────────────────────────────────┤
│  • main.py          - Full dataset pipeline                         │
│  • main_sample.py   - Sample data pipeline (faster)                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  UTILITY MODULES (Optional)                                         │
├─────────────────────────────────────────────────────────────────────┤
│  • predict_job.py           - Production prediction interface       │
│  • validate_model.py        - Model validation & testing            │
│  • analyze_results.py       - Results analysis                      │
│  • create_advanced_plots.py - Advanced visualizations               │
│  • dashboard_app.py         - Streamlit dashboard                   │
│  • job_simulator.py         - Synthetic job generation              │
│  • demo_production_use.py   - Production use examples               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW PIPELINE                          │
└─────────────────────────────────────────────────────────────────────┘

INPUT: task_events.csv (Raw Task-Level Data)
│
├─▶ [Step 1] Data Loading
│   • Load CSV file
│   • Auto-detect format (event-based vs direct)
│   • Map columns (collection_id → job_id, etc.)
│
├─▶ [Step 2] Preprocessing
│   • Clean invalid rows
│   • Handle missing values
│   • Extract task runtimes (end_time - start_time)
│   • Validate data types
│
├─▶ [Step 3] Feature Engineering
│   • Aggregate task-level → job-level
│   • Compute features:
│     - num_tasks (count)
│     - avg_task_runtime (mean)
│     - max_task_runtime (max)
│     - std_task_runtime (std)
│     - scheduling_class (first/mode)
│     - priority (mean)
│
├─▶ [Step 4] Skew Labeling
│   • Apply rule: max >= 2 * avg → skewed (1)
│   • Create binary labels
│   • Generate statistics
│
├─▶ [Step 5] Model Training
│   • Split data (80% train, 20% test)
│   • Scale features (StandardScaler for LR)
│   • Train Logistic Regression
│   • Train Random Forest
│   • Save models + scalers
│
├─▶ [Step 6] Model Evaluation
│   • Predict on test set
│   • Calculate metrics (Accuracy, Precision, Recall, F1)
│   • Generate confusion matrices
│   • Plot feature importance
│
├─▶ [Step 7] Baseline Comparison
│   • Train rule-based baseline
│   • Compare with ML models
│   • Generate comparison plots
│
└─▶ OUTPUT: 
    • job_level_data.csv (Processed data)
    • trained_models.pkl (Saved models)
    • Visualizations (PNG files)
    • Evaluation metrics (Console output)
```

---

## 5. ML Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MACHINE LEARNING PIPELINE                        │
└─────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │  Job-Level Dataset  │
                    │  (Features + Labels)│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Train/Test Split  │
                    │  (80/20 stratified) │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
        ┌───────────────┐           ┌───────────────┐
        │  Training Set │           │   Test Set    │
        │   (80%)       │           │    (20%)      │
        └───────┬───────┘           └───────┬───────┘
                │                           │
                ▼                           │
        ┌───────────────┐                   │
        │ Feature       │                   │
        │ Scaling       │                   │
        │ (StandardScaler)                  │
        └───────┬───────┘                   │
                │                           │
    ┌───────────┴───────────┐               │
    │                       │               │
    ▼                       ▼               │
┌──────────┐         ┌──────────┐           │
│ Logistic │         │  Random  │           │
│Regression│         │  Forest  │           │
└────┬─────┘         └─────┬────┘           │
     │                     │                │
     └──────────┬──────────┘                │
                │                           │
                ▼                           │
        ┌───────────────┐                   │
        │ Model Saving  │                   │
        │ (trained_models.pkl)              │
        └───────┬───────┘                   │
                │                           │
                └───────────┬───────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   Evaluation  │
                    │   (Test Set)  │
                    └───────┬───────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
        ┌───────────────┐     ┌────────────────┐
        │   Metrics     │     │  Visualizations│
        │ (Acc, Prec,   │     │  (Confusion    │
        │  Rec, F1)     │     │   Matrix, etc) │
        └───────────────┘     └────────────────┘
```

---

## 6. Technology Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TECHNOLOGY STACK                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  PROGRAMMING LANGUAGE                                               │
│  • Python 3.x                                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  DATA PROCESSING                                                    │
│  • pandas      - Data manipulation and analysis                     │
│  • numpy       - Numerical computations                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  MACHINE LEARNING                                                   │
│  • scikit-learn - ML models (Logistic Regression, Random Forest)    │
│                 - Model evaluation metrics                          │
│                 - Data preprocessing (StandardScaler)               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  VISUALIZATION                                                      │
│  • matplotlib  - Static plots (confusion matrix, feature importance)│
│  • seaborn     - Statistical visualizations                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  INTERACTIVE DASHBOARD (Optional)                                   │
│  • streamlit   - Web-based dashboard for live simulation            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  DEVELOPMENT TOOLS                                                  │
│  • Jupyter     - Interactive data exploration                       │
│  • Git         - Version control                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT SCENARIOS                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SCENARIO 1: RESEARCH/DEVELOPMENT (Current)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │  Local       │                                                   │
│  │  Machine     │                                                   │
│  │  (Windows)   │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ├─▶ Python Environment (.venv)                              │
│         ├─▶ Data Storage (data/raw/, data/processed/)               │
│         ├─▶ Model Storage (models/)                                 │
│         └─▶ Execution: python main.py                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  SCENARIO 2: PRODUCTION DEPLOYMENT (Future)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │   Job        │───▶│  Prediction  │───▶│  Scheduler   │          │
│  │  Metadata    │    │    Service   │    │  (YARN/K8s)  │           │
│  │  (Source)    │    │  (API/Model) │    │              │           │
│  └──────────────┘    └──────────────┘    └──────────────┘           │
│         │                   │                                       │
│         │                   │                                       │
│         ▼                   ▼                                       │
│  ┌──────────────┐    ┌──────────────┐                               │
│  │  Feature     │    │  Monitoring  │                               │
│  │  Store       │    │  Dashboard   │                               │
│  └──────────────┘    └──────────────┘                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. Key Design Decisions

### 8.1 Modular Architecture
- **Decision**: Separate modules for each pipeline stage
- **Rationale**: 
  - Easy to test individual components
  - Reusable code
  - Clear separation of concerns

### 8.2 Pre-Execution Features Only
- **Decision**: Use only features computable before job execution
- **Rationale**: 
  - Enables early prediction
  - Practical for real-world deployment
  - No need to wait for job completion

### 8.3 Lightweight ML Models
- **Decision**: Logistic Regression + Random Forest (no deep learning)
- **Rationale**:
  - Fast training and inference
  - Interpretable results
  - Low computational requirements
  - Suitable for research reproducibility

### 8.4 Auto-Detection of Data Format
- **Decision**: Automatically detect CSV format (event-based vs direct)
- **Rationale**:
  - Handles different dataset versions
  - Robust to format variations
  - User-friendly (no manual configuration)

### 8.5 Model Persistence
- **Decision**: Save models as pickle files
- **Rationale**:
  - Easy to load and use
  - Preserves scalers and models together
  - Standard Python practice

---

## 9. Scalability Considerations

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SCALABILITY ASPECTS                              │
└─────────────────────────────────────────────────────────────────────┘

CURRENT LIMITATIONS:
• Single-machine execution
• In-memory data processing (limited by RAM)
• Sequential pipeline execution

POTENTIAL IMPROVEMENTS:
┌─────────────────────────────────────────────────────────────────────┐
│  For Larger Datasets:                                               │
│  • Use Dask or Spark for distributed processing                     │
│  • Implement chunked data loading                                   │
│  • Parallel feature engineering                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  For Production:                                                    │
│  • Deploy as REST API (Flask/FastAPI)                               │
│  • Use model serving (MLflow, TensorFlow Serving)                   │
│  • Implement caching for predictions                                │
│  • Add monitoring and logging                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  For Real-Time:                                                     │
│  • Stream processing (Kafka + Spark Streaming)                      │
│  • Incremental model updates                                        │
│  • Feature store integration                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. Security & Best Practices

```
┌─────────────────────────────────────────────────────────────────────┐
│                  SECURITY & BEST PRACTICES                          │
└─────────────────────────────────────────────────────────────────────┘

✅ IMPLEMENTED:
• Input validation in preprocessing
• Error handling in all modules
• Type checking and conversion
• Safe file operations (path handling)

⚠️ FOR PRODUCTION:
• API authentication/authorization
• Input sanitization
• Rate limiting
• Model versioning
• Secure model storage
• Audit logging
```

---

## 11. Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                   COMPONENT INTERACTIONS                            │
└─────────────────────────────────────────────────────────────────────┘

main.py (Orchestrator)
│
├─▶ data_loader.py
│   └─▶ Returns: Raw DataFrame
│
├─▶ preprocessing.py
│   ├─▶ clean_task_events()
│   ├─▶ extract_task_runtimes()
│   └─▶ prepare_for_aggregation()
│   └─▶ Returns: Cleaned DataFrame with runtimes
│
├─▶ feature_engineering.py
│   ├─▶ aggregate_to_job_level()
│   ├─▶ encode_categorical_features()
│   └─▶ prepare_features_for_training()
│   └─▶ Returns: Job-level features DataFrame
│
├─▶ skew_labeling.py
│   ├─▶ label_skewed_jobs()
│   └─▶ Returns: Labeled DataFrame
│
├─▶ train_model.py
│   ├─▶ split_data()
│   ├─▶ train_logistic_regression()
│   ├─▶ train_random_forest()
│   └─▶ save_models()
│   └─▶ Returns: Trained models + scalers
│
├─▶ evaluate_model.py
│   ├─▶ evaluate_model()
│   ├─▶ plot_confusion_matrix()
│   ├─▶ plot_feature_importance()
│   └─▶ Returns: Evaluation metrics + plots
│
└─▶ baseline.py
    ├─▶ evaluate_baseline()
    └─▶ compare_with_ml_models()
    └─▶ Returns: Baseline metrics + comparison
```

---

## 12. Data Schema

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SCHEMAS                                │
└─────────────────────────────────────────────────────────────────────┘

RAW DATA (task_events.csv):
┌─────────────────────┬──────────────┬────────────────────────────┐
│ Column              │ Type         │ Description                │
├─────────────────────┼──────────────┼────────────────────────────┤
│ collection_id       │ int/str      │ Job identifier             │
│ instance_index      │ int/str      │ Task identifier            │
│ start_time          │ int          │ Start time (microseconds)  │
│ end_time            │ int          │ End time (microseconds)    │
│ scheduling_class    │ int          │ Scheduling class (0-3)     │
│ priority            │ int          │ Priority (0-450)           │
│ ...                 │ ...         │ Other metadata              │
└─────────────────────┴──────────────┴────────────────────────────┘

PROCESSED DATA (job_level_data.csv):
┌─────────────────────┬──────────────┬────────────────────────────┐
│ Column              │ Type         │ Description                │
├─────────────────────┼──────────────┼────────────────────────────┤
│ job_id              │ int          │ Job identifier             │
│ num_tasks           │ int          │ Number of tasks            │
│ avg_task_runtime    │ float        │ Average runtime            │
│ max_task_runtime    │ float        │ Maximum runtime            │
│ std_task_runtime    │ float        │ Standard deviation         │
│ scheduling_class    │ int          │ Scheduling class           │
│ priority            │ float        │ Average priority           │
│ is_skewed           │ int (0/1)    │ Skew label                 │
└─────────────────────┴──────────────┴────────────────────────────┘
```

---

## 13. Model Architecture Details

```
┌─────────────────────────────────────────────────────────────────────┐
│                      MODEL ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────┘

LOGISTIC REGRESSION:
┌─────────────────────────────────────────────────────────────────────┐
│ Input Features (6) → StandardScaler → Logistic Regression → Output  │
│                                                                     │
│ Hyperparameters:                                                    │
│ • Solver: LBFGS                                                     │
│ • Max iterations: 1000                                              │
│ • Class weight: None (can use 'balanced' for imbalanced data)       │
└─────────────────────────────────────────────────────────────────────┘

RANDOM FOREST:
┌─────────────────────────────────────────────────────────────────────┐
│ Input Features (6) → Random Forest Classifier → Output              │
│                                                                     │
│ Hyperparameters:                                                    │
│ • N estimators: 100                                                 │
│ • Max depth: 10                                                     │
│ • Random state: 42 (for reproducibility)                            │
└─────────────────────────────────────────────────────────────────────┘

BASELINE (Rule-based):
┌─────────────────────────────────────────────────────────────────────┐
│ max_task_runtime > threshold → skewed (1)                           │
│                                                                     │
│ • Threshold optimized on training set                               │
│ • Maximizes F1-score                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 14. File Structure with Dependencies

```
project_root/
│
├── data/                    # Data storage
│   ├── raw/                # Input data (ignored by git)
│   └── processed/          # Output data (ignored by git)
│
├── src/                     # Core modules (imported by main.py)
│   ├── data_loader.py      # ← main.py imports
│   ├── preprocessing.py    # ← main.py imports
│   ├── feature_engineering.py  # ← main.py imports
│   ├── skew_labeling.py    # ← main.py imports
│   ├── train_model.py      # ← main.py imports
│   ├── evaluate_model.py   # ← main.py imports
│   └── baseline.py         # ← main.py imports
│
├── models/                  # Generated outputs
│   ├── trained_models.pkl  # ← train_model.py saves
│   └── *.png               # ← evaluate_model.py saves
│
├── notebooks/               # Exploration
│   └── exploration.ipynb
│
├── main.py                  # Entry point (orchestrates all modules)
├── main_sample.py          # Alternative entry (sample data)
│
├── predict_job.py          # Uses trained_models.pkl
├── dashboard_app.py        # Uses predict_job.py + job_simulator.py
├── job_simulator.py        # Used by dashboard_app.py
│
└── requirements.txt        # Dependencies
```

---

## 15. Execution Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                      EXECUTION FLOW                                 │
└─────────────────────────────────────────────────────────────────────┘

User runs: python main.py
│
├─▶ [1] Load Dataset
│   └─▶ data_loader.load_task_events()
│       └─▶ Returns: DataFrame (task-level)
│
├─▶ [2] Preprocess
│   ├─▶ preprocessing.clean_task_events()
│   ├─▶ preprocessing.extract_task_runtimes()
│   └─▶ preprocessing.prepare_for_aggregation()
│       └─▶ Returns: DataFrame (cleaned, with runtimes)
│
├─▶ [3] Feature Engineering
│   ├─▶ feature_engineering.aggregate_to_job_level()
│   └─▶ feature_engineering.encode_categorical_features()
│       └─▶ Returns: DataFrame (job-level features)
│
├─▶ [4] Labeling
│   └─▶ skew_labeling.label_skewed_jobs()
│       └─▶ Returns: DataFrame (features + labels)
│       └─▶ Saves: job_level_data.csv
│
├─▶ [5] Training Preparation
│   ├─▶ feature_engineering.prepare_features_for_training()
│   └─▶ train_model.split_data()
│       └─▶ Returns: X_train, X_test, y_train, y_test
│
├─▶ [6] Model Training
│   ├─▶ train_model.train_logistic_regression()
│   ├─▶ train_model.train_random_forest()
│   └─▶ train_model.save_models()
│       └─▶ Saves: trained_models.pkl
│
├─▶ [7] Evaluation
│   ├─▶ evaluate_model.evaluate_all_models()
│   ├─▶ evaluate_model.plot_confusion_matrix()
│   └─▶ evaluate_model.plot_feature_importance()
│       └─▶ Saves: PNG files
│
└─▶ [8] Baseline Comparison
    ├─▶ baseline.evaluate_baseline()
    └─▶ baseline.compare_with_ml_models()
        └─▶ Prints: Comparison table
```

---

## Summary

This architecture provides:

✅ **Modular Design**: Each component has a single responsibility  
✅ **Scalable Pipeline**: Easy to extend with new models or features  
✅ **Reproducible**: Fixed random seeds, clear data flow  
✅ **Production-Ready**: Model persistence, prediction interface  
✅ **Research-Friendly**: Well-documented, easy to understand  
✅ **Flexible**: Handles different data formats, optional components  

The architecture follows ML best practices and is designed for both research reproducibility and potential production deployment.
