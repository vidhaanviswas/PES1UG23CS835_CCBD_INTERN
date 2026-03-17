# High-Level Architecture

## Early Prediction of Data Skew in Cloud-Based Big Data Jobs Using Lightweight Machine Learning Models

---

## 1. System Overview

```mermaid
flowchart LR
    A[Raw Data<br/>CSV Files] --> B[Processing<br/>Pipeline]
    B --> C[ML Models<br/>Trained]
    C --> D[Predictions<br/>& Outputs]
```

---

## 2. High-Level Architecture Diagram

```mermaid
flowchart TD
    A[data/raw/task_events.csv<br/>Google Cluster Traces 2019] --> B[src/data_loader.py<br/>Load & auto-detect format]
    B --> C[src/preprocessing.py<br/>Clean, extract runtimes]
    C --> D[src/feature_engineering.py<br/>Aggregate to job-level]
    C --> E[src/skew_labeling.py<br/>Label: max >= 2 x avg]
    D --> F[Job-Level Features]
    E --> G[Skew Labels 0/1]
    F --> H[data/processed/job_level_data.csv]
    G --> H
    H --> I[main.py / main_sample.py]
    I --> J[src/splitters.py<br/>time / template / rolling]
    J --> K[src/train_model.py<br/>fit split + calibration holdout<br/>SMOTE on fit subset only]
    K --> L[models/trained_models.pkl<br/>LR, RF, XGBoost, LightGBM]
    L --> M[src/evaluate_model.py<br/>Accuracy, F1, PR-AUC<br/>ROC-AUC, Brier, ECE]
    M --> N[outputs/ metrics + PNG plots]
    L --> O[predict_job.py<br/>per-model tuned thresholds]
    O --> P[mitigation_simulation.py]
    P --> Q[outputs/mitigation_impact_*.csv]
```

---

## 3. Component Architecture

```mermaid
flowchart TD
    subgraph src ["Core Pipeline Modules (src/)"]
        A[data_loader.py] --> B[preprocessing.py]
        B --> C[feature_engineering.py]
        B --> D[skew_labeling.py]
        C --> E[train_model.py]
        D --> E
        E --> F[evaluate_model.py]
        F --> G[baseline.py]
    end
    subgraph scripts ["Execution Scripts"]
        H[main.py]
        I[main_sample.py]
    end
    subgraph utils ["Utility Scripts"]
        J[predict_job.py]
        K[validate_model.py]
        L[mitigation_simulation.py]
    end
    src --> utils
```

---

## 4. Data Flow Architecture

```mermaid
flowchart TD
    IN[task_events.csv] --> S1["Step 1: Data Loading<br/>Load CSV, auto-detect format<br/>Map columns"]
    S1 --> S2["Step 2: Preprocessing<br/>Clean invalid rows, handle missing values<br/>Extract task runtimes"]
    S2 --> S3["Step 3: Feature Engineering<br/>Aggregate to job-level<br/>num_tasks, scheduling_class, priority<br/>cpu/memory/disk request stats"]
    S3 --> S4["Step 4: Skew Labeling<br/>max >= 2 x avg → skewed<br/>Create binary labels"]
    S4 --> S5["Step 5: Model Training<br/>Time/template/rolling splits<br/>Calibration holdout, SMOTE on fit only<br/>LR, RF, XGBoost, LightGBM"]
    S5 --> S6["Step 6: Model Evaluation<br/>Accuracy, Precision, Recall, F1<br/>Confusion matrices, feature importance"]
    S6 --> S7["Step 7: Baseline Comparison<br/>Rule-based baseline<br/>Compare with ML models"]
    S7 --> OUT["Outputs<br/>job_level_data.csv<br/>trained_models.pkl<br/>PNG visualizations"]
```

---

## 5. ML Pipeline Architecture

```mermaid
flowchart TD
    A[Job-Level Dataset<br/>Features + Labels] --> B[Train/Test Split<br/>80/20 time-based]
    B --> C[Training Set 80%]
    B --> D[Test Set 20%]
    C --> E[Calibration Holdout Split<br/>80% fit / 20% calib]
    E --> F[SMOTE on fit subset]
    F --> G[Logistic Regression]
    F --> H[Random Forest]
    F --> I[XGBoost]
    F --> J[LightGBM]
    G --> K[Calibration on holdout<br/>CalibratedClassifierCV]
    H --> K
    I --> K
    J --> K
    K --> L[models/trained_models.pkl]
    L --> M[Evaluation on Test Set]
    D --> M
    M --> N[Metrics<br/>Acc, Prec, Rec, F1<br/>PR-AUC, ROC-AUC, Brier, ECE]
    M --> O[Visualizations<br/>Confusion Matrix<br/>Feature Importance]
```

---

## 6. Technology Stack

```mermaid
flowchart LR
    subgraph lang ["Language"]
        Python["Python 3.x"]
    end
    subgraph data ["Data Processing"]
        pandas["pandas<br/>Data manipulation"]
        numpy["numpy<br/>Numerical computations"]
    end
    subgraph ml ["Machine Learning"]
        sklearn["scikit-learn<br/>LR, RF, metrics, StandardScaler"]
        xgboost["xgboost"]
        lightgbm["lightgbm"]
        imblearn["imbalanced-learn<br/>SMOTE"]
    end
    subgraph viz ["Visualization"]
        matplotlib["matplotlib<br/>Plots, confusion matrix"]
        seaborn["seaborn<br/>Statistical plots"]
        shap["shap<br/>Explainability"]
    end
    subgraph tools ["Development Tools"]
        jupyter["Jupyter<br/>Exploration"]
        streamlit["streamlit<br/>Dashboards"]
    end
    lang --> data --> ml --> viz
```

---

## 7. Deployment Architecture

```mermaid
flowchart TD
    subgraph dev ["Scenario 1: Research/Development (Current)"]
        A[Local Machine] --> B[Python venv]
        A --> C[data/raw/ + data/processed/]
        A --> D[models/]
        A --> E[python main.py / main_sample.py]
    end
    subgraph prod ["Scenario 2: Production Deployment (Future)"]
        F[Job Metadata Source] --> G[Prediction Service<br/>REST API + Model]
        G --> H[Scheduler<br/>YARN / K8s]
        F --> I[Feature Store]
        G --> J[Monitoring Dashboard]
    end
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

```mermaid
flowchart LR
    CUR["Current Limitations<br/>Single-machine<br/>In-memory processing<br/>Sequential pipeline"]
    CUR --> DS["Larger Datasets<br/>Dask or Spark<br/>Chunked data loading<br/>Parallel feature engineering"]
    CUR --> PR["Production Scale<br/>REST API via FastAPI<br/>Model serving with MLflow<br/>Prediction caching + logging"]
    CUR --> RT["Real-Time<br/>Kafka + Spark Streaming<br/>Incremental model updates<br/>Feature store integration"]
```

---

## 10. Security & Best Practices

**Implemented:**
- Input validation in preprocessing
- Error handling in all modules
- Type checking and conversion
- Safe file operations (path handling)

**For Production:**
- API authentication/authorization
- Input sanitization
- Rate limiting
- Model versioning
- Secure model storage
- Audit logging

---

## 11. Component Interaction Diagram

```mermaid
flowchart TD
    M["main.py (Orchestrator)"]
    M --> DL["data_loader.py<br/>load_task_events()<br/>→ Raw DataFrame"]
    M --> PP["preprocessing.py<br/>clean_task_events()<br/>extract_task_runtimes()<br/>→ Cleaned DataFrame"]
    M --> FE["feature_engineering.py<br/>aggregate_to_job_level()<br/>encode_categorical_features()<br/>→ Job-level features"]
    M --> SL["skew_labeling.py<br/>label_skewed_jobs()<br/>→ Labeled DataFrame"]
    M --> TM["train_model.py<br/>split_data(), train_all_models()<br/>save_models()<br/>→ trained_models.pkl"]
    M --> EM["evaluate_model.py<br/>evaluate_all_models()<br/>plot_confusion_matrix()<br/>→ Metrics + PNG plots"]
    M --> BL["baseline.py<br/>evaluate_baseline()<br/>compare_with_ml_models()<br/>→ Baseline comparison"]
```

---

## 12. Data Schema

**RAW DATA (task_events.csv):**

| Column | Type | Description |
|---|---|---|
| collection_id | int/str | Job identifier |
| instance_index | int/str | Task identifier |
| start_time | int | Start time (microseconds) |
| end_time | int | End time (microseconds) |
| scheduling_class | int | Scheduling class (0-3) |
| priority | int | Priority (0-450) |

**PROCESSED DATA (job_level_data.csv):**

| Column | Type | Description |
|---|---|---|
| job_id | int | Job identifier |
| num_tasks | int | Number of tasks |
| avg_task_runtime | float | Average task runtime |
| max_task_runtime | float | Maximum task runtime |
| std_task_runtime | float | Runtime standard deviation |
| scheduling_class | int | Scheduling class |
| priority | float | Average priority |
| is_skewed | int (0/1) | Skew label |

---

## 13. Model Architecture Details

```mermaid
flowchart LR
    subgraph lr ["Logistic Regression"]
        A1["Input Features x10"] --> A2[StandardScaler] --> A3["LogisticRegression<br/>solver=lbfgs, max_iter=1000"]
    end
    subgraph rf ["Random Forest"]
        B1["Input Features x10"] --> B2["RandomForestClassifier<br/>n_estimators=100, max_depth=10"]
    end
    subgraph xgb ["XGBoost"]
        C1["Input Features x10"] --> C2[XGBClassifier]
    end
    subgraph lgbm ["LightGBM"]
        D1["Input Features x10"] --> D2[LGBMClassifier]
    end
    subgraph bl ["Baseline (Rule-based)"]
        E1["num_tasks > threshold"] --> E2["skewed = 1<br/>threshold maximises F1"]
    end
```

---

## 14. File Structure with Dependencies

```
project_root/
│
├── data/
│   ├── raw/                        # Input data (ignored by git)
│   └── processed/                  # Engineered job-level data
│
├── src/                            # Core pipeline modules
│   ├── data_loader.py              # ← imported by main.py
│   ├── preprocessing.py            # ← imported by main.py
│   ├── feature_engineering.py      # ← imported by main.py
│   ├── skew_labeling.py            # ← imported by main.py
│   ├── splitters.py                # ← imported by main.py
│   ├── train_model.py              # ← imported by main.py
│   ├── evaluate_model.py           # ← imported by main.py
│   └── baseline.py                 # ← imported by main.py
│
├── models/
│   └── trained_models.pkl          # ← saved by train_model.py
│
├── outputs/                        # Metrics, plots, CSVs
│
├── notebooks/
│   └── exploration.ipynb
│
├── main.py                         # Full pipeline entry point
├── main_sample.py                  # Sampled pipeline + rolling eval
├── predict_job.py                  # Single & batch inference
├── mitigation_simulation.py        # Policy impact simulation
├── validate_model.py               # Model validation utilities
└── requirements.txt
```

---

## 15. Execution Flow

```mermaid
flowchart TD
    U["python main.py"] --> S1["1. Load Dataset<br/>data_loader.load_task_events()<br/>→ task-level DataFrame"]
    S1 --> S2["2. Preprocess<br/>clean_task_events()<br/>extract_task_runtimes()<br/>→ cleaned DataFrame"]
    S2 --> S3["3. Feature Engineering<br/>aggregate_to_job_level()<br/>encode_categorical_features()<br/>→ job-level features"]
    S3 --> S4["4. Labeling<br/>label_skewed_jobs()<br/>→ Saves job_level_data.csv"]
    S4 --> S5["5. Training Preparation<br/>prepare_features_for_training()<br/>split_data()"]
    S5 --> S6["6. Model Training<br/>train_all_models()<br/>→ Saves trained_models.pkl"]
    S6 --> S7["7. Evaluation<br/>evaluate_all_models()<br/>→ Saves PNG plots"]
    S7 --> S8["8. Baseline Comparison<br/>evaluate_baseline()<br/>compare_with_ml_models()<br/>→ Prints comparison table"]
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
