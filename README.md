# Early Prediction of Data Skew in Cloud-Based Big Data Jobs Using Lightweight Machine Learning Models

## Project Overview

This research project implements a machine learning pipeline to predict data skew in cloud-based big data jobs **before execution** using lightweight ML models. The project uses the Google Cluster Workload Traces (2019 sample) dataset to train and evaluate models that can identify potentially skewed jobs based on pre-execution features.

> 📐 **Architecture Documentation**: See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system architecture, component interactions, and design decisions.

## Dataset Description

The project uses the **Google Cluster Workload Traces (2019 sample)** dataset, specifically the `task_events.csv` file. This dataset contains information about tasks submitted to Google's cluster, including:

- **Job and Task IDs**: Identifiers for jobs and their constituent tasks
- **Timestamps**: Event timestamps for task lifecycle events
- **Event Types**: Submit, schedule, evict, fail, finish, kill, lost, update events
- **Resource Requests**: CPU, memory, and disk space requests
- **Scheduling Information**: Scheduling class and priority

### Dataset Location

Place the `task_events.csv` file in the `data/raw/` directory before running the pipeline.

## Data Skew Definition

A job is labeled as **skewed** if:

```
max_task_runtime >= 2 * average_task_runtime
```

- **Skewed jobs (label = 1)**: Jobs where the maximum task runtime is at least twice the average task runtime
- **Non-skewed jobs (label = 0)**: All other jobs

This definition captures jobs with significant runtime imbalance among tasks, which can lead to performance degradation and resource underutilization.

## Features (Pre-Execution Only)

The model uses only **pre-execution features** that can be computed before job execution:

1. **num_tasks**: Number of tasks in the job
2. **avg_task_runtime**: Average runtime of tasks in the job
3. **max_task_runtime**: Maximum runtime of tasks in the job
4. **std_task_runtime**: Standard deviation of task runtimes
5. **scheduling_class**: Scheduling class of the job (encoded)
6. **priority**: Average priority of tasks in the job

These features are aggregated from task-level data to job-level features.

## Machine Learning Models

The project implements and compares two lightweight ML models:

1. **Logistic Regression**: A linear classifier with feature scaling
2. **Random Forest Classifier**: An ensemble method with 100 trees and max depth of 10

Both models are trained on the same features and evaluated using standard classification metrics.

## Evaluation Metrics

The following metrics are used to evaluate model performance:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Proportion of predicted skewed jobs that are actually skewed
- **Recall**: Proportion of actual skewed jobs that are correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification results

## Baseline Comparison

A simple rule-based baseline is implemented for comparison:

- **Baseline Rule**: If `max_task_runtime > threshold` → skewed, else non-skewed
- The threshold is optimized to maximize F1-score on the training data

The ML models are compared against this baseline to demonstrate their effectiveness.

## Project Structure

```
project_root/
│
├── data/
│   ├── raw/
│   │   └── task_events.csv          # Input dataset (place your file here)
│   └── processed/
│       └── job_level_data.csv       # Generated: processed job-level data
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py               # Dataset loading utilities
│   ├── preprocessing.py             # Data cleaning and validation
│   ├── feature_engineering.py       # Job-level feature extraction
│   ├── skew_labeling.py             # Skew labeling logic
│   ├── train_model.py               # ML model training
│   ├── evaluate_model.py            # Model evaluation and visualization
│   └── baseline.py                  # Baseline model implementation
│
├── models/
│   ├── trained_models.pkl           # Generated: saved ML models
│   ├── confusion_matrix_*.png       # Generated: confusion matrix plots
│   └── feature_importance_*.png     # Generated: feature importance plots
│
├── notebooks/
│   └── exploration.ipynb            # Data exploration notebook
│
├── main.py                          # Main pipeline script (full dataset)
├── main_sample.py                   # Sample data execution script
├── requirements.txt                 # Python dependencies
└── README.md                        # This file

Optional utility scripts (not required for core pipeline):
├── dashboard_app.py                 # Streamlit dashboard for live simulation
├── job_simulator.py                 # Job simulation for dashboard
├── predict_job.py                  # Prediction utilities for production use
├── validate_model.py               # Model validation script
├── analyze_results.py              # Results analysis script
├── create_advanced_plots.py        # Advanced visualization generation
└── demo_production_use.py          # Production use demonstration
```

## Installation

1. **Clone or download this repository**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Google Cluster Workload Traces dataset**:
   - Download the 2019 **sample dataset** (recommended for testing)
   - Or download the full dataset if you have sufficient storage and memory
   - Extract `task_events.csv` and place it in `data/raw/` directory
   
   **Note**: For most users, the sample dataset is sufficient. The project automatically handles both sample and full datasets.

## How to Run

### Quick Start with Sample Data (Recommended)

**If you're using sample data** (instead of full 2TB dataset), use:
```powershell
python main_sample.py
```

This script is pre-configured for sample data and runs much faster. It loads a subset of the dataset (default: 500,000 rows) for quick testing.

### Run the Complete Pipeline

Execute the main pipeline script:

**Windows (PowerShell/CMD):**
```powershell
python main.py
```

**Linux/Mac:**
```bash
python main.py
```

This will execute the complete 8-step pipeline:
1. Load dataset from `data/raw/task_events.csv`
2. Preprocess and clean data
3. Engineer job-level features
4. Label skewed jobs
5. Prepare features for training
6. Train ML models (Logistic Regression and Random Forest)
7. Evaluate all models
8. Compare with baseline and generate outputs

### Run Individual Modules

You can also run individual modules for testing:

```bash
# Test data loading
python src/data_loader.py

# Test preprocessing
python src/preprocessing.py

# Test feature engineering
python src/feature_engineering.py

# Test skew labeling
python src/skew_labeling.py

# Test model training
python src/train_model.py

# Test model evaluation
python src/evaluate_model.py

# Test baseline
python src/baseline.py
```

### Use Jupyter Notebook

Open `notebooks/exploration.ipynb` for interactive data exploration and analysis.

## Expected Outputs

After running the pipeline, you should see:

1. **Processed Data**:
   - `data/processed/job_level_data.csv`: Clean job-level dataset with features and labels

2. **Trained Models**:
   - `models/trained_models.pkl`: Saved ML models and scalers

3. **Evaluation Results** (printed to console):
   - Accuracy, Precision, Recall, F1-score for each model
   - Comparison table showing baseline vs ML models

4. **Visualizations** (generated in `models/` directory):
   - `confusion_matrix_logistic_regression.png`
   - `confusion_matrix_random_forest.png`
   - `confusion_matrix_baseline.png`
   - `feature_importance_random_forest.png`
   - `roc_curves.png` (if using advanced plots)
   - `precision_recall_curves.png` (if using advanced plots)
   - `feature_comparison.png` (if using advanced plots)
   - `metrics_comparison.png` (if using advanced plots)

## Implementation Details

### Data Processing Pipeline

1. **Data Loading**: Reads `task_events.csv` with proper column names
2. **Cleaning**: Removes invalid rows, handles missing values
3. **Runtime Extraction**: Computes task runtimes from submit and finish events
4. **Aggregation**: Aggregates task-level data to job-level features
5. **Labeling**: Applies skew definition to create binary labels

### Model Training

- **Train-Test Split**: 80-20 split with stratification
- **Feature Scaling**: StandardScaler for Logistic Regression
- **Hyperparameters**:
  - Logistic Regression: LBFGS solver, max_iter=1000
  - Random Forest: 100 estimators, max_depth=10

### Model Evaluation

- All models evaluated on the same test set
- Metrics computed using scikit-learn
- Visualizations generated using matplotlib and seaborn

## Key Constraints

- ✅ Uses only pre-execution features
- ✅ Lightweight ML models (no deep learning)
- ✅ Runs on a single machine
- ✅ Reproducible with fixed random seeds
- ✅ No modifications to Spark/Hadoop internals

## Troubleshooting

### Dataset Not Found

If you see `FileNotFoundError`, ensure:
- `task_events.csv` is placed in `data/raw/` directory
- The file name is exactly `task_events.csv`

### Memory Issues

If the dataset is too large:
- Use `main_sample.py` instead of `main.py` - it's pre-configured to load a sample (500,000 rows by default)
- Or modify `main.py` to use `get_sample_data()` function from `data_loader.py` to load a subset

### Import Errors

If you see import errors:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Run from the project root directory

## Research Applications

This project can be used for:

- **Resource Allocation**: Predict skewed jobs to allocate resources more efficiently
- **Scheduling Optimization**: Prioritize or reschedule potentially skewed jobs
- **Performance Monitoring**: Early detection of jobs that may cause performance issues
- **Cost Optimization**: Identify jobs that may lead to resource waste

## Additional Utility Scripts

The project includes several utility scripts that enhance functionality but are not required for the core pipeline:

### Analysis and Visualization

- **`analyze_results.py`**: Comprehensive analysis of processed data, feature statistics, and model insights
- **`create_advanced_plots.py`**: Generates advanced visualizations including ROC curves, Precision-Recall curves, and feature comparison plots

### Model Validation and Production

- **`validate_model.py`**: Validates model loading, robustness (cross-validation), and prediction interface
- **`predict_job.py`**: Provides functions for making single or batch predictions on new job features
- **`demo_production_use.py`**: Demonstrates production-ready usage scenarios

### Dashboard

- **`dashboard_app.py`**: Interactive Streamlit dashboard for live job simulation and visualization
- **`job_simulator.py`**: Generates synthetic job data for dashboard demonstration

## Production Use

### Validating Model Readiness

To verify your model works correctly and is ready for production:

```powershell
# Run comprehensive validation
python validate_model.py

# Test prediction interface
python predict_job.py

# See complete demonstration
python demo_production_use.py
```

### Using the Model for Predictions

**Single Job Prediction:**
```python
from predict_job import predict_job_skew

job_features = {
    'num_tasks': 100,
    'avg_task_runtime': 200000000,  # From historical data
    'max_task_runtime': 450000000,
    'std_task_runtime': 50000000,
    'scheduling_class': 2,
    'priority': 150
}

result = predict_job_skew(job_features)
if result['is_skewed']:
    print(f"⚠️ Job at risk! Probability: {result['skew_probability']:.2%}")
```

**Batch Prediction:**
```python
from predict_job import predict_from_dataframe
import pandas as pd

jobs_df = pd.read_csv('new_jobs.csv')
predictions = predict_from_dataframe(jobs_df)
high_risk = predictions[predictions['predicted_skew'] == 1]
```

For production integration, refer to the `predict_job.py` module which provides functions for single and batch predictions.

## Live Simulation Dashboard (Optional)

You can run a **local live job simulation dashboard** that generates synthetic jobs, predicts skew using your trained model, and visualizes results in real-time. This is an optional feature for demonstrations.

### Prerequisites

Ensure you have run the main pipeline first to generate trained models:
```powershell
python main.py
# or
python main_sample.py
```

### Run the dashboard

```powershell
streamlit run dashboard_app.py
```

The dashboard will open in your browser (typically at `http://localhost:8501`).

### What it does
- Simulates an incoming stream of jobs (no live cluster needed)
- Uses `models/trained_models.pkl` to predict skew probability
- Shows a live table + charts (flagged jobs, probability distribution)
- Provides interactive controls for model selection and threshold adjustment
- Includes plain-English explanations for non-technical audiences

## Future Enhancements

Potential improvements:

- Additional feature engineering (e.g., task size distribution)
- More ML models (e.g., XGBoost, SVM)
- Cross-validation for more robust evaluation
- Feature selection to identify most important features
- Real-time prediction API

## Citation

If you use this code in your research, please cite:

```
Early Prediction of Data Skew in Cloud-Based Big Data Jobs 
Using Lightweight Machine Learning Models
Dataset: Google Cluster Workload Traces (2019)
```

## License

This project is for research purposes.

## Contact

For questions or issues, please refer to the project repository.

---

**Note**: This project is designed for research and educational purposes. Ensure you have proper access rights to the Google Cluster Workload Traces dataset before use.
