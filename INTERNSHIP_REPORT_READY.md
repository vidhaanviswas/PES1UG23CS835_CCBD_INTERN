# Internship at <Company Full Name>

# Early Prediction of Data Skew in Cloud-Based Big Data Jobs Using Lightweight Machine Learning Models

**Submitted by:** PES1UG23CS835  
**Under the guidance of:** <External Guide Name, Designation>  
**Internship duration:** <Internship Duration in Weeks>  
**Department of Computer Science and Engineering**  
**Faculty of Engineering, PES University**  
**Bengaluru, Karnataka, India**

---

## Company Certificate

Attach the company completion certificate here. If the company completion certificate is not available, attach the first page of the acceptance or joining letter as permitted in the guidelines.

---

## Declaration

I hereby declare that the project entitled **Early Prediction of Data Skew in Cloud-Based Big Data Jobs Using Lightweight Machine Learning Models** has been carried out at **<Company Full Name>** by me under the guidance of **<External Guide Name, Designation>** and submitted in partial fulfillment of the credits for the degree of Bachelor of Technology in Computer Science and Engineering of PES University, Bengaluru during the academic semester **<Internship Duration>**. The matter embodied in this report has not been submitted to any other university or institution for the award of any degree.

---

## Acknowledgement

I would like to express my gratitude to my guide **<Guide/Manager Name>**, **<Designation>**, **<Company Name>** for their continuous guidance, assistance, and encouragement throughout the development of this project.

I am grateful to the internship coordinators **Prof. Mahitha G** and **Dr. Bhargavi M**, Department of Computer Science and Engineering, PES University, for organizing, managing, and supporting the internship process.

I take this opportunity to thank **Dr. Shylaja S S**, Chairperson, Department of Computer Science and Engineering, PES University, for the knowledge and support received from the department.

I am deeply grateful to **Prof. Jawahar Doreswamy**, Chancellor, PES University, and **Dr. Suryaprasad J**, Vice-Chancellor, PES University, for providing opportunities and encouragement throughout this internship.

Finally, this internship could not have been completed without the continual support and encouragement of my parents and friends.

---

## Abstract

This internship project investigates whether data skew in distributed big data jobs can be predicted before execution using only pre-execution information. Data skew occurs when one or more tasks in a job take significantly longer than the average task, creating stragglers that delay the entire job. The project addresses this problem by building a leakage-free machine learning pipeline that extracts job-level features from cluster traces, labels jobs using runtime statistics, trains several lightweight models, and evaluates them using metrics suitable for imbalanced classification.

The implemented workflow uses raw task event data from Google Cluster Workload Traces, processes the records into job-level samples, and creates binary skew labels using the condition that the slowest task runtime is at least twice the average task runtime. Models including Logistic Regression, Random Forest, XGBoost, and LightGBM are trained and calibrated. The system also includes baseline comparison, synthetic data generation for controlled experiments, threshold tuning, logging, and mitigation simulation. The results show that the real trace contains severe class imbalance and weak signal in the available pre-execution features, while synthetic data with stronger feature correlations demonstrates that the prediction problem is learnable when the right information is available.

The project demonstrates a complete research and engineering pipeline for early skew prediction, highlights the limitations of the available real data sample, and provides a reproducible base for future work in scheduling and performance optimization for distributed systems.

---

## Table of Contents

1. Introduction  
2. Company / Research Center Brief Introduction  
3. Internship Project Details  
4. Project Abstract and Scope  
5. Project Design Details and Technologies Used  
6. Coding / Implementation Details  
7. Project Results and Learning Outcomes  
8. Conclusion  
9. References / Bibliography

---

## List of Tables and Figures

### Tables

1. Job-level pre-execution features  
2. Model performance on real data  
3. Key dataset statistics  
4. Learning outcomes summary

### Figures

1. End-to-end data flow diagram  
2. Model training and evaluation workflow  
3. Confusion matrix for best real-data model  
4. Feature importance summary  
5. Synthetic vs real transfer comparison

---

## 1. Introduction

### 1.1 Background

Distributed data processing systems such as Hadoop, Spark, and similar cluster-based platforms run large jobs by splitting work into many tasks. In practice, these tasks do not always execute at the same speed. A small number of slower tasks can dominate the total completion time of the job, even if most tasks finish quickly. This imbalance is commonly referred to as data skew.

Data skew matters because it causes resource underutilization, longer job latency, inefficient scheduling, and poor user experience. In cloud and cluster environments, even a small number of stragglers can significantly reduce the throughput of the whole system. For that reason, predicting skew early is useful for workload scheduling, resource planning, and proactive mitigation.

### 1.2 Problem Statement

The core problem addressed in this internship is whether skewed jobs can be identified **before execution** using only information available at submission time. This is a stricter and more practical setting than runtime detection because the model must avoid leakage from task completion times or other post-execution signals.

The classification target is binary:

skewed = 1 if max(task runtime) >= 2 x avg(task runtime)

otherwise the job is labeled non-skewed.

### 1.3 Objectives

The project was designed with the following objectives:

1. Build a leakage-free pipeline for job-level skew prediction.
2. Use only pre-execution features that are available at submission time.
3. Compare multiple lightweight machine learning models suitable for deployment.
4. Evaluate performance using metrics that are appropriate for highly imbalanced data.
5. Study the limits of the real dataset and validate the idea using synthetic data.
6. Provide a reproducible workflow with logging, saved artifacts, and clear outputs.

### 1.4 Motivation

The motivation behind this work is operational. If a scheduler can identify jobs that are likely to suffer from skew before execution begins, it can take preventive action such as changing resource allocation, adjusting scheduling priorities, or applying mitigation policies. The project therefore combines machine learning with systems thinking: the value is not only in prediction accuracy, but also in whether the prediction can support practical action.

---

## 2. Company / Research Center Brief Introduction

<Company Full Name> is the internship organization where the project work was carried out. The work environment focused on practical software engineering, data processing, and model-driven analysis. The internship exposed the student to real-world experimentation, reproducible workflows, and the discipline required to move from exploratory analysis to a complete deliverable.

In this setting, the project was approached as a research-oriented engineering task. The focus was on understanding the problem, defining a measurable target, implementing a reliable pipeline, and validating the results through experiments. The internship environment also encouraged documentation, versioned outputs, and careful interpretation of results rather than relying on a single metric or a single model.

If your company description needs to be specific, replace this section with the actual domain, products, team function, and technology stack of the internship host.

---

## 3. Internship Project Details

### 3.1 Project Title

**Early Prediction of Data Skew in Cloud-Based Big Data Jobs Using Lightweight Machine Learning Models**

### 3.2 Relationship to a Larger Project

The internship work can be understood as part of a larger effort to build intelligent observability and mitigation tools for distributed systems. The current project focuses on one key subproblem: predicting skew before execution using metadata available at job submission time. This forms the foundation for a more complete scheduling or mitigation system that could later include runtime monitoring, adaptive resource allocation, and automated policy decisions.

### 3.3 My Role and Responsibilities

My responsibilities during the internship included:

1. Studying the problem of data skew and identifying how it affects distributed job performance.
2. Preparing and cleaning task event data from cluster traces.
3. Designing features that are safe to use before execution and avoiding leakage.
4. Implementing skew labeling logic based on task runtime statistics.
5. Training and comparing several machine learning models.
6. Evaluating results with imbalanced-data metrics and generating plots and tables.
7. Documenting the methodology, limitations, and outcomes in a reproducible format.

### 3.4 Project Abstract and Scope

The scope of the project covers end-to-end analysis from raw trace files to model evaluation. It includes data loading, preprocessing, runtime extraction, job-level aggregation, skew labeling, feature engineering, model training, threshold tuning, and result analysis. The project also includes supporting utilities for prediction, validation, logging, and mitigation simulation.

The project deliberately stays within a pre-execution setting. It does not use runtime information as input features during prediction. This design choice makes the problem more difficult but also more realistic for deployment because the model can be used at job submission time.

### 3.5 Key Deliverables

The main deliverables from the internship are:

1. A cleaned and labeled job-level dataset.
2. A reusable machine learning training and evaluation pipeline.
3. A set of trained models for skew prediction.
4. Comparison against a simple rule-based baseline.
5. A synthetic data workflow for controlled experiments.
6. A mitigation simulation module for studying operational impact.
7. A written report and supporting documentation.

---

## 4. Project Abstract and Scope

### 4.1 Problem Context

The input data comes from a cluster workload trace that records task events over time. Each job contains multiple tasks, and the goal is to infer whether the job will be skewed based only on the information known before execution starts. In the real world, this would support early decision making in a scheduler or workload manager.

### 4.2 Why This Problem is Hard

This is a difficult prediction problem for several reasons:

1. The positive class is rare.
2. Many useful signals are hidden until runtime.
3. Job behavior can vary widely across templates and priorities.
4. The available real sample is small compared with the complexity of the workload.
5. Pre-execution features often have weak direct correlation with skew.

Because of these factors, a simple high-accuracy model can still be poor in practice. For that reason, the evaluation emphasizes precision, recall, PR-AUC, calibration, and baseline comparison rather than accuracy alone.

### 4.3 Scope Boundaries

The project scope includes:

1. Pre-execution prediction.
2. Job-level classification.
3. Lightweight supervised learning models.
4. Reproducible experiments and logging.

The project does not attempt to:

1. Modify the actual cluster scheduler.
2. Perform online runtime intervention.
3. Use deep learning models that would be difficult to interpret or deploy for this use case.

### 4.4 Expected Value

If the prediction quality improves enough, the model can support practical decisions such as prioritizing risky jobs, allocating extra resources, or triggering skew-aware mitigation policies. Even when predictive performance is limited, the work still provides value by clarifying what data is missing and what kinds of features are necessary for reliable skew prediction.

---

## 5. Project Design Details and Technologies Used

### 5.1 System Design

The project uses a modular pipeline architecture. Each stage has a separate responsibility, which keeps the code maintainable and makes the experiments easier to reproduce. The system follows a linear data flow:

1. Load raw task events.
2. Clean and normalize the data.
3. Extract runtime and job-level statistics.
4. Create skew labels.
5. Engineer pre-execution features.
6. Split data into train and test sets.
7. Train and calibrate models.
8. Evaluate results and store artifacts.

### 5.2 Feature Design

The most important design principle is that prediction features must be available before execution. The features used in the model are therefore limited to metadata and aggregated submission-time properties. Typical features include job size, scheduling class, priority, and statistics derived from resource requests such as mean and standard deviation.

Table 1 summarizes the main pre-execution features.

| Feature | Description | Reason for Use |
|---|---|---|
| `num_tasks` | Number of tasks in the job | Larger jobs are more likely to contain imbalance |
| `scheduling_class` | Scheduling category | Reflects workload urgency and behavior |
| `priority` | Task priority | Lower-priority jobs may experience more variability |
| `cpu_request_mean` | Mean CPU request | Indicates average resource demand |
| `cpu_request_std` | CPU request variability | Captures task heterogeneity |
| `memory_request_mean` | Mean memory request | Captures memory demand |
| `memory_request_std` | Memory request variability | Captures imbalance in memory requirements |
| `disk_space_request_mean` | Mean disk request | Captures storage demand |
| `disk_space_request_std` | Disk request variability | Captures storage heterogeneity |
| `different_machine_constraint_mean` | Placement flexibility | Indicates scheduling constraint level |

### 5.3 Model Selection

The project uses lightweight models that are practical for fast experimentation and deployment:

1. Logistic Regression.
2. Random Forest.
3. XGBoost.
4. LightGBM.

These models were selected because they provide probability outputs, are relatively efficient, and can be calibrated for use in decision support systems. They also create a good balance between interpretability and nonlinear modeling capability.

### 5.4 Data Handling Strategy

Class imbalance is one of the central challenges in this project. The positive class is very small compared with the negative class, so the pipeline uses methods such as SMOTE, class weighting, and calibration to improve minority-class learning. Evaluation uses PR-AUC and recall-focused analysis because accuracy alone would hide the actual difficulty of the task.

### 5.5 Tools and Technologies

The main technologies used in the project are:

1. Python.
2. pandas and numpy for data processing.
3. scikit-learn for modeling and evaluation.
4. XGBoost and LightGBM for gradient boosting models.
5. imbalanced-learn for SMOTE.
6. matplotlib and seaborn for visualizations.
7. Jupyter Notebook for exploratory analysis.
8. Logging utilities for reproducibility.

### 5.6 Guideline-Based Formatting Considerations

The report follows the structure requested in the internship guidelines: title page, company certificate, declaration, acknowledgement, abstract, contents, list of tables and figures, introduction, company brief, project details, implementation, results, conclusion, and references. The document is also written in a style that can be transferred into Word and formatted with 1.5 line spacing, Times New Roman 12 pt body text, and bold underlined section headings as required.

---

## 6. Coding / Implementation Details

### 6.1 Repository Structure

The implementation is organized into reusable scripts and modules. The major folders and files support data processing, model training, evaluation, inference, and documentation.

| Area | Example Files | Purpose |
|---|---|---|
| Data loading and preprocessing | `src/data_loader.py`, `src/preprocessing.py` | Read and clean raw trace data |
| Feature and label creation | `src/feature_engineering.py`, `src/skew_labeling.py` | Build job-level features and labels |
| Data splitting | `src/splitters.py` | Create time-based or template-based splits |
| Model training | `src/train_model.py` | Fit and calibrate classifiers |
| Evaluation | `src/evaluate_model.py`, `src/baseline.py` | Compute metrics and compare with baseline |
| Logging | `src/logger.py` | Save reproducible output logs |
| Inference and simulation | `predict_job.py`, `mitigation_simulation.py`, `live_predict_from_file.py` | Run prediction and impact analysis |

### 6.2 End-to-End Pipeline

The end-to-end flow can be summarized as follows:

1. Load raw task event data from the cluster trace.
2. Filter and clean the records.
3. Group task-level events into job-level samples.
4. Extract runtimes and calculate skew labels.
5. Build a feature table using only pre-execution information.
6. Split the data in a way that reduces leakage.
7. Train several models with imbalance handling.
8. Calibrate probabilities and optimize thresholds.
9. Evaluate performance using classification and calibration metrics.
10. Save models, plots, logs, and summary tables.

### 6.3 Labeling Method

Jobs are labeled as skewed when the runtime of the slowest task is at least twice the average runtime of tasks in that job. This definition is simple, transparent, and easy to reproduce. It also matches the practical intuition that a single slow task can make the job noticeably worse.

The project uses runtime statistics only for labeling, not for prediction features. This separation is important because it avoids leakage and preserves the integrity of the evaluation.

### 6.4 Training Methodology

The training workflow includes class balancing, calibration, and cross-validation. Since the data is imbalanced, SMOTE is used on the training portion to synthesize minority examples. The models are then calibrated so that predicted probabilities are more meaningful for threshold-based decisions.

Training is kept lightweight enough to run on a standard laptop. This makes the work easier to reproduce and demonstrates that the pipeline does not depend on specialized hardware.

### 6.5 Baseline Comparison

To ensure the machine learning models are useful, the project compares them against a rule-based baseline using a simple threshold on `num_tasks`. This is important because a model should beat a practical heuristic, not just an empty reference point.

### 6.6 Logging and Reproducibility

All major scripts support logging so that experiments can be reproduced later. The saved logs document dataset statistics, model metrics, thresholds, and analysis results. This is particularly important for internship work because the report should be backed by a traceable execution history.

### 6.7 Inference and Operational Use

The project includes helper scripts for prediction and mitigation simulation. These scripts show how the trained model could be used after training: a job can be scored, a threshold can be applied, and the result can be interpreted as a risk signal for skew-aware handling.

---

## 7. Project Results and Learning Outcomes

### 7.1 Dataset Summary

The real cluster trace sample used in the project contains severe class imbalance. After preprocessing, the usable job-level dataset contains 4,057 valid jobs, of which only 68 are labeled skewed.

| Statistic | Value |
|---|---|
| Total valid jobs | 4,057 |
| Skewed jobs | 68 |
| Non-skewed jobs | 3,989 |
| Positive class share | 1.68% |
| Imbalance ratio | Approximately 1:58.7 |

This imbalance strongly influences model behavior. A model can achieve high overall accuracy by predicting the majority class, while still failing to identify the rare skewed cases that matter most.

### 7.2 Real Data Performance

The following table summarizes the main results on the held-out real dataset.

| Model | Accuracy | Precision | Recall | F1-Score | PR-AUC | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9483 | 0.0732 | 0.8235 | 0.1364 | 0.0424 | 0.5200 |
| Random Forest | 0.9520 | 0.0769 | 0.7941 | 0.1429 | 0.0929 | 0.5200 |
| XGBoost | 0.9483 | 0.0714 | 0.8529 | 0.1333 | 0.0928 | 0.4963 |
| LightGBM | 0.9483 | 0.0714 | 0.8529 | 0.1333 | 0.0927 | 0.4903 |

### 7.3 Interpretation of Results

The results show a mixed picture. Recall is relatively high because the models flag many of the skewed jobs, but precision is very low because many of those flags are false positives. PR-AUC remains low, which indicates that the ranking quality is still weak. ROC-AUC is close to random behavior, which confirms that the available pre-execution features in this real sample do not strongly separate the classes.

Even so, the models are still useful from an engineering perspective because they outperform or match simple heuristics in some settings and reveal where the data is insufficient. The real value of the project is not only the metrics themselves, but also the insight that the current trace sample is too limited for strong deployment claims.

### 7.4 Baseline Comparison

The rule-based baseline uses a threshold on job size. It provides a simple, transparent reference point, but it cannot fully capture the complexity of skew behavior. The machine learning models generally improve recall and provide a probabilistic score, but the overall gain is limited by the weak signal in the real data.

### 7.5 Synthetic Data Insights

The synthetic workflow is an important validation step. When the synthetic data was generated with explicit correlations between job features and skew, the models achieved near-perfect performance. This result demonstrates that the modeling pipeline is capable of learning the problem when the data contains the right signal.

The synthetic experiments therefore serve a key purpose: they separate pipeline capability from dataset quality. If a model performs well on synthetic data but poorly on real data, the limitation is likely the data distribution rather than the implementation itself.

### 7.6 Mitigation and Threshold Analysis

Threshold optimization and mitigation simulation were used to study how model predictions could influence downstream decisions. This is important because a prediction model for skew is only useful if its outputs can drive actions such as resource changes, scheduling interventions, or alerting.

### 7.7 Key Learning Outcomes

This internship produced several practical learning outcomes:

1. I learned how to turn raw trace data into a structured machine learning dataset.
2. I gained experience working with imbalanced classification problems.
3. I understood the importance of leakage-free feature design.
4. I learned how calibration and thresholding affect model usefulness.
5. I saw how synthetic experiments can validate a pipeline even when real data is limited.
6. I improved my ability to write reproducible, research-style documentation.
7. I developed a better understanding of how ML can support distributed systems decisions.

### 7.8 Lessons from the Project

The main lesson from the project is that model quality depends heavily on data quality and feature relevance. A technically correct pipeline is not enough if the available features do not contain enough information to separate the classes. At the same time, the project shows that careful preprocessing, calibration, and validation can still produce a useful research artifact and a practical foundation for future work.

---

## 8. Conclusion

This internship project successfully built a complete, leakage-free machine learning pipeline for early prediction of data skew in cloud-based big data jobs. The work covered data preparation, skew labeling, feature engineering, model training, evaluation, logging, and mitigation analysis. The final system shows that the problem is learnable in principle, but the real trace sample used in the project is highly imbalanced and does not provide strong predictive signal with the available pre-execution features.

The project therefore contributes in two ways. First, it provides a reproducible implementation that can be extended with better data, better features, or more advanced deployment logic. Second, it clarifies the practical limitations of the current dataset and shows why future work must focus on richer traces or stronger feature sources if meaningful real-world prediction is desired.

Overall, the internship was valuable both as a software engineering exercise and as a systems research exercise. It improved my understanding of distributed workload behavior, machine learning for imbalanced problems, and the importance of building solutions that are scientifically defensible and operationally relevant.

---

## 9. References / Bibliography

[1] Google Cluster Trace documentation and publicly available workload trace descriptions used for the raw event data study.  
[2] scikit-learn documentation, for model training, evaluation, calibration, and preprocessing utilities.  
[3] XGBoost documentation, for gradient boosting model implementation.  
[4] LightGBM documentation, for efficient boosting-based classification.  
[5] imbalanced-learn documentation, for SMOTE-based minority class handling.  
[6] pandas and numpy documentation, for data wrangling and numerical processing.

---

## Appendix A. Suggested Final Submission Checklist

Before converting this draft into the final Word report, verify the following:

1. Replace all placeholder names with the actual company, guide, and internship duration details.
2. Insert the company certificate or joining letter page.
3. Add the final formatted Table of Contents in Word with page numbers.
4. Insert the List of Tables and List of Figures with actual page references.
5. Apply the required Word formatting: Times New Roman, 12 pt body text, 1.5 line spacing, and the specified title styling.
6. Add any company-specific restrictions or redactions if the project details cannot be shared.
7. Verify that the reference list matches the final citations used in the report.
