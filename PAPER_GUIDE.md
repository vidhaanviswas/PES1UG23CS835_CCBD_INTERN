# Using Logs for Research Papers

This guide shows how to extract and format metrics from log files for research paper tables and figures.

## Extracting Metric Tables

### Example: Comparison Table

**From log file:**
```
================================================================================
MODEL COMPARISON (Time-based Split)
================================================================================
Model                     PR-AUC    ROC-AUC   F1        Precision Recall    Brier     ECE
------------------------------------------------------------------------------------------------
Logistic Regression       0.8567    0.9234    0.7891    0.8123    0.7654    0.1234    0.0456
Random Forest             0.8789    0.9345    0.8012    0.8234    0.7823    0.1123    0.0389
XGBoost                   0.8923    0.9456    0.8156    0.8345    0.7967    0.1056    0.0312
Baseline                  0.7234    0.8456    0.6789    0.7123    0.6456    0.1567    0.0678
```

**Convert to LaTeX:**
```latex
\begin{table}[h]
\centering
\caption{Model Performance Comparison (Time-based Split)}
\label{tab:model_comparison}
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{PR-AUC} & \textbf{ROC-AUC} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} & \textbf{Brier} \\
\hline
Logistic Regression & 0.8567 & 0.9234 & 0.7891 & 0.8123 & 0.7654 & 0.1234 \\
Random Forest       & 0.8789 & 0.9345 & 0.8012 & 0.8234 & 0.7823 & 0.1123 \\
XGBoost            & \textbf{0.8923} & \textbf{0.9456} & \textbf{0.8156} & \textbf{0.8345} & \textbf{0.7967} & \textbf{0.1056} \\
Baseline           & 0.7234 & 0.8456 & 0.6789 & 0.7123 & 0.6456 & 0.1567 \\
\hline
\end{tabular}
\end{table}
```

## Extracting Feature Importance

### Example: Top Features

**From explainability log:**
```
================================================================================
EXPLAINABILITY: random_forest
================================================================================
Top features (SHAP):
  num_tasks: 0.3456
  cpu_request_mean: 0.2345
  memory_request_mean: 0.1789
  scheduling_class: 0.1234
  priority: 0.0987
  disk_space_request_mean: 0.0789
  cpu_request_std: 0.0567
  memory_request_std: 0.0456
  disk_space_request_std: 0.0234
  different_machine_constraint_mean: 0.0123
```

**Convert to paper format:**
```
The most important features for skew prediction are:
1. Number of tasks (SHAP value: 0.346)
2. Mean CPU request (0.235)
3. Mean memory request (0.179)
4. Scheduling class (0.123)
5. Task priority (0.099)
```

## Extracting Results for Figures

### Example: Early-Execution Timing Study

**From early_exec_experiment log:**
```
=== Early-exec experiment: k=0.05 ===
...
PR-AUC: 0.7123

=== Early-exec experiment: k=0.10 ===
...
PR-AUC: 0.7867

=== Early-exec experiment: k=0.20 ===
...
PR-AUC: 0.8345
```

**Data for plotting:**
```python
k_values = [0.05, 0.10, 0.20]
pr_auc_values = [0.7123, 0.7867, 0.8345]

plt.plot(k_values, pr_auc_values, marker='o')
plt.xlabel('Fraction of Tasks Observed')
plt.ylabel('PR-AUC')
plt.title('Early-Execution Prediction Quality')
```

## Comparing Splits

### Example: Time vs Template Split

**Extract from log:**
```
Time-based split:
  XGBoost PR-AUC: 0.8923
  
Template-based split:
  XGBoost PR-AUC: 0.8756
```

**Discussion point:**
```
Our model achieves 0.892 PR-AUC on the time-based split (evaluating
on chronologically future jobs) and 0.876 on the template-based split
(evaluating on unseen job templates). The slight degradation (1.7%)
suggests good generalization to new workload patterns.
```

## Mitigation Policy Results

### Example: Policy Simulation

**From mitigation_simulation log:**
```
================================================================================
MITIGATION POLICY SIMULATION
================================================================================
Total jobs: 5000
Actions taken: 1250 (25.0%)
True Positives: 900 (18.0%)
False Positives: 350 (7.0%)
Duration reduction (skewed jobs): 27.3%
Duration increase (non-skewed jobs): 5.2%
Net improvement: 18.9%
```

**Convert to results section:**
```
We simulated a mitigation policy that reschedules jobs predicted as
skewed (threshold: 0.7). On a test set of 5,000 jobs:
- Policy triggered on 25% of jobs
- 72% true positive rate (900/1250)
- Reduced duration of skewed jobs by 27.3%
- Net improvement of 18.9% in overall cluster efficiency
```

## Automated Extraction Script

```powershell
# Search all logs for key metrics
findstr "PR-AUC" outputs\*.log > results_pr_auc.txt
findstr "F1-Score" outputs\*.log > results_f1.txt
findstr "calibration error" outputs\*.log > results_calibration.txt

# Extract feature importance
findstr "Top features" outputs\explainability*.log > results_features.txt

# Extract timing results
findstr "k=" outputs\early_exec*.log > results_timing.txt
```

## Tips for Paper Writing

1. **Run multiple times**: Execute pipeline multiple times with different random seeds and report mean ± std

2. **Save configurations**: Keep a record of which log corresponds to which experiment setup

3. **Extract systematically**: Use grep/findstr to pull all instances of a metric across experiments

4. **Version control logs**: Archive important logs with descriptive names:
   ```
   outputs/main_baseline_experiment.log
   outputs/main_with_smote.log
   outputs/main_final_results.log
   ```

5. **Cross-reference**: Keep notes mapping log files to paper sections:
   ```
   Table 1 (Model Comparison) → main_20240115_143052.log
   Table 2 (Feature Importance) → explainability_20240115_150234.log
   Figure 3 (Timing Study) → early_exec_20240115_163145.log
   ```

## Example Paper Sections

### Results Section Template

```markdown
## 4. Experimental Results

### 4.1 Model Performance

We evaluated four models on the time-based split (Table 1). XGBoost
achieved the best performance with PR-AUC of 0.892, outperforming
the rule-based baseline by 23.3%.

[Log: main_20240115_143052.log, lines 145-167]

### 4.2 Feature Importance

The most influential features for skew prediction were: (1) number of
tasks, (2) mean CPU request, and (3) mean memory request (Figure 2).

[Log: explainability_20240115_150234.log, lines 34-56]

### 4.3 Early Prediction

Using only the first 10% of tasks, our model achieves 0.787 PR-AUC,
demonstrating the feasibility of early skew detection (Figure 3).

[Log: early_exec_20240115_163145.log, lines 78-95]
```

## Checklist for Paper Submission

- [ ] All metric tables extracted and verified
- [ ] Feature importance rankings documented
- [ ] Timing study results compiled
- [ ] Standard deviations computed (multiple runs)
- [ ] Log files archived with paper submission
- [ ] Results reproducible from saved logs
- [ ] Figures generated from logged data
- [ ] Cross-references between logs and paper sections

## Conclusion

By systematically saving and organizing log files, you can:
- Extract metrics efficiently for tables
- Generate data for figures
- Document experimental configurations
- Ensure reproducibility
- Speed up paper writing significantly

See [LOGGING_GUIDE.md](LOGGING_GUIDE.md) for more details on the logging system.
