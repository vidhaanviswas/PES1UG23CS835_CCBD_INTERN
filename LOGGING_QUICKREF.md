# Output Logging - Quick Reference

## Before vs After

### Before (Terminal Output Only)
```
> python main.py
Loading data...
Training models...
[lots of output...]
Evaluation Results:
  Model: Logistic Regression
  PR-AUC: 0.8567
  ...
[output scrolls off screen]
```

**Problems:**
- ❌ Output scrolls away and is hard to review
- ❌ Cannot easily compare results from different runs
- ❌ No permanent record of experiments
- ❌ Difficult to extract metrics for papers

### After (Saved to File)
```
> python main.py --save-log
Logging to: outputs/main_20240115_143052.log
================================================================================
Loading data...
Training models...
[all output shown in terminal AND saved to file]
```

**Benefits:**
- ✅ Complete output saved to timestamped log file
- ✅ Can review results at your leisure
- ✅ Easy to compare multiple experiments
- ✅ Permanent record for reproducibility
- ✅ Search/grep through logs for specific metrics

## Command Reference

| Script | Command | Log File |
|--------|---------|----------|
| Main pipeline | `python main.py --save-log` | `outputs/main_TIMESTAMP.log` |
| Sample pipeline | `python main_sample.py --save-log` | `outputs/main_sample_TIMESTAMP.log` |
| Early-execution | `python early_exec_experiment.py --save-log` | `outputs/early_exec_experiment_TIMESTAMP.log` |
| Explainability | `python explainability_report.py --save-log` | `outputs/explainability_report_TIMESTAMP.log` |
| Mitigation | `python mitigation_simulation.py --save-log` | `outputs/mitigation_simulation_TIMESTAMP.log` |

## Quick Workflow

### 1. Run with Logging
```powershell
python main_sample.py --save-log
```

### 2. Find Your Log
```powershell
dir outputs\
# Look for: main_sample_20240115_143052.log
```

### 3. Review Results
```powershell
notepad outputs\main_sample_20240115_143052.log
```

### 4. Search for Metrics
```powershell
findstr "PR-AUC" outputs\*.log
findstr "F1-Score" outputs\*.log
```

### 5. Compare Experiments
```powershell
python analyze_logs.py
```

## Log File Contents

Each log contains:
- ✓ Complete pipeline execution output
- ✓ Data loading statistics
- ✓ Model training progress
- ✓ All evaluation metrics (PR-AUC, ROC-AUC, F1, etc.)
- ✓ Comparison tables
- ✓ Feature importance rankings
- ✓ Error messages and warnings

## Tips

💡 **Run overnight experiments with logging**: Set up long runs and review results in the morning

💡 **Compare configurations**: Run same script with different parameters and compare logs

💡 **Extract for papers**: Copy metric tables from logs directly to publications

💡 **Keep for reproducibility**: Archive logs by date/experiment

## More Information

- See [LOGGING_GUIDE.md](LOGGING_GUIDE.md) for complete documentation
- See [LOGGING_IMPLEMENTATION.md](LOGGING_IMPLEMENTATION.md) for technical details
- Run `python logging_example.py` for usage examples
- Run `python test_logging.py` to test the feature
