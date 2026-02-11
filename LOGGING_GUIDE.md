# Output Logging Guide

## Overview

All main scripts now support saving terminal output to log files using the `--save-log` flag. This makes it much easier to review results, compare experiments, and prepare research papers.

## Quick Start

Simply add `--save-log` to any of the main scripts:

```powershell
python main.py --save-log
python main_sample.py --save-log
python early_exec_experiment.py --save-log
python explainability_report.py --save-log
python mitigation_simulation.py --save-log
```

## Output Directory

- All log files are saved to: `outputs/`
- File naming format: `<script_name>_<timestamp>.log`
- Example: `outputs/main_20240115_143052.log`

## Benefits

✓ **Easy Review**: Scroll through complete output at your own pace  
✓ **Comparison**: Compare results across different runs side-by-side  
✓ **Searchable**: Use text editors or grep to find specific metrics  
✓ **Reproducibility**: Keep logs for documentation and verification  
✓ **Paper-Ready**: Extract metrics directly for research papers  

## Example Workflow

### 1. Run experiments with logging

```powershell
# Run main pipeline
python main_sample.py --save-log

# Run timing study
python early_exec_experiment.py --save-log

# Generate explainability report
python explainability_report.py --save-log
```

### 2. Review saved logs

```powershell
# List all log files
dir outputs\

# View a specific log
notepad outputs\main_sample_20240115_143052.log

# Search for specific metrics
findstr "PR-AUC" outputs\*.log
```

### 3. Compare results

Open multiple log files and compare:
- Model performance metrics
- Feature importance rankings  
- Early-execution timing results
- Policy simulation outcomes

## Log File Contents

Each log file contains the complete terminal output including:

- Script configuration and parameters
- Data loading and preprocessing steps
- Model training progress
- Evaluation metrics (PR-AUC, ROC-AUC, F1, etc.)
- Comparison tables
- Feature importance rankings
- Confusion matrices
- Error messages and warnings

## Tips

1. **Run experiments overnight**: Use `--save-log` for long-running experiments, review results in the morning

2. **Compare configurations**: Run the same script with different parameters and compare log files

3. **Extract tables**: Copy metric tables from logs directly into research papers

4. **Archive results**: Keep log files organized by date/experiment for reproducibility

5. **Grep/Search**: Use command-line tools to extract specific metrics across all runs:
   ```powershell
   findstr "F1-Score" outputs\*.log
   findstr "PR-AUC" outputs\*.log
   ```

## Technical Details

The logging system uses a "tee" approach:
- Output is displayed in terminal in real-time
- Simultaneously written to timestamped log file
- Original stdout restored after completion
- Exception-safe (logs closed even if script errors)

## Implementation

The logging functionality is provided by `src/logger.py`:
- `setup_logging(script_name)`: Initialize logger
- `close_logging(logger)`: Clean up and restore stdout
- `TeeLogger`: Class that writes to both terminal and file

All scripts use the same pattern:
```python
if args.save_log:
    logger = setup_logging("script_name")
try:
    main()
finally:
    if logger:
        close_logging(logger)
```
