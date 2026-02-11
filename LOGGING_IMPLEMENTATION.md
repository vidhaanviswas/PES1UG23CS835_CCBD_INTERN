# Output Logging Feature - Implementation Summary

## Overview

Added comprehensive logging capabilities to all main pipeline scripts, allowing users to save terminal output to timestamped log files for easier review, comparison, and analysis.

## Problem Statement

Terminal output from long-running experiments is difficult to review because:
- Output scrolls off the screen
- Cannot easily search or compare results
- Hard to extract metrics for research papers
- No permanent record of experiment results

## Solution

Implemented a `--save-log` command-line flag for all main scripts that:
- Saves complete terminal output to timestamped log files
- Still displays output in terminal in real-time (tee functionality)
- Creates organized outputs/ directory structure
- Provides utilities for log analysis and comparison

## Files Modified/Created

### New Files

1. **src/logger.py** - Core logging utilities
   - `TeeLogger` class: Writes to both terminal and file
   - `setup_logging()`: Initialize logger with timestamp
   - `close_logging()`: Clean up and restore stdout

2. **outputs/.gitkeep** - Log output directory placeholder

3. **LOGGING_GUIDE.md** - Complete user guide for logging features

4. **logging_example.py** - Examples of using --save-log flag

5. **test_logging.py** - Test script to verify logging works

6. **analyze_logs.py** - Utility to extract and compare metrics from logs

### Modified Files

1. **main.py**
   - Added `import argparse` and `from logger import setup_logging, close_logging`
   - Changed `if __name__ == "__main__"` to parse arguments
   - Wrapped main() with logging setup/teardown

2. **main_sample.py**
   - Same changes as main.py

3. **early_exec_experiment.py**
   - Added argparse and logging support
   - Wrapped main() with logging setup/teardown

4. **explainability_report.py**
   - Added argparse and logging support
   - Wrapped run() with logging setup/teardown

5. **mitigation_simulation.py**
   - Added argparse and logging support
   - Wrapped run() with logging setup/teardown

6. **README.md**
   - Added "Saving Outputs to Files" section
   - Updated project structure to include outputs/ and logger.py
   - Updated expected outputs section
   - Added reference to LOGGING_GUIDE.md

7. **ARCHITECTURE_SUMMARY.md**
   - Added note about --save-log flag at top

## Usage Examples

### Basic Usage

```powershell
# Run any script with logging
python main.py --save-log
python main_sample.py --save-log
python early_exec_experiment.py --save-log
python explainability_report.py --save-log
python mitigation_simulation.py --save-log
```

### Test Logging

```powershell
# Test that logging works correctly
python test_logging.py
```

### Analyze Logs

```powershell
# Compare metrics across all log files
python analyze_logs.py

# See example usage
python logging_example.py
```

### Review Logs

```powershell
# List all log files
dir outputs\

# View specific log
notepad outputs\main_sample_20240115_143052.log

# Search for metrics
findstr "PR-AUC" outputs\*.log
```

## Technical Implementation

### Logging Architecture

```
┌─────────────────┐
│  Script with    │
│  --save-log     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ setup_logging() │
│ - Create output │
│   directory     │
│ - Generate      │
│   timestamp     │
│ - Replace       │
│   sys.stdout    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   TeeLogger     │
│ - Write to      │
│   terminal      │
│ - Write to      │
│   log file      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ close_logging() │
│ - Restore       │
│   sys.stdout    │
│ - Close file    │
└─────────────────┘
```

### TeeLogger Class

```python
class TeeLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)  # Display in terminal
        self.log.write(message)        # Save to file
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
```

### Argument Parsing Pattern

All scripts follow this consistent pattern:

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("--save-log", action="store_true", 
                        help="Save terminal output to log file")
    args = parser.parse_args()
    
    logger = None
    if args.save_log:
        logger = setup_logging("script_name")
    
    try:
        main()
    finally:
        if logger:
            close_logging(logger)
```

## File Naming Convention

Log files use this format:
```
<script_name>_<timestamp>.log
```

Examples:
- `main_20240115_143052.log`
- `main_sample_20240115_150234.log`
- `early_exec_experiment_20240115_163145.log`
- `explainability_report_20240115_171523.log`

Timestamp format: `YYYYMMDD_HHMMSS`

## Benefits

1. **Easy Review**: Scroll through complete output at leisure
2. **Comparison**: Open multiple logs side-by-side to compare experiments
3. **Searchable**: Use grep/findstr to find specific metrics
4. **Reproducibility**: Keep permanent record of all experiment results
5. **Paper-Ready**: Extract metrics directly for research papers
6. **No Loss**: Still see output in terminal during execution

## Directory Structure

```
project_root/
├── outputs/                          # NEW: Log output directory
│   ├── .gitkeep
│   ├── main_20240115_143052.log
│   ├── main_sample_20240115_150234.log
│   └── ...
├── src/
│   ├── logger.py                     # NEW: Logging utilities
│   └── ...
├── main.py                           # MODIFIED: Added logging support
├── main_sample.py                    # MODIFIED: Added logging support
├── early_exec_experiment.py          # MODIFIED: Added logging support
├── explainability_report.py          # MODIFIED: Added logging support
├── mitigation_simulation.py          # MODIFIED: Added logging support
├── test_logging.py                   # NEW: Test script
├── analyze_logs.py                   # NEW: Log analysis utility
├── logging_example.py                # NEW: Usage examples
├── LOGGING_GUIDE.md                  # NEW: Complete user guide
└── README.md                         # MODIFIED: Added logging docs
```

## Testing

Run the test script to verify logging works:

```powershell
python test_logging.py
```

Expected behavior:
1. Creates timestamped log file in outputs/
2. Displays test messages in terminal
3. Saves same messages to log file
4. Shows confirmation message after logger closes

Then check:
```powershell
dir outputs\test_*.log
notepad outputs\test_<timestamp>.log
```

## Future Enhancements

Possible improvements for later:
1. JSON output format for machine-readable logs
2. CSV export of metric tables
3. Automatic log comparison reports
4. Integration with experiment tracking platforms (MLflow, W&B)
5. Log compression for long-term storage

## Summary

All main scripts now support `--save-log` flag to save complete terminal output to timestamped log files in the `outputs/` directory. This makes it much easier to review results, compare experiments, and prepare research papers. The implementation is consistent across all scripts and includes utilities for log analysis and comparison.
