"""
Log Analysis Utility

Helper script to extract and compare metrics from saved log files.
"""

import sys
import re
from pathlib import Path
from typing import Dict, List
import argparse


def extract_metrics_from_log(log_path: Path) -> Dict:
    """Extract key metrics from a log file."""
    metrics = {
        'file': log_path.name,
        'timestamp': None,
        'pr_auc': {},
        'roc_auc': {},
        'f1': {},
        'precision': {},
        'recall': {},
        'brier': {},
        'ece': {}
    }
    
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract timestamp from filename
    match = re.search(r'_(\d{8}_\d{6})\.log', log_path.name)
    if match:
        metrics['timestamp'] = match.group(1)
    
    # Extract model metrics (simplified patterns)
    model_names = ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm']
    
    for model in model_names:
        # Look for PR-AUC pattern
        pr_pattern = rf'{model}.*?PR-AUC:\s*([\d.]+)'
        match = re.search(pr_pattern, content, re.IGNORECASE)
        if match:
            metrics['pr_auc'][model] = float(match.group(1))
        
        # Look for ROC-AUC pattern
        roc_pattern = rf'{model}.*?ROC-AUC:\s*([\d.]+)'
        match = re.search(roc_pattern, content, re.IGNORECASE)
        if match:
            metrics['roc_auc'][model] = float(match.group(1))
        
        # Look for F1 pattern
        f1_pattern = rf'{model}.*?F1-Score:\s*([\d.]+)'
        match = re.search(f1_pattern, content, re.IGNORECASE)
        if match:
            metrics['f1'][model] = float(match.group(1))
    
    return metrics


def print_comparison_table(logs: List[Dict]):
    """Print a comparison table of metrics across logs."""
    if not logs:
        print("No log files found.")
        return
    
    print("=" * 100)
    print("LOG FILE COMPARISON")
    print("=" * 100)
    
    # Header
    print(f"{'Timestamp':<20} {'Model':<25} {'PR-AUC':<10} {'ROC-AUC':<10} {'F1':<10}")
    print("-" * 100)
    
    for log in logs:
        timestamp = log['timestamp'] or 'Unknown'
        
        # Get all models present
        models = set(log['pr_auc'].keys()) | set(log['roc_auc'].keys()) | set(log['f1'].keys())
        
        for i, model in enumerate(sorted(models)):
            if i == 0:
                ts_display = timestamp
            else:
                ts_display = ''
            
            pr = log['pr_auc'].get(model, 0.0)
            roc = log['roc_auc'].get(model, 0.0)
            f1 = log['f1'].get(model, 0.0)
            
            print(f"{ts_display:<20} {model:<25} {pr:<10.4f} {roc:<10.4f} {f1:<10.4f}")
        
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare log files")
    parser.add_argument("--dir", default="outputs", 
                        help="Directory containing log files (default: outputs)")
    parser.add_argument("--pattern", default="*.log",
                        help="File pattern to match (default: *.log)")
    args = parser.parse_args()
    
    log_dir = Path(args.dir)
    
    if not log_dir.exists():
        print(f"Error: Directory '{log_dir}' does not exist.")
        sys.exit(1)
    
    # Find all log files
    log_files = sorted(log_dir.glob(args.pattern))
    
    if not log_files:
        print(f"No log files found matching '{args.pattern}' in '{log_dir}'")
        sys.exit(0)
    
    print(f"Found {len(log_files)} log file(s):")
    for lf in log_files:
        print(f"  - {lf.name}")
    print()
    
    # Extract metrics from each log
    logs = []
    for log_file in log_files:
        try:
            metrics = extract_metrics_from_log(log_file)
            # Only include if we found some metrics
            if metrics['pr_auc'] or metrics['roc_auc'] or metrics['f1']:
                logs.append(metrics)
        except Exception as e:
            print(f"Warning: Failed to parse {log_file.name}: {e}")
    
    # Print comparison table
    print_comparison_table(logs)
    
    print("\nNote: This is a simplified extraction. For detailed metrics,")
    print("      open the log files directly in a text editor.")


if __name__ == "__main__":
    main()
