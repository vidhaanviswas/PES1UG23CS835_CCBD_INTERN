"""
Example: Using Output Logging

This script demonstrates how to use the --save-log feature
to save terminal output to files for later review.
"""

print("=" * 80)
print("OUTPUT LOGGING EXAMPLES")
print("=" * 80)

examples = [
    {
        "command": "python main.py --save-log",
        "description": "Run main pipeline and save all output to outputs/main_TIMESTAMP.log"
    },
    {
        "command": "python main_sample.py --save-log",
        "description": "Run sample pipeline and save output to outputs/main_sample_TIMESTAMP.log"
    },
    {
        "command": "python early_exec_experiment.py --save-log",
        "description": "Run early-execution experiment and save results to outputs/early_exec_experiment_TIMESTAMP.log"
    },
    {
        "command": "python explainability_report.py --save-log",
        "description": "Generate feature importance report and save to outputs/explainability_report_TIMESTAMP.log"
    },
    {
        "command": "python mitigation_simulation.py --save-log",
        "description": "Run policy simulation and save results to outputs/mitigation_simulation_TIMESTAMP.log"
    }
]

for i, ex in enumerate(examples, 1):
    print(f"\n{i}. {ex['description']}")
    print(f"   Command: {ex['command']}")

print("\n" + "=" * 80)
print("BENEFITS OF LOGGING")
print("=" * 80)
print("✓ Terminal output is hard to scroll through and review")
print("✓ Log files preserve complete output for later analysis")
print("✓ Easy to compare results across different runs")
print("✓ Can search/grep through logs for specific metrics")
print("✓ Timestamped filenames prevent accidental overwrites")
print("✓ Great for research paper preparation and result tables")

print("\n" + "=" * 80)
print("LOG FILE LOCATIONS")
print("=" * 80)
print("All log files are saved to: outputs/")
print("Format: outputs/<script_name>_<timestamp>.log")
print("Example: outputs/main_20240115_143052.log")

print("\n" + "=" * 80)
print("USAGE TIPS")
print("=" * 80)
print("1. Run experiments with --save-log flag")
print("2. Review log files at your leisure")
print("3. Extract key metrics using text editors or grep")
print("4. Keep logs for reproducibility and documentation")
print("5. Compare outputs from different configurations")

print("\n" + "=" * 80)
