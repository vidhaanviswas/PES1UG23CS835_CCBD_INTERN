"""
Test script to verify logging functionality works correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from logger import setup_logging, close_logging

def test_logging():
    """Test the logging functionality."""
    
    # Test with logging
    print("\n" + "=" * 80)
    print("TESTING LOGGING FUNCTIONALITY")
    print("=" * 80)
    
    logger = setup_logging("test")
    
    print("\n1. This output should appear in both terminal and log file")
    print("2. Testing multi-line output:")
    print("   - Line 1")
    print("   - Line 2")
    print("   - Line 3")
    
    print("\n3. Testing formatted output:")
    metrics = {
        'PR-AUC': 0.8567,
        'ROC-AUC': 0.9234,
        'F1-Score': 0.7891,
        'Precision': 0.8123,
        'Recall': 0.7654
    }
    
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")
    
    print("\n4. Testing error handling:")
    print("   This is a warning message")
    print("   This is an error message")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print("\nCheck the outputs/ directory for the log file.")
    
    close_logging(logger)
    print("\nThis line should only appear in terminal (after logger closed).")


if __name__ == "__main__":
    test_logging()
