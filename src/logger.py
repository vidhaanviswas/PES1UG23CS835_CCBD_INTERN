"""
Logging utilities for saving outputs to files.
"""

import sys
from pathlib import Path
from datetime import datetime


class TeeLogger:
    """Logs output to both console and file."""
    
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


def setup_logging(script_name: str, output_dir: str = "outputs") -> TeeLogger:
    """
    Set up logging to save outputs to a file.
    
    Parameters:
    -----------
    script_name : str
        Name of the script (used for filename)
    output_dir : str
        Directory to save log files
        
    Returns:
    --------
    TeeLogger
        Logger instance
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_path / f"{script_name}_{timestamp}.log"
    
    # Set up tee logger
    logger = TeeLogger(str(log_file))
    sys.stdout = logger
    
    print(f"Logging to: {log_file}")
    print("=" * 80)
    
    return logger


def close_logging(logger: TeeLogger):
    """Close the logger and restore stdout."""
    sys.stdout = logger.terminal
    logger.close()
