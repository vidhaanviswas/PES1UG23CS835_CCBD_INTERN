"""
Append synthetic pre-execution job metadata rows to a CSV file for live demos.

Usage:
  python generate_live_jobs.py --output-file data/live_jobs.csv --rows 50 --interval 1.5
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import pandas as pd


def _make_row(job_id: int) -> dict:
    # Values are sampled to resemble plausible pre-execution metadata.
    num_tasks = random.randint(5, 300)
    scheduling_class = random.choice([0, 1, 2, 3])
    priority = random.uniform(1, 450)

    cpu_mean = random.uniform(0.1, 2.5)
    mem_mean = random.uniform(0.1, 3.0)
    disk_mean = random.uniform(0.05, 2.0)

    return {
        "job_id": f"LIVE_{job_id:06d}",
        "num_tasks": num_tasks,
        "scheduling_class": scheduling_class,
        "priority": priority,
        "cpu_request_mean": cpu_mean,
        "cpu_request_std": random.uniform(0.01, cpu_mean * 0.5),
        "memory_request_mean": mem_mean,
        "memory_request_std": random.uniform(0.01, mem_mean * 0.5),
        "disk_space_request_mean": disk_mean,
        "disk_space_request_std": random.uniform(0.005, disk_mean * 0.5),
        "different_machine_constraint_mean": random.choice([0.0, 0.1, 0.2, 0.3, 0.5]),
    }


def run_generator(output_file: Path, rows: int, interval: float, start_id: int) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing live rows to: {output_file}")

    job_id = start_id
    for _ in range(rows):
        row = _make_row(job_id)
        df = pd.DataFrame([row])

        write_header = not output_file.exists()
        df.to_csv(output_file, mode="a", index=False, header=write_header)

        print(f"Appended {row['job_id']}")
        job_id += 1
        time.sleep(interval)

    print("Done generating live rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate live demo input rows")
    parser.add_argument("--output-file", type=Path, required=True, help="CSV destination")
    parser.add_argument("--rows", type=int, default=30, help="Rows to append")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between rows")
    parser.add_argument("--start-id", type=int, default=1, help="Starting numeric job id")
    args = parser.parse_args()

    run_generator(
        output_file=args.output_file,
        rows=args.rows,
        interval=args.interval,
        start_id=args.start_id,
    )


if __name__ == "__main__":
    main()
