"""
Job Stream Simulator

Generates synthetic "incoming jobs" for a live dashboard demo.
It samples realistic feature distributions from `data/processed/job_level_data.csv`
if available; otherwise falls back to reasonable defaults.

This is meant for *simulation/demonstration* (no live traffic required).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


FEATURE_COLS = [
    "num_tasks",
    "avg_task_runtime",
    "max_task_runtime",
    "std_task_runtime",
    "scheduling_class",
    "priority",
]


@dataclass
class FeatureDistributions:
    num_tasks_values: np.ndarray
    avg_values: np.ndarray
    max_values: np.ndarray
    std_values: np.ndarray
    scheduling_values: np.ndarray
    priority_values: np.ndarray


def _fallback_distributions(rng: np.random.Generator) -> FeatureDistributions:
    # Conservative defaults if processed file isn't present.
    num_tasks = rng.integers(1, 500, size=5000)
    avg_rt = rng.uniform(1e6, 3e8, size=5000)
    std_rt = rng.uniform(0, 5e7, size=5000)
    # Keep max >= avg typically, sometimes much larger to mimic skew.
    max_rt = avg_rt * rng.uniform(1.0, 2.8, size=5000)
    scheduling = rng.integers(0, 4, size=5000)
    priority = rng.integers(0, 451, size=5000)
    return FeatureDistributions(num_tasks, avg_rt, max_rt, std_rt, scheduling, priority)


def load_distributions(
    processed_csv_path: str = "data/processed/job_level_data.csv",
    seed: int = 42,
) -> FeatureDistributions:
    """
    Load empirical feature distributions from processed data.
    """
    rng = np.random.default_rng(seed)
    path = Path(processed_csv_path)
    if not path.exists():
        return _fallback_distributions(rng)

    df = pd.read_csv(path)
    # If file doesn't contain expected columns, fallback.
    if any(c not in df.columns for c in FEATURE_COLS):
        return _fallback_distributions(rng)

    # Clean NaNs and coerce numeric.
    df = df.copy()
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS)
    if df.empty:
        return _fallback_distributions(rng)

    # Clip extreme negatives just in case.
    df["num_tasks"] = df["num_tasks"].clip(lower=1).astype(int)
    df["std_task_runtime"] = df["std_task_runtime"].clip(lower=0)
    df["max_task_runtime"] = df["max_task_runtime"].clip(lower=0)
    df["avg_task_runtime"] = df["avg_task_runtime"].clip(lower=0)

    return FeatureDistributions(
        num_tasks_values=df["num_tasks"].to_numpy(),
        avg_values=df["avg_task_runtime"].to_numpy(),
        max_values=df["max_task_runtime"].to_numpy(),
        std_values=df["std_task_runtime"].to_numpy(),
        scheduling_values=df["scheduling_class"].to_numpy(),
        priority_values=df["priority"].to_numpy(),
    )


def simulate_jobs(
    n: int,
    dists: FeatureDistributions,
    rng: Optional[np.random.Generator] = None,
    start_job_id: int = 1,
) -> pd.DataFrame:
    """
    Generate N synthetic jobs with realistic feature patterns.

    Returns a dataframe with columns:
    - job_id (string)
    - FEATURE_COLS...
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample with replacement from empirical values for realism.
    def pick(values: np.ndarray) -> np.ndarray:
        idx = rng.integers(0, len(values), size=n)
        return values[idx]

    num_tasks = pick(dists.num_tasks_values).astype(int)
    avg_rt = pick(dists.avg_values).astype(float)
    std_rt = pick(dists.std_values).astype(float)
    scheduling = pick(dists.scheduling_values).astype(int)
    priority = pick(dists.priority_values).astype(float)

    # Generate max runtime: loosely coupled to avg and std, with occasional spikes.
    # This makes the stream include some likely-skew patterns.
    base_ratio = rng.uniform(1.0, 1.4, size=n)
    spike = rng.random(n) < 0.08  # 8% spiky jobs
    base_ratio[spike] = rng.uniform(1.6, 3.0, size=spike.sum())
    max_rt = np.maximum(avg_rt * base_ratio, avg_rt + 2.0 * std_rt)

    job_ids = [f"SIM_JOB_{start_job_id + i:06d}" for i in range(n)]
    out = pd.DataFrame(
        {
            "job_id": job_ids,
            "num_tasks": num_tasks,
            "avg_task_runtime": avg_rt,
            "max_task_runtime": max_rt,
            "std_task_runtime": std_rt,
            "scheduling_class": scheduling,
            "priority": priority,
        }
    )
    return out

