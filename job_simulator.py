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
    "scheduling_class",
    "priority",
    "cpu_request_mean",
    "cpu_request_std",
    "memory_request_mean",
    "memory_request_std",
    "disk_space_request_mean",
    "disk_space_request_std",
    "different_machine_constraint_mean",
]


@dataclass
class FeatureDistributions:
    num_tasks_values: np.ndarray
    scheduling_values: np.ndarray
    priority_values: np.ndarray
    cpu_mean_values: np.ndarray
    cpu_std_values: np.ndarray
    mem_mean_values: np.ndarray
    mem_std_values: np.ndarray
    disk_mean_values: np.ndarray
    disk_std_values: np.ndarray
    dmc_mean_values: np.ndarray


def _fallback_distributions(rng: np.random.Generator) -> FeatureDistributions:
    # Conservative defaults if processed file isn't present.
    num_tasks = rng.integers(1, 500, size=5000)
    scheduling = rng.integers(0, 4, size=5000)
    priority = rng.integers(0, 451, size=5000)
    cpu_mean = rng.uniform(0.1, 2.0, size=5000)
    cpu_std = rng.uniform(0.0, 0.5, size=5000)
    mem_mean = rng.uniform(0.1, 2.0, size=5000)
    mem_std = rng.uniform(0.0, 0.5, size=5000)
    disk_mean = rng.uniform(0.1, 2.0, size=5000)
    disk_std = rng.uniform(0.0, 0.5, size=5000)
    dmc_mean = rng.uniform(0.0, 1.0, size=5000)
    return FeatureDistributions(
        num_tasks,
        scheduling,
        priority,
        cpu_mean,
        cpu_std,
        mem_mean,
        mem_std,
        disk_mean,
        disk_std,
        dmc_mean,
    )


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
    df["cpu_request_std"] = df["cpu_request_std"].clip(lower=0)
    df["memory_request_std"] = df["memory_request_std"].clip(lower=0)
    df["disk_space_request_std"] = df["disk_space_request_std"].clip(lower=0)

    return FeatureDistributions(
        num_tasks_values=df["num_tasks"].to_numpy(),
        scheduling_values=df["scheduling_class"].to_numpy(),
        priority_values=df["priority"].to_numpy(),
        cpu_mean_values=df["cpu_request_mean"].to_numpy(),
        cpu_std_values=df["cpu_request_std"].to_numpy(),
        mem_mean_values=df["memory_request_mean"].to_numpy(),
        mem_std_values=df["memory_request_std"].to_numpy(),
        disk_mean_values=df["disk_space_request_mean"].to_numpy(),
        disk_std_values=df["disk_space_request_std"].to_numpy(),
        dmc_mean_values=df["different_machine_constraint_mean"].to_numpy(),
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
    scheduling = pick(dists.scheduling_values).astype(int)
    priority = pick(dists.priority_values).astype(float)
    cpu_mean = pick(dists.cpu_mean_values).astype(float)
    cpu_std = pick(dists.cpu_std_values).astype(float)
    mem_mean = pick(dists.mem_mean_values).astype(float)
    mem_std = pick(dists.mem_std_values).astype(float)
    disk_mean = pick(dists.disk_mean_values).astype(float)
    disk_std = pick(dists.disk_std_values).astype(float)
    dmc_mean = pick(dists.dmc_mean_values).astype(float)

    job_ids = [f"SIM_JOB_{start_job_id + i:06d}" for i in range(n)]
    out = pd.DataFrame(
        {
            "job_id": job_ids,
            "num_tasks": num_tasks,
            "scheduling_class": scheduling,
            "priority": priority,
            "cpu_request_mean": cpu_mean,
            "cpu_request_std": cpu_std,
            "memory_request_mean": mem_mean,
            "memory_request_std": mem_std,
            "disk_space_request_mean": disk_mean,
            "disk_space_request_std": disk_std,
            "different_machine_constraint_mean": dmc_mean,
        }
    )
    return out

