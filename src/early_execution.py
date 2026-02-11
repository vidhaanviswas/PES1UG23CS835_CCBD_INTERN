"""
Early-Execution Feature Module

Builds early-execution features from the first k% of tasks in each job.
"""

import math
import pandas as pd


def compute_early_runtime_features(
    task_runtimes: pd.DataFrame,
    k: float = 0.1,
    min_tasks: int = 5,
) -> pd.DataFrame:
    """
    Compute early-execution runtime features from the first k% tasks.

    Parameters:
    -----------
    task_runtimes : pd.DataFrame
        Dataframe with job_id, runtime, and a time column (end_time or start_time).
    k : float
        Fraction of tasks to consider (0 < k <= 1).
    min_tasks : int
        Minimum tasks per job to include in early slice.

    Returns:
    --------
    pd.DataFrame
        Job-level early runtime features.
    """
    if not (0 < k <= 1):
        raise ValueError("k must be in (0, 1].")

    required = {"job_id", "runtime"}
    missing = required - set(task_runtimes.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = task_runtimes.copy()
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df = df[df["runtime"].notna()]
    df = df[df["runtime"] > 0]

    time_col = None
    if "end_time" in df.columns:
        time_col = "end_time"
    elif "start_time" in df.columns:
        time_col = "start_time"
    elif "timestamp" in df.columns:
        time_col = "timestamp"

    if time_col:
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")

    features = []
    for job_id, group in df.groupby("job_id"):
        if time_col:
            group = group.sort_values(time_col)
        elif "task_index" in group.columns:
            group = group.sort_values("task_index")

        n = len(group)
        if n == 0:
            continue
        take = max(min_tasks, int(math.ceil(k * n)))
        take = min(take, n)
        early = group.iloc[:take]

        features.append(
            {
                "job_id": job_id,
                "early_num_tasks": len(early),
                "early_avg_task_runtime": early["runtime"].mean(),
                "early_max_task_runtime": early["runtime"].max(),
                "early_std_task_runtime": early["runtime"].std(ddof=0) if len(early) > 1 else 0.0,
            }
        )

    return pd.DataFrame(features)
