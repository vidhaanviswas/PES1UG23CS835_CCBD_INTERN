"""
Data split utilities for leakage-aware evaluation.
"""

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def add_template_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "user_name" in out.columns:
        out["template_id"] = out["user_name"].astype(str)
        return out

    # Fallback: bucket by scheduling_class, priority, and num_tasks.
    sched = out.get("scheduling_class", 0)
    priority = out.get("priority", 0)
    num_tasks = out.get("num_tasks", 0)

    out["priority_bucket"] = pd.qcut(pd.Series(priority), q=5, duplicates="drop")
    out["num_tasks_bucket"] = pd.qcut(pd.Series(num_tasks), q=5, duplicates="drop")

    out["template_id"] = (
        sched.astype(str)
        + "|"
        + out["priority_bucket"].astype(str)
        + "|"
        + out["num_tasks_bucket"].astype(str)
    )
    return out


def time_based_split(df: pd.DataFrame, time_col: str = "submit_time", test_size: float = 0.2):
    if time_col not in df.columns or df[time_col].isna().all():
        df_sorted = df.reset_index(drop=True)
    else:
        df_sorted = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train_df = df_sorted.iloc[:split_idx].copy()
    test_df = df_sorted.iloc[split_idx:].copy()
    return train_df, test_df


def template_based_split(df: pd.DataFrame, template_col: str = "template_id",
                        test_size: float = 0.2, random_state: int = 42):
    if template_col not in df.columns:
        raise ValueError(f"Template column '{template_col}' not found for group split")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    groups = df[template_col]
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
