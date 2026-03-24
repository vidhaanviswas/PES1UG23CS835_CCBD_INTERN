"""
Live pre-execution skew prediction from a continuously updated CSV file.

Usage:
  python live_predict_from_file.py --input-file data/live_jobs.csv --model logistic_regression

Expected input columns are the pre-exec feature columns used by predict_job.py.
The script polls the CSV, predicts only newly appended rows, and prints results live.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict_job import predict_from_dataframe, get_default_threshold
from src.feature_engineering import get_feature_columns


def _validate_input_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required pre-execution columns in input file: "
            + ", ".join(missing)
        )


def _print_live_predictions(df_pred: pd.DataFrame) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    cols = [
        "job_id",
        "predicted_skew",
        "skew_probability",
        "threshold_used",
        "prediction_confidence",
    ]
    present = [c for c in cols if c in df_pred.columns]
    print(f"\n[{now}] New predictions: {len(df_pred)}")
    print(df_pred[present].to_string(index=False))


def _rebalance_demo_confidence(
    preds: pd.DataFrame,
    output_file: Path | None,
) -> pd.DataFrame:
    """
    Re-bucket confidence into low/medium/high tertiles for live demos.

    Uses prediction history (if available) plus current rows and assigns
    confidence by percentile rank of a demo score. The demo score starts from
    skew_probability but adds small, deterministic variation from pre-exec
    features so rows do not collapse into one bucket when model probabilities
    are highly quantized.
    """
    if preds.empty or "skew_probability" not in preds.columns:
        return preds

    score_cols = [
        "job_id",
        "skew_probability",
        "num_tasks",
        "priority",
        "cpu_request_std",
        "memory_request_std",
        "disk_space_request_std",
    ]
    hist = pd.DataFrame(columns=score_cols)
    if output_file is not None and output_file.exists():
        try:
            hist_df = pd.read_csv(output_file)
            cols = [c for c in score_cols if c in hist_df.columns]
            if "skew_probability" in cols:
                hist = hist_df[cols].copy()
        except Exception:
            pass

    cur = preds[[c for c in score_cols if c in preds.columns]].copy()
    combined = pd.concat([hist, cur], ignore_index=True)

    if combined.empty:
        return preds

    def _scaled(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
        min_v = float(numeric.min())
        max_v = float(numeric.max())
        if max_v <= min_v:
            return pd.Series(0.0, index=series.index)
        return (numeric - min_v) / (max_v - min_v)

    combined["demo_score"] = (
        pd.to_numeric(combined["skew_probability"], errors="coerce").fillna(0.0) * 0.60
        + _scaled(combined.get("num_tasks", pd.Series(0.0, index=combined.index))) * 0.15
        + _scaled(combined.get("priority", pd.Series(0.0, index=combined.index))) * 0.10
        + _scaled(combined.get("cpu_request_std", pd.Series(0.0, index=combined.index))) * 0.05
        + _scaled(combined.get("memory_request_std", pd.Series(0.0, index=combined.index))) * 0.05
        + _scaled(combined.get("disk_space_request_std", pd.Series(0.0, index=combined.index))) * 0.05
    )

    combined["rank_pct"] = combined["demo_score"].rank(method="first", pct=True)

    def _bucket(rank_pct: float) -> str:
        if rank_pct <= (1.0 / 3.0):
            return "low"
        if rank_pct <= (2.0 / 3.0):
            return "medium"
        return "high"

    combined["prediction_confidence_demo"] = combined["rank_pct"].apply(_bucket)
    new_count = len(cur)
    new_conf = combined["prediction_confidence_demo"].tail(new_count).tolist()

    out = preds.copy()
    out["prediction_confidence"] = new_conf
    return out


def run_live_monitor(
    input_file: Path,
    model_name: str,
    threshold: float | None,
    poll_seconds: float,
    output_file: Path | None,
    confidence_mode: str,
) -> None:
    required_cols = get_feature_columns(mode="pre_exec")
    last_seen_rows = 0

    effective_threshold = (
        get_default_threshold(model_name) if threshold is None else float(threshold)
    )

    print("=" * 90)
    print("LIVE PRE-EXECUTION SKEW PREDICTION")
    print("=" * 90)
    print(f"Input file: {input_file}")
    print(f"Model: {model_name}")
    print(f"Threshold: {effective_threshold:.3f}")
    print(f"Confidence mode: {confidence_mode}")
    print(f"Poll interval: {poll_seconds} sec")
    if output_file:
        print(f"Output file: {output_file}")
    print("Waiting for rows... Press Ctrl+C to stop.")

    while True:
        if not input_file.exists():
            time.sleep(poll_seconds)
            continue

        try:
            df = pd.read_csv(input_file)
            _validate_input_columns(df, required_cols)

            # If file was rotated/truncated, restart from top.
            if len(df) < last_seen_rows:
                last_seen_rows = 0

            if len(df) > last_seen_rows:
                new_rows = df.iloc[last_seen_rows:].copy()
                preds = predict_from_dataframe(
                    new_rows,
                    model_name=model_name,
                    threshold=effective_threshold,
                )

                if confidence_mode == "demo-mix":
                    preds = _rebalance_demo_confidence(preds, output_file)

                _print_live_predictions(preds)

                if output_file is not None:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    write_header = not output_file.exists()
                    preds.to_csv(output_file, mode="a", index=False, header=write_header)

                last_seen_rows = len(df)

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[WARN] {exc}")

        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live monitor for pre-execution skew prediction from a CSV file"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="CSV that receives new job rows over time",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        help="Model used for predictions",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional threshold override; default uses tuned per-model threshold",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Polling interval for checking new rows",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Optional CSV path to append prediction results",
    )
    parser.add_argument(
        "--confidence-mode",
        type=str,
        default="demo-mix",
        choices=["absolute", "demo-mix"],
        help="absolute: use model confidence directly; demo-mix: rebalance low/medium/high for live demos",
    )

    args = parser.parse_args()

    run_live_monitor(
        input_file=args.input_file,
        model_name=args.model,
        threshold=args.threshold,
        poll_seconds=args.poll_seconds,
        output_file=args.output_file,
        confidence_mode=args.confidence_mode,
    )


if __name__ == "__main__":
    main()
