"""
Live Job Simulation + Skew Prediction Dashboard (Streamlit)

Runs locally. Simulates incoming jobs, predicts skew risk using the trained ML model,
and displays results live.

Run:
  streamlit run dashboard_app.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from job_simulator import load_distributions, simulate_jobs
from predict_job import predict_from_dataframe


st.set_page_config(page_title="Data Skew Prediction Dashboard", layout="wide")


@dataclass(frozen=True)
class RiskBand:
    label: str
    help: str


def _risk_band(prob: float, threshold: float) -> RiskBand:
    """
    Convert a probability into a plain-English risk band.

    We keep the wording non-technical so the dashboard is easy to understand.
    """
    # Below this: clearly low risk.
    low_cutoff = max(0.0, threshold * 0.5)
    # Near the threshold: "watchlist".
    watch_cutoff = max(0.0, threshold * 0.85)

    if prob < low_cutoff:
        return RiskBand(
            label="Low risk",
            help="Unlikely to be skewed based on patterns learned from historical data.",
        )
    if prob < watch_cutoff:
        return RiskBand(
            label="Moderate risk",
            help="Some warning signs; monitor if resources are tight or deadlines are strict.",
        )
    if prob < threshold:
        return RiskBand(
            label="Watchlist",
            help="Close to the alert threshold; consider a quick check before running.",
        )
    return RiskBand(
        label="High risk",
        help="Likely to be skewed; recommended to review/mitigate before execution.",
    )


def _fmt_duration(x: object) -> str:
    """
    Human-friendly duration string.

    Note: in your processed dataset, task runtimes may be in microseconds (as seen in sample traces).
    We show both raw and a scaled view to avoid confusion.
    """
    try:
        v = float(x)
    except Exception:
        return "-"
    if v < 0:
        return "-"

    # Heuristic: if very large, treat as microseconds and convert to seconds.
    # (In the user's sample run we saw values around 1e8..3e8 which align with microseconds.)
    if v >= 1e6:
        secs = v / 1e6
        if secs < 60:
            return f"{secs:,.2f} s"
        mins = secs / 60.0
        if mins < 60:
            return f"{mins:,.2f} min"
        hrs = mins / 60.0
        return f"{hrs:,.2f} hr"

    # Otherwise treat as seconds.
    if v < 60:
        return f"{v:,.2f} s"
    mins = v / 60.0
    if mins < 60:
        return f"{mins:,.2f} min"
    hrs = mins / 60.0
    return f"{hrs:,.2f} hr"


def _init_state() -> None:
    if "jobs" not in st.session_state:
        st.session_state.jobs = pd.DataFrame()
    if "sim_job_counter" not in st.session_state:
        st.session_state.sim_job_counter = 1
    if "last_tick" not in st.session_state:
        st.session_state.last_tick = time.time()
    if "paused" not in st.session_state:
        st.session_state.paused = False
    if "selected_job_id" not in st.session_state:
        st.session_state.selected_job_id = None


def main() -> None:
    _init_state()

    st.title("Early Prediction of Data Skew — Live Simulation Dashboard")
    st.caption("A simple, self-explanatory demo for research presentations (no live cluster needed).")

    # Quick explainer for non-technical audiences.
    with st.expander("What is “data skew” and why should we care?", expanded=True):
        st.markdown(
            """
**Data skew** happens when a job’s work is not evenly split across its tasks.
If one task takes *much longer* than the others, the whole job can slow down.

This dashboard simulates incoming jobs, estimates their **risk of skew before execution**,
and highlights jobs that might need attention.

**Skew definition used in this research:** a job is labeled *skewed* if its slowest task runtime is at least **2×** the average task runtime.
            """.strip()
        )

    with st.sidebar:
        st.header("Controls")
        st.write("Use these settings to control the demo. The defaults work well for presentations.")
        model_name = st.selectbox(
            "Model used for prediction",
            options=["logistic_regression", "random_forest"],
            index=0,
            help="Both models were trained on historical (job-level) features derived from Google Cluster traces.",
        )
        jobs_per_tick = st.slider(
            "Jobs per refresh",
            min_value=1,
            max_value=200,
            value=25,
            step=1,
            help="How many new simulated jobs arrive each refresh cycle.",
        )
        refresh_ms = st.slider(
            "Refresh interval (ms)",
            min_value=250,
            max_value=5000,
            value=1000,
            step=250,
            help="How often the dashboard generates new simulated jobs.",
        )
        max_rows = st.slider(
            "Keep last N jobs",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Keeps the dashboard fast by showing only the most recent jobs.",
        )
        skew_threshold = st.slider(
            "Alert threshold (probability)",
            min_value=0.05,
            max_value=0.95,
            value=0.70,
            step=0.05,
            help="Jobs above this probability are flagged as “High risk” for skew.",
        )
        st.session_state.paused = st.toggle(
            "Pause simulation",
            value=bool(st.session_state.paused),
            help="Pause stops new jobs from being generated so you can explain the screen.",
        )
        st.divider()
        if st.button("Reset stream", type="primary"):
            st.session_state.jobs = pd.DataFrame()
            st.session_state.sim_job_counter = 1
            st.session_state.last_tick = time.time()
            st.session_state.selected_job_id = None
            st.rerun()

    # Auto-refresh (disabled when paused)
    # Streamlit deprecated `st.experimental_set_query_params` in favor of `st.query_params`.
    # We don't need to set anything here; accessing it is enough to avoid older query-param issues.
    _ = st.query_params
    if not st.session_state.paused:
        try:
            _ = st.autorefresh(interval=refresh_ms, key="autorefresh")
        except Exception:
            pass

    # Tabs: presentable narrative flow.
    tab_overview, tab_stream, tab_how = st.tabs(["Overview", "Live stream", "How it works"])

    # Load distributions from processed file if exists (more realistic)
    processed_path = Path("data/processed/job_level_data.csv")
    dists = load_distributions(str(processed_path), seed=42)

    # Generate/predict only when not paused.
    if not st.session_state.paused:
        start_id = int(st.session_state.sim_job_counter)
        new_jobs = simulate_jobs(jobs_per_tick, dists=dists, start_job_id=start_id)
        st.session_state.sim_job_counter = start_id + jobs_per_tick

        preds = predict_from_dataframe(new_jobs, model_name=model_name)
        preds["flagged"] = (preds["skew_probability"] >= skew_threshold).astype(int)
        preds["risk_band"] = preds["skew_probability"].apply(lambda p: _risk_band(float(p), skew_threshold).label)
        preds["ingest_time"] = pd.Timestamp.utcnow()

        if st.session_state.jobs.empty:
            st.session_state.jobs = preds
        else:
            st.session_state.jobs = pd.concat([st.session_state.jobs, preds], ignore_index=True)
        if len(st.session_state.jobs) > max_rows:
            st.session_state.jobs = st.session_state.jobs.iloc[-max_rows:].reset_index(drop=True)

    df_all = st.session_state.jobs.copy()

    # KPI row
    total = len(df_all)
    flagged = int(df_all["flagged"].sum()) if total else 0
    avg_prob = float(df_all["skew_probability"].mean()) if total else 0.0
    high_conf = int((df_all["prediction_confidence"] == "high").sum()) if total else 0

    with tab_overview:
        st.subheader("At a glance")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Jobs in view", f"{total:,}", help="How many recent jobs are shown in the dashboard window.")
        c2.metric(
            f"Alerts (p ≥ {skew_threshold:.2f})",
            f"{flagged:,}",
            help="Jobs above the alert threshold. These are the ones you would review first.",
        )
        c3.metric("Average risk", f"{avg_prob:.2%}", help="Average skew probability across jobs in view.")
        c4.metric(
            "High-confidence",
            f"{high_conf:,}",
            help="Count of predictions labeled high-confidence by the prediction module.",
        )

        st.divider()
        st.subheader("What should a non-technical viewer look for?")
        st.markdown(
            """
- **Alerts**: jobs that are likely to suffer from skew (slowest task dominates).
- **Risk band**: simple label (Low / Moderate / Watchlist / High) to make decisions quick.
- **Top risky jobs**: the “worst” jobs first so the team can act immediately.
            """.strip()
        )

        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Risk distribution (recent jobs)")
            if total:
                st.bar_chart(df_all["skew_probability"].value_counts(bins=20).sort_index())
            else:
                st.info("No jobs yet. Start the simulation to generate jobs.")
        with right:
            st.subheader("Alerts vs no alerts")
            if total:
                st.bar_chart(df_all["flagged"].value_counts().rename(index={0: "not_alert", 1: "alert"}))
            else:
                st.info("No jobs yet.")

    with tab_stream:
        st.subheader("Live job stream (latest first)")
        st.caption("Tip: pause the simulation (left sidebar) to explain a specific job.")

        show_cols = [
            "job_id",
            "risk_band",
            "skew_probability",
            "flagged",
            "prediction_confidence",
            "num_tasks",
            "avg_task_runtime",
            "max_task_runtime",
            "std_task_runtime",
            "scheduling_class",
            "priority",
        ]

        if total:
            df_view = df_all.sort_values("ingest_time", ascending=False).copy()
            # Add human-friendly runtime columns (keeps the raw values too for research).
            df_view["avg_task_runtime_readable"] = df_view["avg_task_runtime"].map(_fmt_duration)
            df_view["max_task_runtime_readable"] = df_view["max_task_runtime"].map(_fmt_duration)
            df_view["std_task_runtime_readable"] = df_view["std_task_runtime"].map(_fmt_duration)

            # Put readable versions next to raw columns for clarity.
            df_display = df_view.copy()
            df_display.insert(df_display.columns.get_loc("avg_task_runtime") + 1, "avg_rt", df_view["avg_task_runtime_readable"])
            df_display.insert(df_display.columns.get_loc("max_task_runtime") + 1, "max_rt", df_view["max_task_runtime_readable"])
            df_display.insert(df_display.columns.get_loc("std_task_runtime") + 1, "std_rt", df_view["std_task_runtime_readable"])

            show_cols_pretty = [
                "job_id",
                "risk_band",
                "skew_probability",
                "flagged",
                "prediction_confidence",
                "num_tasks",
                "avg_rt",
                "max_rt",
                "std_rt",
                "scheduling_class",
                "priority",
            ]
            st.dataframe(df_display[show_cols_pretty], use_container_width=True, height=460)

            st.divider()
            st.subheader("Explain one job (pick from the latest jobs)")
            latest_ids = df_view["job_id"].head(200).tolist()
            selected = st.selectbox("Select a job_id", options=latest_ids, index=0)
            st.session_state.selected_job_id = selected

            row = df_view[df_view["job_id"] == selected].iloc[0].to_dict()
            rb = _risk_band(float(row["skew_probability"]), skew_threshold)

            c1, c2, c3 = st.columns(3)
            c1.metric("Risk band", rb.label, help=rb.help)
            c2.metric("Skew probability", f'{float(row["skew_probability"]):.2%}')
            c3.metric("Alert", "Yes" if int(row["flagged"]) == 1 else "No")

            with st.expander("Why did the model say this? (plain English)", expanded=True):
                st.markdown(
                    f"""
The model looks at a job’s **task runtime pattern** and **job metadata**.
The strongest warning sign in this research is usually a **high variation** in task runtimes (**std_task_runtime**).

For this job:
- **Number of tasks**: {int(row.get("num_tasks", 0))}
- **Average task runtime**: {_fmt_duration(row.get("avg_task_runtime"))}
- **Max task runtime**: {_fmt_duration(row.get("max_task_runtime"))}
- **Runtime variation (std)**: {_fmt_duration(row.get("std_task_runtime"))}

If the max runtime is much larger than the typical runtime (or variation is very high), the job is more likely to be skewed.
                    """.strip()
                )

        else:
            st.info("No jobs yet. Unpause the simulation to start generating jobs.")

    with tab_how:
        st.subheader("How this demo maps to the real world")
        st.markdown(
            """
### What is simulated?
- Incoming jobs are **synthetic** (generated locally).
- Their feature distributions are sampled from your processed dataset when available:
  `data/processed/job_level_data.csv`.

### What is real?
- The **trained model** is real and loaded from:
  `models/trained_models.pkl`.
- Predictions are produced by the same code you would use in production (`predict_job.py`).

### In a production system, replace the simulator with:
- Job metadata from your scheduler / workflow tool (Airflow, Kubernetes, YARN, etc.)
- Pre-execution estimates from:
  - historical runs (feature store),
  - job DAG/task count,
  - input size estimates,
  - sampling/profiling.

### Why the threshold slider exists
No model is perfect. The threshold lets you choose the trade-off:
- Lower threshold: catch more skewed jobs (higher recall), but more false alarms.
- Higher threshold: fewer false alarms (higher precision), but you may miss some skew.
            """.strip()
        )

        st.divider()
        st.subheader("Data sources used by the dashboard")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Processed dataset**")
            st.write(str(processed_path.resolve()))
            st.write("Status:", "Found" if processed_path.exists() else "Not found (fallback simulation used)")
        with c2:
            model_path = Path("models/trained_models.pkl")
            st.write("**Trained model file**")
            st.write(str(model_path.resolve()))
            st.write("Status:", "Found" if model_path.exists() else "Not found (run training first)")

    st.caption(
        "Note: This is a local simulation for research demos. In production, use real job metadata + estimated pre-execution features."
    )


if __name__ == "__main__":
    main()

