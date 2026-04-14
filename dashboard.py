"""Streamlit dashboard for LLM Eval Harness results.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_result_file(path: Path) -> dict:
    return json.loads(path.read_text())


def get_result_files() -> list:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)


def main() -> None:
    st.set_page_config(page_title="LLM Eval Harness", layout="wide")
    st.title("LLM Eval Harness Dashboard")

    result_files = get_result_files()
    if not result_files:
        st.warning("No results found in /results directory. Run an evaluation first.")
        return

    # ── Sidebar: file selection ──────────────────────────────────────────
    st.sidebar.header("Select Results")
    file_names = [f.name for f in result_files]
    selected_file = st.sidebar.selectbox("Result file", file_names)
    data = load_result_file(RESULTS_DIR / selected_file)

    # Optional baseline for comparison
    compare_file = st.sidebar.selectbox(
        "Compare with baseline (optional)", ["None"] + file_names
    )

    # ── Summary metrics ──────────────────────────────────────────────────
    st.header("Summary")
    meta = data.get("metadata", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Model", meta.get("model", "N/A"))
    col2.metric("Questions", meta.get("total_questions", 0))
    col3.metric("Timestamp", meta.get("timestamp", "N/A")[:19])

    summary = data.get("summary", {})
    score_metrics = {
        k: v for k, v in summary.items()
        if k not in ("avg_latency_ms", "total_input_tokens", "total_output_tokens")
    }

    if score_metrics:
        cols = st.columns(len(score_metrics))
        for i, (metric_name, value) in enumerate(sorted(score_metrics.items())):
            delta = None
            if compare_file != "None":
                baseline_data = load_result_file(RESULTS_DIR / compare_file)
                baseline_val = baseline_data.get("summary", {}).get(metric_name)
                if baseline_val is not None:
                    delta = round(value - baseline_val, 4)
            cols[i].metric(metric_name, f"{value:.4f}", delta=delta)

    # Latency and token stats
    perf_cols = st.columns(3)
    perf_cols[0].metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f}ms")
    perf_cols[1].metric("Total Input Tokens", f"{summary.get('total_input_tokens', 0):,}")
    perf_cols[2].metric("Total Output Tokens", f"{summary.get('total_output_tokens', 0):,}")

    # ── Score distribution chart ─────────────────────────────────────────
    results = data.get("results", [])
    if results and results[0].get("scores"):
        st.header("Score Distribution")

        rows = []
        for r in results:
            for metric_name, score in r.get("scores", {}).items():
                rows.append({
                    "Question ID": r["id"],
                    "Metric": metric_name,
                    "Score": score,
                })

        if rows:
            df_scores = pd.DataFrame(rows)

            # Bar chart: average by metric
            avg_by_metric = df_scores.groupby("Metric")["Score"].mean().reset_index()
            st.bar_chart(avg_by_metric.set_index("Metric"))

            # Per-question heatmap table
            st.subheader("Per-Question Scores")
            pivot = df_scores.pivot(index="Question ID", columns="Metric", values="Score")
            st.dataframe(
                pivot.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1),
                use_container_width=True,
            )

    # ── Latency chart ────────────────────────────────────────────────────
    if results:
        st.header("Latency by Question")
        lat_df = pd.DataFrame([
            {"Question ID": r["id"], "Latency (ms)": r.get("latency_ms", 0)}
            for r in results
        ])
        st.bar_chart(lat_df.set_index("Question ID"))

    # ── Token usage chart ────────────────────────────────────────────────
    if results:
        st.header("Token Usage")
        token_df = pd.DataFrame([
            {
                "Question ID": r["id"],
                "Input Tokens": r.get("input_tokens", 0),
                "Output Tokens": r.get("output_tokens", 0),
            }
            for r in results
        ])
        st.bar_chart(token_df.set_index("Question ID"))

    # ── Detailed results table ───────────────────────────────────────────
    st.header("Detailed Results")
    for r in results:
        with st.expander(f"[{r['id']}] {r.get('question', '')[:80]}"):
            st.markdown(f"**Expected:** {r.get('expected_answer', '')}")
            st.markdown(f"**Actual:** {r.get('actual_answer', '')}")
            st.markdown(f"**Latency:** {r.get('latency_ms', 0):.0f}ms")
            if r.get("scores"):
                st.json(r["scores"])

    # ── Model comparison ─────────────────────────────────────────────────
    if compare_file != "None":
        st.header("Model Comparison")
        baseline_data = load_result_file(RESULTS_DIR / compare_file)

        comp_rows = []
        for metric_name in sorted(score_metrics.keys()):
            curr = summary.get(metric_name)
            base = baseline_data.get("summary", {}).get(metric_name)
            if curr is not None and base is not None:
                comp_rows.append({
                    "Metric": metric_name,
                    f"Current ({meta.get('model', '')})": round(curr, 4),
                    f"Baseline ({baseline_data.get('metadata', {}).get('model', '')})": round(base, 4),
                    "Delta": round(curr - base, 4),
                })

        if comp_rows:
            st.dataframe(pd.DataFrame(comp_rows), use_container_width=True)


if __name__ == "__main__":
    main()
