from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def compare_results(
    current_path: str,
    baseline_path: str,
    threshold: float = 0.05,
) -> Dict:
    """Compare current eval results against a baseline and flag regressions.

    A regression is flagged when a metric drops by more than `threshold`
    (default 5%) compared to the baseline.
    """
    current = json.loads(Path(current_path).read_text())
    baseline = json.loads(Path(baseline_path).read_text())

    regressions: List[Dict] = []
    improvements: List[Dict] = []
    unchanged: List[str] = []

    baseline_summary = baseline.get("summary", {})
    current_summary = current.get("summary", {})

    # Compare summary-level metrics
    all_metrics = set(baseline_summary.keys()) | set(current_summary.keys())
    score_metrics = [m for m in all_metrics if m not in (
        "avg_latency_ms", "total_input_tokens", "total_output_tokens",
    )]

    for metric in sorted(score_metrics):
        base_val = baseline_summary.get(metric)
        curr_val = current_summary.get(metric)

        if base_val is None or curr_val is None:
            continue

        delta = curr_val - base_val

        if delta < -threshold:
            regressions.append({
                "metric": metric,
                "baseline": round(base_val, 4),
                "current": round(curr_val, 4),
                "delta": round(delta, 4),
            })
        elif delta > threshold:
            improvements.append({
                "metric": metric,
                "baseline": round(base_val, 4),
                "current": round(curr_val, 4),
                "delta": round(delta, 4),
            })
        else:
            unchanged.append(metric)

    # Per-question comparison
    per_question: List[Dict] = []
    baseline_by_id = {r["id"]: r for r in baseline.get("results", [])}

    for result in current.get("results", []):
        qid = result["id"]
        base_result = baseline_by_id.get(qid)
        if not base_result:
            continue

        base_scores = base_result.get("scores", {})
        curr_scores = result.get("scores", {})

        q_regressions = []
        for metric in base_scores:
            if metric in curr_scores:
                delta = curr_scores[metric] - base_scores[metric]
                if delta < -threshold:
                    q_regressions.append({
                        "metric": metric,
                        "baseline": round(base_scores[metric], 4),
                        "current": round(curr_scores[metric], 4),
                        "delta": round(delta, 4),
                    })

        if q_regressions:
            per_question.append({
                "id": qid,
                "question": result.get("question", ""),
                "regressions": q_regressions,
            })

    # Latency comparison
    latency_delta = None
    if "avg_latency_ms" in baseline_summary and "avg_latency_ms" in current_summary:
        latency_delta = {
            "baseline_ms": baseline_summary["avg_latency_ms"],
            "current_ms": current_summary["avg_latency_ms"],
            "delta_ms": round(
                current_summary["avg_latency_ms"] - baseline_summary["avg_latency_ms"], 1
            ),
        }

    report = {
        "passed": len(regressions) == 0,
        "threshold": threshold,
        "baseline_file": baseline_path,
        "current_file": current_path,
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "per_question_regressions": per_question,
        "latency": latency_delta,
    }

    return report


def print_regression_report(report: Dict) -> None:
    """Print a formatted regression report to stdout."""
    status = "PASSED" if report["passed"] else "FAILED"
    print(f"\n{'='*70}")
    print(f"REGRESSION TEST: {status} (threshold: {report['threshold']})")
    print(f"{'='*70}")
    print(f"  Baseline: {report['baseline_file']}")
    print(f"  Current:  {report['current_file']}")

    if report["regressions"]:
        print(f"\n  REGRESSIONS ({len(report['regressions'])}):")
        for r in report["regressions"]:
            print(f"    {r['metric']}: {r['baseline']} -> {r['current']} ({r['delta']:+.4f})")

    if report["improvements"]:
        print(f"\n  IMPROVEMENTS ({len(report['improvements'])}):")
        for r in report["improvements"]:
            print(f"    {r['metric']}: {r['baseline']} -> {r['current']} ({r['delta']:+.4f})")

    if report["unchanged"]:
        print(f"\n  UNCHANGED: {', '.join(report['unchanged'])}")

    if report["per_question_regressions"]:
        print(f"\n  PER-QUESTION REGRESSIONS ({len(report['per_question_regressions'])}):")
        for q in report["per_question_regressions"]:
            print(f"    [{q['id']}] {q['question'][:60]}")
            for r in q["regressions"]:
                print(f"      {r['metric']}: {r['baseline']} -> {r['current']} ({r['delta']:+.4f})")

    if report["latency"]:
        lat = report["latency"]
        print(f"\n  LATENCY: {lat['baseline_ms']}ms -> {lat['current_ms']}ms ({lat['delta_ms']:+.1f}ms)")

    print()
