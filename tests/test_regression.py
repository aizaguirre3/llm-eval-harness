import json

from src.regression import compare_results


def test_regression_detects_score_drop(tmp_path):
    baseline = {
        "metadata": {"model": "sonnet", "total_questions": 1},
        "results": [
            {"id": "q1", "question": "Q?", "scores": {"faithfulness": 0.90}}
        ],
        "summary": {"faithfulness": 0.90, "avg_latency_ms": 100.0,
                     "total_input_tokens": 10, "total_output_tokens": 5},
    }
    current = {
        "metadata": {"model": "sonnet", "total_questions": 1},
        "results": [
            {"id": "q1", "question": "Q?", "scores": {"faithfulness": 0.70}}
        ],
        "summary": {"faithfulness": 0.70, "avg_latency_ms": 120.0,
                     "total_input_tokens": 10, "total_output_tokens": 5},
    }

    base_path = tmp_path / "baseline.json"
    curr_path = tmp_path / "current.json"
    base_path.write_text(json.dumps(baseline))
    curr_path.write_text(json.dumps(current))

    report = compare_results(str(curr_path), str(base_path), threshold=0.05)

    assert report["passed"] is False
    assert len(report["regressions"]) == 1
    assert report["regressions"][0]["metric"] == "faithfulness"
    assert report["regressions"][0]["delta"] == -0.2


def test_regression_passes_within_threshold(tmp_path):
    baseline = {
        "metadata": {"model": "sonnet", "total_questions": 1},
        "results": [],
        "summary": {"faithfulness": 0.90, "avg_latency_ms": 100.0,
                     "total_input_tokens": 10, "total_output_tokens": 5},
    }
    current = {
        "metadata": {"model": "sonnet", "total_questions": 1},
        "results": [],
        "summary": {"faithfulness": 0.87, "avg_latency_ms": 100.0,
                     "total_input_tokens": 10, "total_output_tokens": 5},
    }

    base_path = tmp_path / "baseline.json"
    curr_path = tmp_path / "current.json"
    base_path.write_text(json.dumps(baseline))
    curr_path.write_text(json.dumps(current))

    report = compare_results(str(curr_path), str(base_path), threshold=0.05)

    assert report["passed"] is True
    assert len(report["regressions"]) == 0


def test_regression_detects_improvement(tmp_path):
    baseline = {
        "metadata": {},
        "results": [],
        "summary": {"faithfulness": 0.70, "avg_latency_ms": 100.0,
                     "total_input_tokens": 10, "total_output_tokens": 5},
    }
    current = {
        "metadata": {},
        "results": [],
        "summary": {"faithfulness": 0.90, "avg_latency_ms": 100.0,
                     "total_input_tokens": 10, "total_output_tokens": 5},
    }

    base_path = tmp_path / "baseline.json"
    curr_path = tmp_path / "current.json"
    base_path.write_text(json.dumps(baseline))
    curr_path.write_text(json.dumps(current))

    report = compare_results(str(curr_path), str(base_path), threshold=0.05)

    assert report["passed"] is True
    assert len(report["improvements"]) == 1
    assert report["improvements"][0]["delta"] == 0.2


def test_regression_reports_per_question_and_latency(tmp_path):
    """The per-question drill-down and latency delta are the parts of the
    report RESULTS.md actually shows; neither was covered."""
    baseline = {
        "metadata": {},
        "results": [
            {"id": "q1", "question": "First?", "scores": {"faithfulness": 1.0}},
            {"id": "q2", "question": "Second?", "scores": {"faithfulness": 0.9}},
        ],
        "summary": {"faithfulness": 0.95, "avg_latency_ms": 100.0,
                    "total_input_tokens": 10, "total_output_tokens": 5},
    }
    current = {
        "metadata": {},
        "results": [
            {"id": "q1", "question": "First?", "scores": {"faithfulness": 0.5}},
            {"id": "q2", "question": "Second?", "scores": {"faithfulness": 0.92}},
            {"id": "q3", "question": "New, no baseline", "scores": {"faithfulness": 0.1}},
        ],
        "summary": {"faithfulness": 0.71, "avg_latency_ms": 150.0,
                    "total_input_tokens": 10, "total_output_tokens": 5},
    }
    base_path = tmp_path / "baseline.json"
    curr_path = tmp_path / "current.json"
    base_path.write_text(json.dumps(baseline))
    curr_path.write_text(json.dumps(current))

    report = compare_results(str(curr_path), str(base_path), threshold=0.05)

    assert [q["id"] for q in report["per_question_regressions"]] == ["q1"]
    assert report["per_question_regressions"][0]["regressions"][0]["delta"] == -0.5
    assert report["latency"] == {"baseline_ms": 100.0, "current_ms": 150.0, "delta_ms": 50.0}


def test_performance_counters_are_never_flagged_as_regressions(tmp_path):
    baseline = {"results": [], "summary": {"avg_latency_ms": 100.0,
                                             "total_input_tokens": 10, "total_output_tokens": 5}}
    current = {"results": [], "summary": {"avg_latency_ms": 900.0,
                                            "total_input_tokens": 999, "total_output_tokens": 999}}
    base_path = tmp_path / "b.json"
    curr_path = tmp_path / "c.json"
    base_path.write_text(json.dumps(baseline))
    curr_path.write_text(json.dumps(current))

    report = compare_results(str(curr_path), str(base_path))
    assert report["passed"] is True
    assert report["regressions"] == []
