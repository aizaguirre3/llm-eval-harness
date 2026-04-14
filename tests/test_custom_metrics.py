from src.evaluators.claude_evaluator import EvalResult
from src.loaders.dataset import QAPair
from src.scorers.custom_metrics import (
    BUILTIN_METRICS,
    CustomMetric,
    exact_match,
    keyword_coverage,
    length_ratio,
    load_custom_metrics,
    run_custom_metrics,
)


def _make_eval_result(actual_answer="The answer is 42", expected="42"):
    qa = QAPair(id="t1", question="What is it?", expected_answer=expected, context="Some context")
    return EvalResult(
        qa_pair=qa,
        actual_answer=actual_answer,
        model="test",
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=5,
    )


def test_exact_match_found():
    assert exact_match("q", "42", "The answer is 42", "") == 1.0


def test_exact_match_not_found():
    assert exact_match("q", "42", "The answer is 43", "") == 0.0


def test_length_ratio_equal():
    score = length_ratio("q", "hello world", "hello world", "")
    assert score == 1.0


def test_length_ratio_verbose():
    score = length_ratio("q", "short", "this is a much longer answer than expected", "")
    assert 0.0 < score < 1.0


def test_keyword_coverage():
    score = keyword_coverage("q", "python data science", "Python is used in data engineering", "")
    # "python" and "data" match, "science" doesn't
    assert 0.5 < score < 1.0


def test_load_builtin_metrics():
    metrics = load_custom_metrics("builtin")
    assert len(metrics) == 3
    names = {m.name for m in metrics}
    assert "exact_match" in names
    assert "length_ratio" in names
    assert "keyword_coverage" in names


def test_run_custom_metrics():
    metrics = [BUILTIN_METRICS["exact_match"]]
    results = [_make_eval_result(actual_answer="The answer is 42", expected="42")]
    scores = run_custom_metrics(metrics, results)

    assert len(scores) == 1
    assert scores[0].scores["exact_match"] == 1.0
