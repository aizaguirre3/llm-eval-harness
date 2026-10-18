import pytest

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


# ── exact_match ──────────────────────────────────────────────────────────


def test_exact_match_found():
    assert exact_match("q", "42", "The answer is 42", "") == 1.0


def test_exact_match_not_found():
    assert exact_match("q", "42", "The answer is 43", "") == 0.0


def test_exact_match_is_case_insensitive():
    assert exact_match("q", "Paris", "the capital is paris.", "") == 1.0


def test_exact_match_respects_word_boundaries():
    """Plain substring matching scored these as hits."""
    assert exact_match("q", "yes", "Yesterday it rained.", "") == 0.0
    assert exact_match("q", "4", "The answer is 42.", "") == 0.0


def test_exact_match_multiword_phrase():
    assert exact_match("q", "New York", "She moved to new york last year.", "") == 1.0
    assert exact_match("q", "New York", "She moved to Newark.", "") == 0.0


def test_exact_match_empty_expected_is_vacuously_true():
    assert exact_match("q", "", "anything", "") == 1.0


# ── length_ratio ─────────────────────────────────────────────────────────


def test_length_ratio_equal():
    assert length_ratio("q", "hello world", "hello world", "") == 1.0


def test_length_ratio_verbose():
    score = length_ratio("q", "short", "this is a much longer answer than expected", "")
    assert 0.0 < score < 1.0


def test_length_ratio_terse_is_penalised_symmetrically():
    assert length_ratio("q", "ab", "abcd", "") == pytest.approx(0.5)
    assert length_ratio("q", "abcd", "ab", "") == pytest.approx(0.5)


# ── keyword_coverage ─────────────────────────────────────────────────────


def test_keyword_coverage():
    score = keyword_coverage("q", "python data science", "Python is used in data engineering", "")
    # "python" and "data" match, "science" doesn't
    assert score == pytest.approx(2 / 3)


def test_keyword_coverage_ignores_punctuation():
    """A trailing period must not turn ``science`` into an unmatched ``science.``."""
    assert keyword_coverage("q", "data science.", "Data science is fun", "") == 1.0


def test_keyword_coverage_matches_whole_words_only():
    """Substring matching counted ``is`` as present because of ``this``."""
    assert keyword_coverage("q", "is", "this", "") == 0.0
    assert keyword_coverage("q", "is", "this is it", "") == 1.0


def test_keyword_coverage_counts_distinct_words():
    # "the" appears twice in the expected answer but is one distinct word.
    assert keyword_coverage("q", "the cat and the dog", "cat dog", "") == pytest.approx(2 / 4)


# ── loading and running ──────────────────────────────────────────────────


def test_load_builtin_metrics():
    metrics = load_custom_metrics("builtin")
    assert len(metrics) == 3
    names = {m.name for m in metrics}
    assert {"exact_match", "length_ratio", "keyword_coverage"} <= names


def test_load_metrics_from_decorated_file(tmp_path):
    src = tmp_path / "plugin.py"
    src.write_text(
        "from src.scorers.custom_metrics import metric\n"
        "@metric(name='always_one', description='returns 1')\n"
        "def f(question, expected_answer, actual_answer, context):\n"
        "    return 1.0\n"
    )
    metrics = load_custom_metrics(str(src))
    assert [m.name for m in metrics] == ["always_one"]
    assert metrics[0].description == "returns 1"


def test_load_metrics_from_metrics_list(tmp_path):
    src = tmp_path / "plugin.py"
    src.write_text(
        "from src.scorers.custom_metrics import CustomMetric\n"
        "METRICS = [CustomMetric(name='m', fn=lambda q, e, a, c: 0.5)]\n"
    )
    metrics = load_custom_metrics(str(src))
    assert len(metrics) == 1 and metrics[0].name == "m"


def test_load_metrics_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_custom_metrics(str(tmp_path / "nope.py"))


def test_run_custom_metrics():
    metrics = [BUILTIN_METRICS["exact_match"]]
    results = [_make_eval_result(actual_answer="The answer is 42", expected="42")]
    scores = run_custom_metrics(metrics, results)

    assert len(scores) == 1
    assert scores[0].scores["exact_match"] == 1.0


def test_run_custom_metrics_skips_a_failing_plugin_but_keeps_the_rest(capsys):
    def boom(question, expected_answer, actual_answer, context):
        raise ValueError("bad plugin")

    metrics = [
        CustomMetric(name="boom", fn=boom),
        BUILTIN_METRICS["exact_match"],
    ]
    scores = run_custom_metrics(metrics, [_make_eval_result()])

    assert "boom" not in scores[0].scores
    assert scores[0].scores["exact_match"] == 1.0
    assert "bad plugin" in capsys.readouterr().out
