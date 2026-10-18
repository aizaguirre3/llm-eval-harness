from __future__ import annotations

import importlib.util
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.evaluators.claude_evaluator import EvalResult
from src.scorers.ragas_scorer import ScoreResult

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set[str]:
    """Lower-cased word tokens, punctuation stripped.

    Both built-in text metrics compare on tokens rather than substrings. With
    plain substring tests, an expected keyword ``is`` matched inside ``this``
    and an expected answer ``yes`` matched ``yesterday`` -- both silently
    inflated scores.
    """
    return set(_WORD.findall(text.lower()))


@dataclass
class CustomMetric:
    """A user-defined metric function.

    The function receives (question, expected_answer, actual_answer, context)
    and returns a float score between 0 and 1.
    """

    name: str
    fn: Callable[..., float]
    description: str = ""


# ── Built-in custom metrics ──────────────────────────────────────────────


def exact_match(question: str, expected_answer: str, actual_answer: str, context: str) -> float:
    """1.0 if the expected answer appears in the actual answer as a whole-word phrase.

    Matching is case-insensitive and respects word boundaries, so an expected
    ``"4"`` does not match ``"42"`` and ``"yes"`` does not match ``"yesterday"``.
    """
    expected = expected_answer.strip()
    if not expected:
        return 1.0
    pattern = r"(?<![a-z0-9])" + re.escape(expected.lower()) + r"(?![a-z0-9])"
    return 1.0 if re.search(pattern, actual_answer.lower()) else 0.0


def length_ratio(question: str, expected_answer: str, actual_answer: str, context: str) -> float:
    """Ratio of actual answer length to expected answer length, capped at 1.0.

    Useful for detecting overly verbose or terse responses.
    """
    if not expected_answer:
        return 1.0
    ratio = len(actual_answer) / len(expected_answer)
    # Penalize both too short and too long -- ideal is around 1.0
    if ratio > 1.0:
        return min(1.0, 1.0 / ratio)
    return ratio


def keyword_coverage(
    question: str, expected_answer: str, actual_answer: str, context: str
) -> float:
    """Fraction of distinct words in the expected answer that appear in the actual answer."""
    expected_words = _tokens(expected_answer)
    if not expected_words:
        return 1.0
    actual_words = _tokens(actual_answer)
    return len(expected_words & actual_words) / len(expected_words)


BUILTIN_METRICS = {
    "exact_match": CustomMetric(
        name="exact_match",
        fn=exact_match,
        description="Checks if the expected answer appears in the actual answer as a whole phrase",
    ),
    "length_ratio": CustomMetric(
        name="length_ratio",
        fn=length_ratio,
        description="Ratio of actual to expected answer length",
    ),
    "keyword_coverage": CustomMetric(
        name="keyword_coverage",
        fn=keyword_coverage,
        description="Fraction of expected-answer words found in the actual answer",
    ),
}


# ── Loading and running ──────────────────────────────────────────────────


def load_custom_metrics(path: str) -> list[CustomMetric]:
    """Load custom metrics from a Python file.

    The file should define a `METRICS` list of CustomMetric instances,
    or define functions decorated with @metric.
    If path is "builtin", returns all built-in custom metrics.

    This executes the file as Python. It is a plugin mechanism and is only safe
    for files you wrote or reviewed.
    """
    if path == "builtin":
        return list(BUILTIN_METRICS.values())

    if not Path(path).is_file():
        raise FileNotFoundError(f"Custom metrics file not found: {path}")

    spec = importlib.util.spec_from_file_location("custom_metrics_module", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load custom metrics from: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_metrics_module"] = module
    spec.loader.exec_module(module)

    # Look for METRICS list first
    if hasattr(module, "METRICS"):
        return list(module.METRICS)

    # Otherwise, collect all functions with a _is_metric attribute
    metrics = []
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and getattr(obj, "_is_metric", False):
            metrics.append(
                CustomMetric(
                    name=getattr(obj, "_metric_name", name),
                    fn=obj,
                    description=getattr(obj, "_metric_description", obj.__doc__ or ""),
                )
            )

    return metrics


def metric(name: str = "", description: str = "") -> Callable:
    """Decorator to mark a function as a custom metric."""

    def decorator(fn: Callable) -> Callable:
        fn._is_metric = True
        fn._metric_name = name or fn.__name__
        fn._metric_description = description or fn.__doc__ or ""
        return fn

    return decorator


def run_custom_metrics(
    metrics: list[CustomMetric],
    eval_results: list[EvalResult],
) -> list[ScoreResult]:
    """Run custom metric functions against eval results.

    A metric that raises is reported and skipped for that item rather than
    aborting the run: these are user plugins, and one bad plugin should not
    discard the scores of the others.
    """
    score_results = []

    for r in eval_results:
        scores: dict[str, float] = {}
        for m in metrics:
            try:
                score = m.fn(
                    question=r.qa_pair.question,
                    expected_answer=r.qa_pair.expected_answer,
                    actual_answer=r.actual_answer,
                    context=r.qa_pair.context,
                )
                scores[m.name] = float(score)
            except Exception as e:  # noqa: BLE001 - plugin boundary, see docstring
                print(f"  Warning: metric '{m.name}' failed for [{r.qa_pair.id}]: {e}")

        score_results.append(
            ScoreResult(
                qa_id=r.qa_pair.id,
                question=r.qa_pair.question,
                expected_answer=r.qa_pair.expected_answer,
                actual_answer=r.actual_answer,
                scores=scores,
            )
        )

    return score_results
