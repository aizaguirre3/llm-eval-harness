from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

from src.evaluators.claude_evaluator import EvalResult
from src.scorers.ragas_scorer import ScoreResult


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
    """Returns 1.0 if the expected answer appears verbatim in the actual answer."""
    return 1.0 if expected_answer.lower() in actual_answer.lower() else 0.0


def length_ratio(question: str, expected_answer: str, actual_answer: str, context: str) -> float:
    """Ratio of actual answer length to expected answer length, capped at 1.0.

    Useful for detecting overly verbose or terse responses.
    """
    if not expected_answer:
        return 1.0
    ratio = len(actual_answer) / len(expected_answer)
    # Penalize both too short and too long — ideal is around 1.0
    if ratio > 1.0:
        return min(1.0, 1.0 / ratio)
    return ratio


def keyword_coverage(
    question: str, expected_answer: str, actual_answer: str, context: str
) -> float:
    """Fraction of words from the expected answer that appear in the actual answer."""
    expected_words = set(expected_answer.lower().split())
    if not expected_words:
        return 1.0
    actual_lower = actual_answer.lower()
    matched = sum(1 for w in expected_words if w in actual_lower)
    return matched / len(expected_words)


BUILTIN_METRICS = {
    "exact_match": CustomMetric(name="exact_match", fn=exact_match, description="Checks if expected answer appears in actual answer"),
    "length_ratio": CustomMetric(name="length_ratio", fn=length_ratio, description="Ratio of actual to expected answer length"),
    "keyword_coverage": CustomMetric(name="keyword_coverage", fn=keyword_coverage, description="Fraction of expected words found in actual answer"),
}


# ── Loading and running ──────────────────────────────────────────────────


def load_custom_metrics(path: str) -> List[CustomMetric]:
    """Load custom metrics from a Python file.

    The file should define a `METRICS` list of CustomMetric instances,
    or define functions decorated with @metric.
    If path is "builtin", returns all built-in custom metrics.
    """
    if path == "builtin":
        return list(BUILTIN_METRICS.values())

    spec = importlib.util.spec_from_file_location("custom_metrics_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_metrics_module"] = module
    spec.loader.exec_module(module)

    # Look for METRICS list first
    if hasattr(module, "METRICS"):
        return module.METRICS

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
    metrics: List[CustomMetric],
    eval_results: List[EvalResult],
) -> List[ScoreResult]:
    """Run custom metric functions against eval results."""
    score_results = []

    for r in eval_results:
        scores: Dict[str, float] = {}
        for m in metrics:
            try:
                score = m.fn(
                    question=r.qa_pair.question,
                    expected_answer=r.qa_pair.expected_answer,
                    actual_answer=r.actual_answer,
                    context=r.qa_pair.context,
                )
                scores[m.name] = float(score)
            except Exception as e:
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
