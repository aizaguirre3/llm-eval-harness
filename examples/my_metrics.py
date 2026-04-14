"""Example custom metrics file.

Usage:
    python -m src.runner -d ai_engineering.json --skip-scoring --custom-metrics examples/my_metrics.py

Define metrics either as a METRICS list or using the @metric decorator.
Each metric function receives (question, expected_answer, actual_answer, context)
and should return a float between 0 and 1.
"""

from src.scorers.custom_metrics import CustomMetric, metric


# ── Option 1: Use the @metric decorator ─────────────────────────────────


@metric(name="has_code_block", description="Checks if the response contains a code block")
def has_code_block(question: str, expected_answer: str, actual_answer: str, context: str) -> float:
    return 1.0 if "```" in actual_answer else 0.0


@metric(name="conciseness", description="Penalizes answers over 500 words")
def conciseness(question: str, expected_answer: str, actual_answer: str, context: str) -> float:
    word_count = len(actual_answer.split())
    if word_count <= 200:
        return 1.0
    elif word_count <= 500:
        return 0.7
    return 0.3


# ── Option 2: Define a METRICS list (uncomment to use instead) ──────────

# METRICS = [
#     CustomMetric(
#         name="my_metric",
#         fn=lambda q, e, a, c: 1.0 if len(a) > 10 else 0.0,
#         description="Checks answer is longer than 10 chars",
#     ),
# ]
