from unittest.mock import MagicMock, patch

import pytest

from src.loaders.dataset import QAPair
from src.evaluators.claude_evaluator import EvalResult


@patch("src.runner.RagasScorer")
@patch("src.runner.ClaudeEvaluator")
@patch("src.runner.DatasetLoader")
def test_run_eval_full_pipeline(mock_loader_cls, mock_evaluator_cls, mock_scorer_cls):
    # Setup loader
    qa_pairs = [
        QAPair(id="t1", question="Q1?", expected_answer="A1", context="C1"),
    ]
    mock_loader = MagicMock()
    mock_loader.load.return_value = qa_pairs
    mock_loader_cls.return_value = mock_loader

    # Setup evaluator
    eval_result = EvalResult(
        qa_pair=qa_pairs[0],
        actual_answer="Answer 1",
        model="claude-sonnet-4-20250514",
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=5,
    )
    mock_evaluator = MagicMock()
    mock_evaluator.model = "claude-sonnet-4-20250514"
    mock_evaluator.evaluate_batch.return_value = [eval_result]
    mock_evaluator_cls.return_value = mock_evaluator

    # Setup scorer
    mock_scorer = MagicMock()
    mock_scorer.score.return_value = []
    mock_scorer_cls.return_value = mock_scorer

    from src.runner import run_eval

    result = run_eval(dataset_file="test.json")

    mock_loader.load.assert_called_once_with("test.json")
    mock_evaluator.evaluate_batch.assert_called_once_with(qa_pairs)
    mock_scorer.score.assert_called_once_with([eval_result])
    assert len(result["eval_results"]) == 1


@patch("src.runner.ClaudeEvaluator")
@patch("src.runner.DatasetLoader")
def test_run_eval_skip_scoring(mock_loader_cls, mock_evaluator_cls):
    qa_pairs = [QAPair(id="t1", question="Q?", expected_answer="A")]
    mock_loader = MagicMock()
    mock_loader.load.return_value = qa_pairs
    mock_loader_cls.return_value = mock_loader

    eval_result = EvalResult(
        qa_pair=qa_pairs[0],
        actual_answer="A",
        model="claude-sonnet-4-20250514",
        latency_ms=50.0,
        input_tokens=5,
        output_tokens=3,
    )
    mock_evaluator = MagicMock()
    mock_evaluator.model = "claude-sonnet-4-20250514"
    mock_evaluator.evaluate_batch.return_value = [eval_result]
    mock_evaluator_cls.return_value = mock_evaluator

    from src.runner import run_eval

    result = run_eval(skip_scoring=True)

    assert len(result["eval_results"]) == 1
    assert result["score_results"] == []


@patch("src.runner.DatasetLoader")
def test_run_eval_empty_dataset(mock_loader_cls):
    mock_loader = MagicMock()
    mock_loader.load_by_category.return_value = []
    mock_loader_cls.return_value = mock_loader

    from src.runner import run_eval

    result = run_eval(category="nonexistent")

    assert result["eval_results"] == []
    assert result["score_results"] == []
