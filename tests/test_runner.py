import json
from unittest.mock import MagicMock, patch

import pytest

from src.loaders.dataset import QAPair
from src.evaluators.claude_evaluator import EvalResult
from src.scorers.ragas_scorer import ScoreResult


def _make_qa():
    return QAPair(id="t1", question="Q1?", expected_answer="A1", context="C1")


def _make_eval_result(qa=None):
    qa = qa or _make_qa()
    return EvalResult(
        qa_pair=qa,
        actual_answer="Answer 1",
        model="claude-sonnet-4-20250514",
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=5,
    )


@patch("src.runner.RagasScorer")
@patch("src.runner.ClaudeEvaluator")
@patch("src.runner.DatasetLoader")
def test_run_eval_full_pipeline(mock_loader_cls, mock_evaluator_cls, mock_scorer_cls):
    qa_pairs = [_make_qa()]
    mock_loader = MagicMock()
    mock_loader.load.return_value = qa_pairs
    mock_loader_cls.return_value = mock_loader

    eval_result = _make_eval_result(qa_pairs[0])
    mock_evaluator = MagicMock()
    mock_evaluator.model = "claude-sonnet-4-20250514"
    mock_evaluator.evaluate_batch.return_value = [eval_result]
    mock_evaluator_cls.return_value = mock_evaluator

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
    qa_pairs = [_make_qa()]
    mock_loader = MagicMock()
    mock_loader.load.return_value = qa_pairs
    mock_loader_cls.return_value = mock_loader

    mock_evaluator = MagicMock()
    mock_evaluator.model = "claude-sonnet-4-20250514"
    mock_evaluator.evaluate_batch.return_value = [_make_eval_result(qa_pairs[0])]
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


def test_export_results_json(tmp_path):
    from src.runner import _export_results

    eval_results = [_make_eval_result()]
    score_results = [
        ScoreResult(
            qa_id="t1",
            question="Q1?",
            expected_answer="A1",
            actual_answer="Answer 1",
            scores={"faithfulness": 0.95, "answer_relevancy": 0.88},
        )
    ]

    output_path = tmp_path / "test_output.json"
    result_path = _export_results(eval_results, score_results, "claude-sonnet-4-20250514", str(output_path))

    assert result_path == str(output_path)
    report = json.loads(output_path.read_text())

    assert report["metadata"]["model"] == "claude-sonnet-4-20250514"
    assert report["metadata"]["total_questions"] == 1
    assert len(report["results"]) == 1
    assert report["results"][0]["scores"]["faithfulness"] == 0.95
    assert report["summary"]["faithfulness"] == 0.95
    assert report["summary"]["answer_relevancy"] == 0.88
    assert report["summary"]["avg_latency_ms"] == 100.0
    assert report["summary"]["total_input_tokens"] == 10
    assert report["summary"]["total_output_tokens"] == 5
