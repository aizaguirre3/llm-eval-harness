from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.evaluators.claude_evaluator import EvalResult
from src.loaders.dataset import QAPair
from src.scorers.ragas_scorer import RagasScorer, ScoreResult


def _make_eval_result(qa_id: str = "t1") -> EvalResult:
    return EvalResult(
        qa_pair=QAPair(
            id=qa_id,
            question="What is RAG?",
            expected_answer="Retrieval-augmented generation.",
            context="RAG combines retrieval with generation.",
        ),
        actual_answer="RAG is retrieval-augmented generation.",
        model="claude-sonnet-4-20250514",
        latency_ms=500.0,
        input_tokens=20,
        output_tokens=10,
    )


@patch("src.scorers.ragas_scorer.evaluate")
@patch("src.scorers.ragas_scorer.AnswerRelevancy")
@patch("src.scorers.ragas_scorer.Faithfulness")
@patch("src.scorers.ragas_scorer.llm_factory")
@patch("src.scorers.ragas_scorer.anthropic.Anthropic")
@patch("src.scorers.ragas_scorer.settings")
def test_score_returns_results(
    mock_settings, mock_anthropic, mock_llm_factory, mock_faith, mock_relevancy, mock_evaluate
):
    mock_settings.anthropic_api_key = "sk-test"
    mock_settings.default_model = "claude-sonnet-4-20250514"
    mock_llm_factory.return_value = MagicMock()

    mock_ragas_result = MagicMock()
    mock_ragas_result.to_pandas.return_value = pd.DataFrame(
        {
            "user_input": ["What is RAG?"],
            "response": ["RAG is retrieval-augmented generation."],
            "reference": ["Retrieval-augmented generation."],
            "retrieved_contexts": [["RAG combines retrieval with generation."]],
            "faithfulness": [0.95],
            "answer_relevancy": [0.88],
        }
    )
    mock_evaluate.return_value = mock_ragas_result

    scorer = RagasScorer()
    results = scorer.score([_make_eval_result()])

    assert len(results) == 1
    assert isinstance(results[0], ScoreResult)
    assert results[0].qa_id == "t1"
    assert results[0].scores["faithfulness"] == 0.95
    assert results[0].scores["answer_relevancy"] == 0.88


@patch("src.scorers.ragas_scorer.evaluate")
@patch("src.scorers.ragas_scorer.AnswerRelevancy")
@patch("src.scorers.ragas_scorer.Faithfulness")
@patch("src.scorers.ragas_scorer.llm_factory")
@patch("src.scorers.ragas_scorer.anthropic.Anthropic")
@patch("src.scorers.ragas_scorer.settings")
def test_score_batch(
    mock_settings, mock_anthropic, mock_llm_factory, mock_faith, mock_relevancy, mock_evaluate
):
    mock_settings.anthropic_api_key = "sk-test"
    mock_settings.default_model = "claude-sonnet-4-20250514"
    mock_llm_factory.return_value = MagicMock()

    mock_ragas_result = MagicMock()
    mock_ragas_result.to_pandas.return_value = pd.DataFrame(
        {
            "user_input": ["Q1?", "Q2?"],
            "response": ["A1", "A2"],
            "reference": ["R1", "R2"],
            "retrieved_contexts": [["C1"], ["C2"]],
            "faithfulness": [0.9, 0.8],
            "answer_relevancy": [0.85, 0.75],
        }
    )
    mock_evaluate.return_value = mock_ragas_result

    scorer = RagasScorer()
    results = scorer.score([_make_eval_result("t1"), _make_eval_result("t2")])

    assert len(results) == 2
    assert results[1].scores["faithfulness"] == 0.8
