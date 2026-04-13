from unittest.mock import MagicMock, patch

import pytest

from src.evaluators.claude_evaluator import ClaudeEvaluator, EvalResult
from src.loaders.dataset import QAPair


@pytest.fixture
def mock_response():
    response = MagicMock()
    response.content = [MagicMock(text="Paris is the capital of France.")]
    response.usage = MagicMock(input_tokens=25, output_tokens=10)
    return response


@pytest.fixture
def sample_qa():
    return QAPair(
        id="t1",
        question="What is the capital of France?",
        expected_answer="Paris",
        context="European geography",
        metadata={"category": "geography"},
    )


@patch("src.evaluators.claude_evaluator.anthropic.AsyncAnthropic")
@patch("src.evaluators.claude_evaluator.anthropic.Anthropic")
@patch("src.evaluators.claude_evaluator.settings")
def test_evaluate_single(mock_settings, mock_anthropic_cls, mock_async_cls, mock_response, sample_qa):
    mock_settings.anthropic_api_key = "sk-test"
    mock_settings.default_model = "claude-sonnet-4-20250514"
    mock_settings.langfuse_public_key = ""
    mock_settings.langfuse_secret_key = ""
    mock_settings.langfuse_host = ""

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    evaluator = ClaudeEvaluator()
    result = evaluator.evaluate_single(sample_qa)

    assert isinstance(result, EvalResult)
    assert result.actual_answer == "Paris is the capital of France."
    assert result.input_tokens == 25
    assert result.output_tokens == 10
    assert result.qa_pair.id == "t1"

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert "European geography" in call_kwargs["messages"][0]["content"]


@patch("src.evaluators.claude_evaluator.anthropic.AsyncAnthropic")
@patch("src.evaluators.claude_evaluator.anthropic.Anthropic")
@patch("src.evaluators.claude_evaluator.settings")
def test_evaluate_batch(mock_settings, mock_anthropic_cls, mock_async_cls, mock_response):
    mock_settings.anthropic_api_key = "sk-test"
    mock_settings.default_model = "claude-sonnet-4-20250514"
    mock_settings.langfuse_public_key = ""
    mock_settings.langfuse_secret_key = ""
    mock_settings.langfuse_host = ""

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    pairs = [
        QAPair(id="t1", question="Q1?", expected_answer="A1"),
        QAPair(id="t2", question="Q2?", expected_answer="A2"),
    ]

    evaluator = ClaudeEvaluator()
    results = evaluator.evaluate_batch(pairs)

    assert len(results) == 2
    assert mock_client.messages.create.call_count == 2
