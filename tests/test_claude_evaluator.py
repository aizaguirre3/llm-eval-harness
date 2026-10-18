from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.evaluators.claude_evaluator import ClaudeEvaluator, EvalResult, _response_text
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


def _configure(mock_settings, *, langfuse: bool = False) -> None:
    mock_settings.anthropic_api_key = "test-key"
    mock_settings.default_model = "claude-sonnet-4-20250514"
    mock_settings.langfuse_public_key = "pk-test" if langfuse else ""
    mock_settings.langfuse_secret_key = "sk-test" if langfuse else ""
    mock_settings.langfuse_host = "https://langfuse.test"


@patch("src.evaluators.claude_evaluator.anthropic.AsyncAnthropic")
@patch("src.evaluators.claude_evaluator.anthropic.Anthropic")
@patch("src.evaluators.claude_evaluator.settings")
def test_evaluate_single(mock_settings, mock_anthropic_cls, mock_async_cls, mock_response, sample_qa):
    _configure(mock_settings)
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
    _configure(mock_settings)
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


@patch("src.evaluators.claude_evaluator.anthropic.AsyncAnthropic")
@patch("src.evaluators.claude_evaluator.anthropic.Anthropic")
@patch("src.evaluators.claude_evaluator.settings")
def test_evaluate_batch_concurrent_preserves_order(
    mock_settings, mock_anthropic_cls, mock_async_cls, mock_response
):
    """The concurrent path was advertised but never tested. Results must come
    back in input order regardless of completion order, because the exporter
    and scorer index them positionally."""
    _configure(mock_settings)

    async def fake_create(**kwargs):
        content = kwargs["messages"][0]["content"]
        resp = MagicMock()
        resp.content = [MagicMock(text=f"answer to {content}")]
        resp.usage = MagicMock(input_tokens=1, output_tokens=1)
        return resp

    mock_async_client = MagicMock()
    mock_async_client.messages.create = AsyncMock(side_effect=fake_create)
    mock_async_cls.return_value = mock_async_client

    pairs = [QAPair(id=f"t{i}", question=f"Q{i}?", expected_answer="A") for i in range(6)]

    evaluator = ClaudeEvaluator(max_concurrent=2)
    results = evaluator.evaluate_batch_concurrent(pairs)

    assert [r.qa_pair.id for r in results] == [p.id for p in pairs]
    assert [r.actual_answer for r in results] == [f"answer to Q{i}?" for i in range(6)]
    assert mock_async_client.messages.create.await_count == 6


@patch("src.evaluators.claude_evaluator.Langfuse")
@patch("src.evaluators.claude_evaluator.anthropic.AsyncAnthropic")
@patch("src.evaluators.claude_evaluator.anthropic.Anthropic")
@patch("src.evaluators.claude_evaluator.settings")
def test_langfuse_tracing_uses_v3_observation_api(
    mock_settings, mock_anthropic_cls, mock_async_cls, mock_langfuse_cls, mock_response, sample_qa
):
    """Regression test for the Langfuse 3.0 API change.

    The original code called ``trace().generation()``, which no longer exists,
    so every traced call raised ``AttributeError`` -- and no test ever set the
    Langfuse keys, so it went unnoticed.
    """
    _configure(mock_settings, langfuse=True)
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_cls.return_value = mock_client

    mock_langfuse = MagicMock()
    mock_generation = MagicMock()
    mock_langfuse.start_observation.return_value = mock_generation
    mock_langfuse_cls.return_value = mock_langfuse

    evaluator = ClaudeEvaluator()
    evaluator.evaluate_single(sample_qa)
    evaluator.flush()

    mock_langfuse_cls.assert_called_once_with(
        public_key="pk-test", secret_key="sk-test", host="https://langfuse.test"
    )
    kwargs = mock_langfuse.start_observation.call_args.kwargs
    assert kwargs["as_type"] == "generation"
    assert kwargs["model"] == "claude-sonnet-4-20250514"
    assert kwargs["output"] == "Paris is the capital of France."
    assert kwargs["usage_details"] == {"input": 25, "output": 10}
    assert kwargs["metadata"]["qa_id"] == "t1"
    mock_generation.end.assert_called_once()
    mock_langfuse.flush.assert_called_once()


@patch("src.evaluators.claude_evaluator.Langfuse")
@patch("src.evaluators.claude_evaluator.anthropic.AsyncAnthropic")
@patch("src.evaluators.claude_evaluator.anthropic.Anthropic")
@patch("src.evaluators.claude_evaluator.settings")
def test_langfuse_is_not_constructed_without_keys(
    mock_settings, mock_anthropic_cls, mock_async_cls, mock_langfuse_cls
):
    _configure(mock_settings, langfuse=False)
    evaluator = ClaudeEvaluator()
    assert evaluator.langfuse is None
    mock_langfuse_cls.assert_not_called()
    evaluator.flush()  # must be a no-op, not an error


def test_response_text_concatenates_only_text_blocks():
    """A response can contain non-text blocks (tool use); they carry no ``text``."""
    text_a = MagicMock(text="Hello, ")
    tool = MagicMock(spec=["type", "id", "name", "input"])
    text_b = MagicMock(text="world.")
    response = MagicMock(content=[text_a, tool, text_b])
    assert _response_text(response) == "Hello, world."


def test_response_text_handles_empty_content():
    assert _response_text(MagicMock(content=[])) == ""
