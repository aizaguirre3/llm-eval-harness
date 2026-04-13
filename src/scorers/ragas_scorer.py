from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, List

from langchain_anthropic import ChatAnthropic
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.llms.base import LangchainLLMWrapper

from src.config import settings
from src.evaluators.claude_evaluator import EvalResult

# Suppress deprecation warnings for legacy metrics (still functional in v0.4)
warnings.filterwarnings("ignore", message=".*Importing.*from 'ragas.metrics' is deprecated.*")
from ragas.metrics import (  # noqa: E402
    context_precision,
    context_recall,
    faithfulness,
)


@dataclass
class ScoreResult:
    qa_id: str
    question: str
    expected_answer: str
    actual_answer: str
    scores: Dict[str, float]


class RagasScorer:
    """Scores eval results using RAGAS metrics backed by Claude."""

    def __init__(self, model: str = settings.default_model):
        self.llm = LangchainLLMWrapper(
            ChatAnthropic(
                model=model,
                anthropic_api_key=settings.anthropic_api_key,
                max_tokens=4096,
            )
        )
        self.metrics = [
            faithfulness,
            context_precision,
            context_recall,
        ]

    def _build_dataset(self, eval_results: List[EvalResult]) -> EvaluationDataset:
        samples = []
        for r in eval_results:
            samples.append(
                SingleTurnSample(
                    user_input=r.qa_pair.question,
                    response=r.actual_answer,
                    reference=r.qa_pair.expected_answer,
                    retrieved_contexts=[r.qa_pair.context] if r.qa_pair.context else [],
                )
            )
        return EvaluationDataset(samples=samples)

    def score(self, eval_results: List[EvalResult]) -> List[ScoreResult]:
        dataset = self._build_dataset(eval_results)

        ragas_result = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            show_progress=True,
        )

        df = ragas_result.to_pandas()
        score_results = []
        for i, r in enumerate(eval_results):
            scores = {}
            for col in df.columns:
                if col not in ("user_input", "response", "reference", "retrieved_contexts"):
                    val = df.iloc[i][col]
                    if val is not None and not (isinstance(val, float) and math.isnan(val)):
                        scores[col] = float(val)

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
