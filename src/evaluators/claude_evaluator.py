from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

import anthropic
from langfuse import Langfuse

from src.config import settings
from src.loaders.dataset import QAPair


@dataclass
class EvalResult:
    qa_pair: QAPair
    actual_answer: str
    model: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class ClaudeEvaluator:
    """Sends Q&A pairs to Claude and collects responses with Langfuse tracing."""

    def __init__(
        self,
        model: str = settings.default_model,
        system_prompt: str = "Answer the question concisely and accurately.",
        max_tokens: int = 1024,
        max_concurrent: int = 5,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_concurrent = max_concurrent
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.async_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._init_langfuse()

    def _init_langfuse(self) -> None:
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            self.langfuse = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
        else:
            self.langfuse = None

    def _trace_generation(
        self, trace_name: str, qa: QAPair, result: EvalResult
    ) -> None:
        if not self.langfuse:
            return
        trace = self.langfuse.trace(name=trace_name, input={"question": qa.question})
        trace.generation(
            name="claude_completion",
            model=self.model,
            input=[{"role": "user", "content": qa.question}],
            output=result.actual_answer,
            usage={
                "input": result.input_tokens,
                "output": result.output_tokens,
            },
            metadata={
                "qa_id": qa.id,
                "latency_ms": result.latency_ms,
            },
        )

    def _build_user_content(self, qa: QAPair) -> str:
        if qa.context:
            return f"Context: {qa.context}\n\nQuestion: {qa.question}"
        return qa.question

    def evaluate_single(self, qa: QAPair) -> EvalResult:
        user_content = self._build_user_content(qa)

        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        latency_ms = (time.perf_counter() - start) * 1000

        result = EvalResult(
            qa_pair=qa,
            actual_answer=response.content[0].text,
            model=self.model,
            latency_ms=latency_ms,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        self._trace_generation("eval_single", qa, result)
        return result

    def evaluate_batch(self, qa_pairs: List[QAPair]) -> List[EvalResult]:
        """Evaluate a batch of Q&A pairs sequentially."""
        results = []
        for qa in qa_pairs:
            result = self.evaluate_single(qa)
            results.append(result)
        return results

    async def _evaluate_single_async(
        self, qa: QAPair, semaphore: asyncio.Semaphore
    ) -> EvalResult:
        async with semaphore:
            user_content = self._build_user_content(qa)

            start = time.perf_counter()
            response = await self.async_client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            latency_ms = (time.perf_counter() - start) * 1000

            result = EvalResult(
                qa_pair=qa,
                actual_answer=response.content[0].text,
                model=self.model,
                latency_ms=latency_ms,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

            self._trace_generation("eval_async", qa, result)
            return result

    async def evaluate_batch_async(self, qa_pairs: List[QAPair]) -> List[EvalResult]:
        """Evaluate a batch of Q&A pairs concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        tasks = [self._evaluate_single_async(qa, semaphore) for qa in qa_pairs]
        return await asyncio.gather(*tasks)

    def evaluate_batch_concurrent(self, qa_pairs: List[QAPair]) -> List[EvalResult]:
        """Evaluate a batch concurrently (sync wrapper for async batch)."""
        return asyncio.run(self.evaluate_batch_async(qa_pairs))

    def flush(self) -> None:
        """Flush Langfuse traces. Call after eval runs complete."""
        if self.langfuse:
            self.langfuse.flush()
