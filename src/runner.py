from __future__ import annotations

import argparse
import json
import sys
from typing import List

from src.evaluators.claude_evaluator import ClaudeEvaluator, EvalResult
from src.loaders.dataset import DatasetLoader
from src.scorers.ragas_scorer import RagasScorer, ScoreResult


def run_eval(
    dataset_file: str = "sample_qa.json",
    model: str = "",
    category: str = "",
    skip_scoring: bool = False,
) -> dict:
    """Run the full eval pipeline: load → evaluate → score."""

    # 1. Load dataset
    loader = DatasetLoader()
    if category:
        qa_pairs = loader.load_by_category(dataset_file, category)
        print(f"Loaded {len(qa_pairs)} Q&A pairs (category: {category})")
    else:
        qa_pairs = loader.load(dataset_file)
        print(f"Loaded {len(qa_pairs)} Q&A pairs")

    if not qa_pairs:
        print("No Q&A pairs found. Exiting.")
        return {"eval_results": [], "score_results": []}

    # 2. Evaluate with Claude
    evaluator_kwargs = {}
    if model:
        evaluator_kwargs["model"] = model
    evaluator = ClaudeEvaluator(**evaluator_kwargs)

    print(f"\nEvaluating with {evaluator.model}...")
    eval_results = evaluator.evaluate_batch(qa_pairs)
    evaluator.flush()

    _print_eval_results(eval_results)

    # 3. Score with RAGAS
    score_results = []
    if not skip_scoring:
        print("\nScoring with RAGAS...")
        scorer = RagasScorer(model=evaluator.model)
        score_results = scorer.score(eval_results)
        _print_score_results(score_results)

    return {
        "eval_results": eval_results,
        "score_results": score_results,
    }


def _print_eval_results(results: List[EvalResult]) -> None:
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    for r in results:
        print(f"\n[{r.qa_pair.id}] {r.qa_pair.question}")
        print(f"  Expected: {r.qa_pair.expected_answer[:80]}...")
        print(f"  Actual:   {r.actual_answer[:80]}...")
        print(f"  Latency:  {r.latency_ms:.0f}ms | Tokens: {r.input_tokens}in/{r.output_tokens}out")


def _print_score_results(results: List[ScoreResult]) -> None:
    print(f"\n{'='*70}")
    print("RAGAS SCORES")
    print(f"{'='*70}")
    for r in results:
        print(f"\n[{r.qa_id}] {r.question}")
        for metric, value in r.scores.items():
            print(f"  {metric}: {value:.3f}")

    # Print averages
    if results:
        all_metrics = set()
        for r in results:
            all_metrics.update(r.scores.keys())

        print(f"\n{'-'*70}")
        print("AVERAGES")
        for metric in sorted(all_metrics):
            values = [r.scores[metric] for r in results if metric in r.scores]
            if values:
                avg = sum(values) / len(values)
                print(f"  {metric}: {avg:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Eval Harness - Evaluate LLM responses")
    parser.add_argument(
        "-d", "--dataset",
        default="sample_qa.json",
        help="Dataset filename in /data folder (default: sample_qa.json)",
    )
    parser.add_argument(
        "-m", "--model",
        default="",
        help="Claude model to use (default: from settings)",
    )
    parser.add_argument(
        "-c", "--category",
        default="",
        help="Filter Q&A pairs by category",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip RAGAS scoring (only run Claude evaluation)",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List available datasets and exit",
    )

    args = parser.parse_args()

    if args.list_datasets:
        loader = DatasetLoader()
        datasets = loader.list_datasets()
        print("Available datasets:")
        for d in datasets:
            print(f"  - {d}")
        return

    run_eval(
        dataset_file=args.dataset,
        model=args.model,
        category=args.category,
        skip_scoring=args.skip_scoring,
    )


if __name__ == "__main__":
    main()
