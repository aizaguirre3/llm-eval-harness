from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from src.config import PROJECT_ROOT
from src.evaluators.claude_evaluator import ClaudeEvaluator, EvalResult
from src.loaders.dataset import DatasetLoader
from src.scorers.ragas_scorer import RagasScorer, ScoreResult

RESULTS_DIR = PROJECT_ROOT / "results"


def run_eval(
    dataset_file: str = "sample_qa.json",
    model: str = "",
    category: str = "",
    skip_scoring: bool = False,
    output_json: str = "",
    concurrent: bool = False,
) -> dict:
    """Run the full eval pipeline: load -> evaluate -> score -> export."""

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

    if concurrent:
        print(f"\nEvaluating with {evaluator.model} (concurrent, max {evaluator.max_concurrent} requests)...")
        eval_results = evaluator.evaluate_batch_concurrent(qa_pairs)
    else:
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

    # 4. Export JSON results
    output_path = _export_results(eval_results, score_results, evaluator.model, output_json)
    if output_path:
        print(f"\nResults exported to: {output_path}")

    return {
        "eval_results": eval_results,
        "score_results": score_results,
    }


def _export_results(
    eval_results: List[EvalResult],
    score_results: List[ScoreResult],
    model: str,
    output_json: str,
) -> str:
    RESULTS_DIR.mkdir(exist_ok=True)

    if output_json:
        output_path = Path(output_json)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"eval_{timestamp}.json"

    scores_by_id = {s.qa_id: s.scores for s in score_results}

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "total_questions": len(eval_results),
        },
        "results": [],
        "summary": {},
    }

    for r in eval_results:
        entry = {
            "id": r.qa_pair.id,
            "question": r.qa_pair.question,
            "context": r.qa_pair.context,
            "expected_answer": r.qa_pair.expected_answer,
            "actual_answer": r.actual_answer,
            "latency_ms": round(r.latency_ms, 1),
            "input_tokens": r.input_tokens,
            "output_tokens": r.output_tokens,
            "scores": scores_by_id.get(r.qa_pair.id, {}),
        }
        report["results"].append(entry)

    # Compute summary averages
    if score_results:
        all_metrics = set()
        for s in score_results:
            all_metrics.update(s.scores.keys())

        for metric in sorted(all_metrics):
            values = [s.scores[metric] for s in score_results if metric in s.scores]
            if values:
                report["summary"][metric] = round(sum(values) / len(values), 4)

    # Latency stats
    latencies = [r.latency_ms for r in eval_results]
    report["summary"]["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1)
    report["summary"]["total_input_tokens"] = sum(r.input_tokens for r in eval_results)
    report["summary"]["total_output_tokens"] = sum(r.output_tokens for r in eval_results)

    output_path.write_text(json.dumps(report, indent=2))
    return str(output_path)


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
        "-o", "--output",
        default="",
        help="Output JSON file path (default: results/eval_<timestamp>.json)",
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run evaluations concurrently (faster, uses async API calls)",
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
        output_json=args.output,
        concurrent=args.concurrent,
    )


if __name__ == "__main__":
    main()
