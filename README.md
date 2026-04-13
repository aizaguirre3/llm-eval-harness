# LLM Eval Harness

A cloud-native evaluation harness for LLM responses, built with the Anthropic Claude API, RAGAS for automated scoring, and Langfuse for observability. Supports sequential and concurrent evaluation, model comparison, and automated JSON reporting.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Dataset     │────>│  Claude          │────>│  RAGAS         │
│  Loader      │     │  Evaluator       │     │  Scorer        │
│  (JSON)      │     │  (Anthropic API) │     │  (LLM-as-Judge)│
└──────────────┘     └──────────────────┘     └───────────────┘
       │                 │          │                  │
       │          ┌──────┴──────┐   │           ┌─────┴──────┐
       │          │  Langfuse   │   │           │  JSON      │
       │          │  Tracing    │   │           │  Export    │
       │          └─────────────┘   │           └────────────┘
       │                            │
       │                    ┌───────┴────────┐
       │                    │  Async Engine  │
       │                    │  (concurrent   │
       │                    │   API calls)   │
       │                    └────────────────┘
       │
  ┌────┴────────────────────────────────────┐
  │  Datasets                               │
  │  ai_engineering.json (10 Q&A pairs)     │
  │  sample_qa.json (3 Q&A pairs)           │
  └─────────────────────────────────────────┘
```

**Pipeline:** Load Q&A dataset -> Evaluate with Claude (sync or async) -> Score with RAGAS -> Export JSON report

## Results: Model Comparison

Evaluated 10 AI engineering questions (RAG, embeddings, evaluation, fine-tuning, observability, guardrails, latency optimization, agents) with rich context documents simulating real RAG retrieval.

### Scores

| Metric | Sonnet 4 | Haiku 4.5 |
|--------|----------|-----------|
| **Faithfulness** | 0.816 | 0.815 |
| **Context Precision** | 1.000 | 1.000 |
| **Context Recall** | 0.961 | 0.940 |

### Performance

| Metric | Sonnet 4 | Haiku 4.5 |
|--------|----------|-----------|
| **Avg Latency** | 6,671 ms | 3,122 ms |
| **Total Input Tokens** | 3,096 | 3,096 |
| **Total Output Tokens** | 3,293 | 3,097 |

### Analysis

- **Both models achieve comparable quality scores**, with Sonnet slightly ahead on context recall (0.961 vs 0.940)
- **Haiku is 2.1x faster** at roughly half the latency per request
- **Context precision is perfect (1.0)** for both models, meaning all retrieved context was relevant
- **Faithfulness ~0.82** indicates both models occasionally add accurate information beyond the provided context -- expected behavior for knowledge-intensive questions
- **Cost-quality trade-off favors Haiku** for this task, where speed matters more than marginal quality gains

## Stack

| Component | Purpose |
|-----------|---------|
| **Anthropic Claude API** | LLM inference (evaluation target + RAGAS judge) |
| **RAGAS** | Automated scoring (faithfulness, context precision, context recall) |
| **Langfuse** | Tracing and observability for LLM calls |
| **Pydantic** | Data validation and settings management |
| **asyncio** | Concurrent API calls with semaphore-based rate limiting |
| **GitHub Actions** | CI pipeline with linting and tests |

## Quick Start

### 1. Install

```bash
git clone https://github.com/aizaguirre3/llm-eval-harness.git
cd llm-eval-harness
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required:
- `ANTHROPIC_API_KEY` -- from [console.anthropic.com](https://console.anthropic.com)

Optional (for tracing):
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

### 3. Run

```bash
# Full pipeline: evaluate + score + export
python -m src.runner --dataset ai_engineering.json

# Concurrent evaluation (2x+ faster)
python -m src.runner --dataset ai_engineering.json --concurrent

# Evaluate only (skip RAGAS scoring)
python -m src.runner --skip-scoring

# Model comparison
python -m src.runner --dataset ai_engineering.json --model claude-sonnet-4-20250514
python -m src.runner --dataset ai_engineering.json --model claude-haiku-4-5-20251001

# Filter by category
python -m src.runner --category architecture

# Custom output path
python -m src.runner -o results/my_run.json

# List available datasets
python -m src.runner --list-datasets
```

## RAGAS Metrics

| Metric | What It Measures | Score Range |
|--------|-----------------|-------------|
| **Faithfulness** | Is the answer factually consistent with the provided context? | 0.0 - 1.0 |
| **Context Precision** | Is the retrieved context relevant to the question? | 0.0 - 1.0 |
| **Context Recall** | Does the context cover all information needed for the reference answer? | 0.0 - 1.0 |

## Dataset Format

Place JSON files in the `data/` directory:

```json
[
  {
    "id": "q1",
    "question": "What is retrieval-augmented generation?",
    "expected_answer": "RAG combines retrieval with generation...",
    "context": "Detailed context simulating RAG retrieval results...",
    "metadata": {
      "category": "architecture",
      "difficulty": "medium"
    }
  }
]
```

## Output Format

Results are exported to `results/` as JSON:

```json
{
  "metadata": {
    "timestamp": "2026-04-13T19:08:24",
    "model": "claude-sonnet-4-20250514",
    "total_questions": 10
  },
  "results": [
    {
      "id": "rag-001",
      "question": "How does RAG reduce hallucinations?",
      "expected_answer": "...",
      "actual_answer": "...",
      "latency_ms": 6317.0,
      "input_tokens": 207,
      "output_tokens": 241,
      "scores": {
        "faithfulness": 0.5,
        "context_precision": 1.0,
        "context_recall": 1.0
      }
    }
  ],
  "summary": {
    "context_precision": 1.0,
    "context_recall": 0.9607,
    "faithfulness": 0.8164,
    "avg_latency_ms": 6671.0,
    "total_input_tokens": 3096,
    "total_output_tokens": 3293
  }
}
```

## Testing

```bash
pytest -v

# 14 tests covering:
# - Dataset loading, filtering, and validation
# - Claude API evaluation (sync + mocked)
# - RAGAS scoring with NaN handling
# - Full pipeline orchestration
# - JSON export with summary statistics
```

## Project Structure

```
llm-eval-harness/
├── .github/workflows/ci.yml    # GitHub Actions CI (Python 3.11/3.12)
├── data/
│   ├── ai_engineering.json      # 10 AI engineering Q&A pairs
│   └── sample_qa.json           # 3 sample Q&A pairs
├── results/                     # JSON eval outputs (gitignored)
├── src/
│   ├── config.py                # Settings via pydantic-settings + dotenv
│   ├── runner.py                # CLI orchestrator with argparse
│   ├── loaders/
│   │   └── dataset.py           # Dataset loader + QAPair Pydantic model
│   ├── evaluators/
│   │   └── claude_evaluator.py  # Sync + async Claude evaluator + Langfuse tracing
│   ├── scorers/
│   │   └── ragas_scorer.py      # RAGAS scoring (3 metrics)
│   └── utils/
├── tests/
│   ├── test_dataset_loader.py   # 4 tests
│   ├── test_claude_evaluator.py # 2 tests
│   ├── test_ragas_scorer.py     # 3 tests
│   └── test_runner.py           # 4 tests
├── .env.example
├── .gitignore
└── pyproject.toml
```

## License

MIT
