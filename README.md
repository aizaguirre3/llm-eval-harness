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

## Results: Industry MLOps Evaluation

Evaluated 15 enterprise MLOps questions covering incident response, canary deployments, PII handling, autoscaling, disaster recovery, and more. Full results: **[RESULTS.md](RESULTS.md)**

### RAGAS Scores

| Metric | Sonnet 4 | Haiku 4.5 | Winner |
|--------|----------|-----------|--------|
| **Faithfulness** | 0.671 | **0.817** | Haiku |
| **Context Precision** | **1.000** | **1.000** | Tie |
| **Context Recall** | **1.000** | **1.000** | Tie |

### Performance

| Metric | Sonnet 4 | Haiku 4.5 | Delta |
|--------|----------|-----------|-------|
| **Avg Latency** | 5,015 ms | **2,412 ms** | 2.1x faster |
| **Total Output Tokens** | 3,361 | 3,202 | 5% fewer |

### Key Findings

- **Haiku outperforms Sonnet on faithfulness** (0.82 vs 0.67) for enterprise documentation Q&A
- **Haiku is 2.1x faster** with comparable quality -- clear winner for production use
- **Perfect context precision/recall** (1.0) across both models
- **Cost-quality trade-off strongly favors the smaller model** for this class of tasks

## Stack

| Component | Purpose |
|-----------|---------|
| **Anthropic Claude API** | LLM inference (evaluation target + RAGAS judge) |
| **RAGAS** | Automated scoring (faithfulness, context precision, context recall) |
| **Langfuse** | Tracing and observability for LLM calls |
| **Pydantic** | Data validation and settings management |
| **asyncio** | Concurrent API calls with semaphore-based rate limiting |
| **GitHub Actions** | CI pipeline with linting and tests |

## Features

- **RAGAS Scoring**: Automated evaluation using faithfulness, context precision, and context recall metrics
- **Model Comparison**: Run identical evaluations across models to make data-driven selection decisions
- **Regression Testing**: `--baseline` flag compares results against a previous run, flags score drops exceeding a threshold
- **Custom Metrics**: Extensible plugin system for user-defined scoring functions (exact match, keyword coverage, length ratio, or your own)
- **Streamlit Dashboard**: Interactive visualization with score heatmaps, latency charts, and model comparison
- **Concurrent Evaluation**: Async API calls with semaphore-based rate limiting for 2x+ throughput
- **Langfuse Tracing**: Full observability for every LLM call (optional)

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

# Regression testing against a baseline
python -m src.runner -d industry_mlops.json --baseline results/baseline.json

# Custom metrics (built-in or your own)
python -m src.runner -d industry_mlops.json --skip-scoring --custom-metrics builtin
python -m src.runner -d industry_mlops.json --skip-scoring --custom-metrics examples/my_metrics.py

# Streamlit dashboard
streamlit run dashboard.py
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

# 23 tests covering:
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
│   ├── industry_mlops.json      # 15 enterprise MLOps Q&A pairs
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
│   ├── regression.py            # Baseline comparison + regression detection
│   ├── scorers/
│   │   ├── ragas_scorer.py      # RAGAS scoring (3 metrics)
│   │   └── custom_metrics.py    # Extensible custom metric plugins
│   └── utils/
├── examples/
│   └── my_metrics.py            # Example custom metrics file
├── dashboard.py                 # Streamlit visualization dashboard
├── RESULTS.md                   # Full evaluation report with analysis
├── tests/                       # 23 tests
│   ├── test_dataset_loader.py
│   ├── test_claude_evaluator.py
│   ├── test_ragas_scorer.py
│   ├── test_runner.py
│   ├── test_regression.py
│   └── test_custom_metrics.py
├── .env.example
├── .gitignore
└── pyproject.toml
```

## License

MIT
