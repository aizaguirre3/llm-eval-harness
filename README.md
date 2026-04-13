# LLM Eval Harness

A cloud-native evaluation harness for LLM responses, built with the Anthropic Claude API, RAGAS for automated scoring, and Langfuse for observability.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Dataset     │────>│  Claude          │────>│  RAGAS         │
│  Loader      │     │  Evaluator       │     │  Scorer        │
│  (JSON)      │     │  (Anthropic API) │     │  (5 metrics)   │
└──────────────┘     └──────────────────┘     └───────────────┘
                            │                        │
                     ┌──────┴──────┐          ┌──────┴──────┐
                     │  Langfuse   │          │  JSON       │
                     │  Tracing    │          │  Export     │
                     └─────────────┘          └─────────────┘
```

**Pipeline:** Load Q&A dataset → Send questions to Claude → Score responses with RAGAS → Export results as JSON

## Stack

| Component | Purpose |
|-----------|---------|
| **Anthropic Claude API** | LLM inference (evaluation target + RAGAS judge) |
| **RAGAS** | Automated scoring (faithfulness, relevancy, precision, recall, correctness) |
| **Langfuse** | Tracing and observability for LLM calls |
| **Pydantic** | Data validation and settings management |
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
- `ANTHROPIC_API_KEY` — from [console.anthropic.com](https://console.anthropic.com)

Optional (for tracing):
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`

### 3. Run

```bash
# Full pipeline: evaluate + score + export
python -m src.runner

# Evaluate only (skip RAGAS scoring)
python -m src.runner --skip-scoring

# Filter by category
python -m src.runner --category architecture

# Use a specific model
python -m src.runner --model claude-haiku-4-20250414

# Custom output path
python -m src.runner -o results/my_run.json

# List available datasets
python -m src.runner --list-datasets
```

## RAGAS Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Faithfulness** | Is the answer factually consistent with the provided context? |
| **Answer Relevancy** | Does the answer actually address the question asked? |
| **Context Precision** | Is the retrieved context relevant to the question? |
| **Context Recall** | Does the context cover all information needed for the reference answer? |
| **Factual Correctness** | Is the answer factually correct compared to the reference? |

## Dataset Format

Place JSON files in the `data/` directory:

```json
[
  {
    "id": "q1",
    "question": "What is retrieval-augmented generation?",
    "expected_answer": "RAG combines retrieval with generation...",
    "context": "Optional context for the question.",
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
    "timestamp": "2026-04-13T...",
    "model": "claude-sonnet-4-20250514",
    "total_questions": 3
  },
  "results": [
    {
      "id": "q1",
      "question": "...",
      "expected_answer": "...",
      "actual_answer": "...",
      "latency_ms": 850.2,
      "input_tokens": 45,
      "output_tokens": 120,
      "scores": {
        "faithfulness": 0.95,
        "answer_relevancy": 0.88,
        "context_precision": 0.91,
        "context_recall": 0.85,
        "factual_correctness": 0.92
      }
    }
  ],
  "summary": {
    "faithfulness": 0.93,
    "answer_relevancy": 0.87,
    "avg_latency_ms": 920.5,
    "total_input_tokens": 135,
    "total_output_tokens": 360
  }
}
```

## Testing

```bash
pytest -v
```

## Project Structure

```
llm-eval-harness/
├── .github/workflows/ci.yml   # GitHub Actions CI
├── data/
│   └── sample_qa.json          # Sample Q&A dataset
├── results/                    # JSON eval outputs (gitignored)
├── src/
│   ├── config.py               # Settings via pydantic-settings + dotenv
│   ├── runner.py               # CLI orchestrator
│   ├── loaders/
│   │   └── dataset.py          # Dataset loader + QAPair model
│   ├── evaluators/
│   │   └── claude_evaluator.py # Claude API evaluator + Langfuse tracing
│   ├── scorers/
│   │   └── ragas_scorer.py     # RAGAS scoring (5 metrics)
│   └── utils/
├── tests/
│   ├── test_dataset_loader.py
│   ├── test_claude_evaluator.py
│   ├── test_ragas_scorer.py
│   └── test_runner.py
├── .env.example
├── .gitignore
└── pyproject.toml
```

## License

MIT
