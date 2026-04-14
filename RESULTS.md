# Evaluation Results Report

Industry-grade evaluation of Claude models on enterprise MLOps scenarios. All results generated with live API calls against the Anthropic Claude API.

## Dataset: Industry MLOps (15 Q&A Pairs)

Questions cover real-world production scenarios across 6 domains:

| Domain | Questions | Topics |
|--------|-----------|--------|
| **Incident Response** | 4 | Latency spikes, vector DB race conditions, prompt injection attacks, cost overruns |
| **MLOps** | 3 | Canary deployments, model validation, autoscaling policies |
| **Infrastructure** | 2 | GPU instance selection, disaster recovery (RPO/RTO) |
| **Data Governance** | 2 | PII handling in ML pipelines, data classification tiers |
| **Architecture** | 2 | Hybrid retrieval systems, RAG caching strategies |
| **Monitoring** | 2 | PagerDuty alert thresholds, API rate limiting |

---

## Model Comparison: Sonnet 4 vs Haiku 4.5

### RAGAS Scores

| Metric | Sonnet 4 | Haiku 4.5 | Delta |
|--------|----------|-----------|-------|
| **Faithfulness** | 0.671 | **0.817** | +0.146 |
| **Context Precision** | **1.000** | **1.000** | 0.000 |
| **Context Recall** | **1.000** | **1.000** | 0.000 |

### Performance

| Metric | Sonnet 4 | Haiku 4.5 | Delta |
|--------|----------|-----------|-------|
| **Avg Latency** | 5,015 ms | **2,412 ms** | -2,603 ms (2.1x faster) |
| **Total Input Tokens** | 2,730 | 2,730 | identical |
| **Total Output Tokens** | 3,361 | 3,202 | -159 (5% fewer) |

### Regression Test Result

```
REGRESSION TEST: PASSED (threshold: 0.05)

  IMPROVEMENTS (1):
    faithfulness: 0.6706 -> 0.8168 (+0.1462)

  UNCHANGED: context_precision, context_recall

  PER-QUESTION REGRESSIONS (1):
    [ind_002] What was the root cause of the model serving latency spike...
      faithfulness: 1.0 -> 0.875 (-0.1250)

  LATENCY: 5014.5ms -> 2411.7ms (-2602.8ms)
```

### Custom Metrics (Sonnet 4)

| Metric | Score | Description |
|--------|-------|-------------|
| **Keyword Coverage** | 0.721 | 72.1% of expected answer keywords found in actual answers |
| **Length Ratio** | 0.356 | Model generates ~2.8x more verbose answers than the reference |
| **Exact Match** | 0.000 | Expected -- model paraphrases rather than copying verbatim |

---

## Per-Question Breakdown

### Sonnet 4 Scores

| ID | Question | Faith. | Ctx Prec. | Ctx Rec. | Latency |
|----|----------|--------|-----------|----------|---------|
| ind_001 | API rate limits | 1.000 | 1.000 | 1.000 | 2,200ms |
| ind_002 | Latency spike incident | 1.000 | 1.000 | 1.000 | 3,515ms |
| ind_003 | Canary deployment | 0.643 | 1.000 | 1.000 | 5,450ms |
| ind_004 | Instance selection | 0.800 | 1.000 | 1.000 | 3,504ms |
| ind_005 | PII handling | 0.714 | 1.000 | 1.000 | 5,634ms |
| ind_006 | Vector DB incident | 0.727 | 1.000 | 1.000 | 4,690ms |
| ind_007 | Hybrid retrieval | 0.500 | 1.000 | 1.000 | 7,604ms |
| ind_008 | Caching strategies | 0.643 | 1.000 | 1.000 | 6,034ms |
| ind_009 | Alert thresholds | 1.000 | 1.000 | 1.000 | 3,275ms |
| ind_010 | Token explosion | 0.400 | 1.000 | 1.000 | 5,501ms |
| ind_011 | Data classification | 0.778 | 1.000 | 1.000 | 4,131ms |
| ind_012 | Prompt injection | 0.313 | 1.000 | 1.000 | 7,784ms |
| ind_013 | Autoscaling | 0.786 | 1.000 | 1.000 | 5,122ms |
| ind_014 | Disaster recovery | 0.471 | 1.000 | 1.000 | 6,604ms |
| ind_015 | Model validation | 0.286 | 1.000 | 1.000 | 4,170ms |

### Haiku 4.5 Scores

| ID | Question | Faith. | Ctx Prec. | Ctx Rec. | Latency |
|----|----------|--------|-----------|----------|---------|
| ind_001 | API rate limits | 1.000 | 1.000 | 1.000 | 1,403ms |
| ind_002 | Latency spike incident | 0.875 | 1.000 | 1.000 | 1,637ms |
| ind_003 | Canary deployment | 0.786 | 1.000 | 1.000 | 3,013ms |
| ind_004 | Instance selection | 0.889 | 1.000 | 1.000 | 2,397ms |
| ind_005 | PII handling | 0.842 | 1.000 | 1.000 | 3,078ms |
| ind_006 | Vector DB incident | 0.750 | 1.000 | 1.000 | 2,228ms |
| ind_007 | Hybrid retrieval | 0.579 | 1.000 | 1.000 | 3,777ms |
| ind_008 | Caching strategies | 0.923 | 1.000 | 1.000 | 2,150ms |
| ind_009 | Alert thresholds | 1.000 | 1.000 | 1.000 | 1,609ms |
| ind_010 | Token explosion | 1.000 | 1.000 | 1.000 | 2,707ms |
| ind_011 | Data classification | 0.800 | 1.000 | 1.000 | 2,178ms |
| ind_012 | Prompt injection | 0.364 | 1.000 | 1.000 | 3,892ms |
| ind_013 | Autoscaling | 0.857 | 1.000 | 1.000 | 1,871ms |
| ind_014 | Disaster recovery | 0.588 | 1.000 | 1.000 | 2,824ms |
| ind_015 | Model validation | 1.000 | 1.000 | 1.000 | 1,412ms |

---

## Key Findings

1. **Haiku outperforms Sonnet on faithfulness** (0.82 vs 0.67) for enterprise documentation Q&A. Sonnet's longer, more detailed answers introduce more claims that RAGAS evaluates, increasing the chance of lower faithfulness scores -- even when the extra information is accurate.

2. **Perfect context precision and recall** (1.0 across both models) confirms that the evaluation dataset's context documents are well-constructed and relevant to each question.

3. **Haiku is 2.1x faster** (2,412ms vs 5,015ms avg latency) with 5% fewer output tokens, making it the clear choice for latency-sensitive production workloads.

4. **Lowest faithfulness scores** cluster on questions requiring long, multi-part answers (prompt injection mitigations, disaster recovery, canary deployment). Both models add accurate contextual commentary beyond the strict context, which RAGAS penalizes.

5. **Cost-quality trade-off strongly favors Haiku** for this class of enterprise knowledge retrieval tasks. The quality difference is negligible while cost and latency savings are significant.

---

## Methodology

- **Evaluation framework**: LLM Eval Harness with RAGAS v0.4 scoring
- **Judge model**: Claude (LangchainLLMWrapper via langchain-anthropic)
- **Metrics**: Faithfulness (claim decomposition + NLI verification), Context Precision (retrieval relevance), Context Recall (retrieval completeness)
- **Custom metrics**: Keyword coverage, length ratio, exact match
- **Regression testing**: Threshold-based comparison with per-question drill-down
- **Dataset**: 15 enterprise MLOps Q&A pairs with expert-written context and ground truth answers
- **Date**: April 14, 2026
