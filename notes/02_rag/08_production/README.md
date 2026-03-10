# 2.9–2.11 Production (Reliability, Cost, Observability)

> Keep your RAG system trustworthy after deployment — not just on day one.

## 📌 Key Lesson

A RAG system that works on day 1 can silently degrade over weeks. Index drift, stale data, and increasing latency are the biggest production risks. Observability, automated drift detection, and cost modeling are essential for production RAG.

## Learning Order

```
Reliability & drift (what goes wrong over time)
→ Cost, latency, stability (make it efficient)
→ Observability & debugging (see what's happening)
```

## Files

| #   | File                                                             | Topic                     | Key Concepts                                                                                                                                  |
| --- | ---------------------------------------------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | [01_reliability_drift.md](./01_reliability_drift.md)             | Reliability & Drift       | Index drift detection, data freshness monitoring, regression testing, silent degradation detection (multi-signal monitoring)                  |
| 02  | [02_cost_latency_stability.md](./02_cost_latency_stability.md)   | Cost, Latency & Stability | Token cost modeling, retrieval latency budgets, p95/p99 behavior, failure under load testing, cost optimization strategies                    |
| 03  | [03_observability_debugging.md](./03_observability_debugging.md) | Observability & Debugging | Retrieval trace logging, chunk contribution analysis, source attribution tracking, A/B testing, user feedback integration, quality dashboards |

## Popular Libraries

| Library          | Purpose                         | Notes                                          |
| ---------------- | ------------------------------- | ---------------------------------------------- |
| LangSmith        | Tracing, evaluation, monitoring | Full lifecycle management for LLM apps         |
| LangFuse         | Open-source LLM observability   | Traces, scoring, cost tracking                 |
| Arize Phoenix    | Open-source observability       | Trace retrieval, embeddings visualization      |
| Weights & Biases | Experiment tracking             | Track RAG experiments and evaluations          |
| OpenTelemetry    | Distributed tracing             | Standard for instrumenting retrieval pipelines |

## Common Questions

### Q: What's the biggest production risk for RAG systems?

**A:** **Silent degradation** — the system doesn't crash, it just gives increasingly worse answers. This happens because documents change, new topics appear that aren't in your index, and embedding model behavior drifts. Without monitoring, you won't know until users complain.

### Q: How much does a production RAG system cost?

**A:** Rough guidelines per 1M queries/month:

- **Embedding:** $20-100 (OpenAI) or $0 (self-hosted)
- **Vector DB:** $50-500 (managed) or ~$0 (self-hosted)
- **LLM generation:** $200-2000 (most of the cost)
- **Re-ranking:** $50-200 (if using paid API)

Most of the cost is in the LLM generation step. Optimize by reducing context size and using cheaper models for simple queries.

### Q: What latency should I target?

**A:** For interactive applications: **< 2 seconds end-to-end** (retrieval: < 200ms, re-ranking: < 200ms, generation: < 1.5s). For batch processing, latency matters less — optimize for cost instead.

### Q: Do I need all this observability from day 1?

**A:** No. Start with basic **retrieval trace logging** (log what was retrieved for each query). Add cost tracking and quality dashboards when you go to production. Add A/B testing and user feedback loops when you're optimizing.

## Syllabus Mapping

Maps to **§2.9–2.11** in `p2_rag_depth.md`:

- **§2.9** — Index drift, data freshness, silent degradation detection, regression testing
- **§2.10** — Token cost modeling, retrieval latency budgets, p95/p99 behavior, failure under load
- **§2.11** — Retrieval trace logging, chunk contribution analysis, source attribution tracking, A/B testing, user feedback integration, quality dashboards
