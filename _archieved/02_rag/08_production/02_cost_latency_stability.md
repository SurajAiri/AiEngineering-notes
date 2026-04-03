# Cost, Latency & Stability

## Why It Matters

A RAG system that gives perfect answers but costs $5 per query and takes 10 seconds won't survive in production. Understanding the cost and latency profile of your pipeline is essential for building a sustainable system.

```
RAG PIPELINE COST & LATENCY BREAKDOWN:

  Query ──▶ Embed Query ──▶ Vector Search ──▶ Re-rank ──▶ Generate

  Latency:     50ms           20ms            200ms        800ms
  Cost:        $0.0001        ~$0              $0.001       $0.01

  ┌──────────────────────────────────────────────────────────┐
  │ Component       │ Latency  │ Cost/Query  │ Bottleneck?  │
  ├──────────────────┼──────────┼─────────────┼──────────────┤
  │ Query embedding │ 20-80ms  │ $0.0001     │ Rarely       │
  │ Vector search   │ 5-50ms   │ ~$0         │ At scale     │
  │ BM25 search     │ 5-20ms   │ ~$0         │ No           │
  │ Cross-encoder   │ 100-500ms│ $0.001      │ Sometimes    │
  │ LLM generation  │ 500-3000ms│ $0.01-0.10 │ USUALLY      │
  └──────────────────┴──────────┴─────────────┴──────────────┘

  → LLM generation dominates BOTH cost and latency.
  → Optimizing retrieval saves milliseconds.
  → Optimizing context size saves dollars.
```

---

## Token Cost Modeling

```python
"""
Model the token costs of a RAG pipeline.

No external dependencies needed.
"""

from dataclasses import dataclass


@dataclass
class CostProfile:
    """Cost per 1M tokens for a model."""
    model_name: str
    input_cost_per_1m: float    # $ per 1M input tokens
    output_cost_per_1m: float   # $ per 1M output tokens
    embedding_cost_per_1m: float = 0.0


# Common model costs (as of mid-2024, update as needed)
GPT4O = CostProfile("gpt-4o", input_cost_per_1m=2.50, output_cost_per_1m=10.00)
GPT4O_MINI = CostProfile("gpt-4o-mini", input_cost_per_1m=0.15, output_cost_per_1m=0.60)
CLAUDE_SONNET = CostProfile("claude-3.5-sonnet", input_cost_per_1m=3.00, output_cost_per_1m=15.00)
ADA_EMBEDDING = CostProfile("text-embedding-3-small", input_cost_per_1m=0.02, embedding_cost_per_1m=0.02)


def estimate_query_cost(
    num_context_chunks: int,
    avg_chunk_tokens: int = 200,
    system_prompt_tokens: int = 200,
    query_tokens: int = 30,
    output_tokens: int = 300,
    model: CostProfile = GPT4O_MINI,
    embedding_model: CostProfile = ADA_EMBEDDING,
    use_reranker: bool = False,
    reranker_model: CostProfile | None = None,
    num_rerank_candidates: int = 20,
) -> dict:
    """
    Estimate the cost of a single RAG query.

    Returns detailed cost breakdown.
    """
    # Embedding cost (query only — docs already embedded)
    embedding_cost = (query_tokens / 1_000_000) * embedding_model.embedding_cost_per_1m

    # Context tokens
    context_tokens = num_context_chunks * avg_chunk_tokens
    total_input = system_prompt_tokens + query_tokens + context_tokens

    # LLM cost
    input_cost = (total_input / 1_000_000) * model.input_cost_per_1m
    output_cost = (output_tokens / 1_000_000) * model.output_cost_per_1m

    # Reranker cost (cross-encoder processes query+doc pairs)
    reranker_cost = 0.0
    if use_reranker and reranker_model:
        # Each candidate: query + chunk scored by the model
        rerank_tokens = num_rerank_candidates * (query_tokens + avg_chunk_tokens)
        reranker_cost = (rerank_tokens / 1_000_000) * reranker_model.input_cost_per_1m

    total = embedding_cost + input_cost + output_cost + reranker_cost

    return {
        "embedding_cost": embedding_cost,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "reranker_cost": reranker_cost,
        "total_cost": total,
        "total_input_tokens": total_input,
        "total_output_tokens": output_tokens,
        "cost_per_1k_queries": total * 1000,
        "cost_per_100k_queries": total * 100_000,
    }


# ─── Usage ───
if __name__ == "__main__":
    # Scenario 1: Lean setup (gpt-4o-mini, 5 chunks)
    lean = estimate_query_cost(
        num_context_chunks=5,
        avg_chunk_tokens=200,
        model=GPT4O_MINI,
    )
    print("=== Lean Setup (gpt-4o-mini, 5 chunks) ===")
    print(f"  Cost per query: ${lean['total_cost']:.6f}")
    print(f"  Cost per 1K queries: ${lean['cost_per_1k_queries']:.4f}")
    print(f"  Cost per 100K queries: ${lean['cost_per_100k_queries']:.2f}")

    # Scenario 2: Premium setup (gpt-4o, 10 chunks, reranker)
    premium = estimate_query_cost(
        num_context_chunks=10,
        avg_chunk_tokens=200,
        model=GPT4O,
        use_reranker=True,
        reranker_model=GPT4O_MINI,
        num_rerank_candidates=20,
    )
    print("\n=== Premium Setup (gpt-4o, 10 chunks, reranker) ===")
    print(f"  Cost per query: ${premium['total_cost']:.6f}")
    print(f"  Cost per 1K queries: ${premium['cost_per_1k_queries']:.4f}")
    print(f"  Cost per 100K queries: ${premium['cost_per_100k_queries']:.2f}")

    # Scenario 3: Cost optimization — reduce chunks
    print("\n=== Impact of chunk count on cost (gpt-4o) ===")
    for chunks in [3, 5, 8, 10, 15, 20]:
        result = estimate_query_cost(num_context_chunks=chunks, model=GPT4O)
        print(f"  {chunks:2d} chunks: ${result['total_cost']:.5f}/query "
              f"({result['total_input_tokens']} input tokens)")
```

---

## Retrieval Latency Budgets

```
LATENCY BUDGET ALLOCATION:

  Total budget: 2000ms (typical for interactive use)

  ┌─────────────────────┬──────────┬───────────┐
  │ Component           │ Budget   │ Typical   │
  ├─────────────────────┼──────────┼───────────┤
  │ Query embedding     │ 100ms    │ 30-80ms   │
  │ Vector search       │ 100ms    │ 10-50ms   │
  │ BM25 search         │  50ms    │ 5-20ms    │
  │ Re-ranking          │ 300ms    │ 100-300ms │
  │ Context assembly    │  50ms    │ 10-30ms   │
  │ LLM generation      │ 1200ms   │ 800-2000ms│
  │ Network overhead    │ 200ms    │ 50-200ms  │
  └─────────────────────┴──────────┴───────────┘
  Total:                  2000ms

  IF OVER BUDGET:
  1. Stream LLM output (perceived latency drops dramatically)
  2. Cache query embeddings for common queries
  3. Use smaller re-ranker or skip for high-confidence results
  4. Reduce k (fewer chunks = faster search + generation)
```

### Latency Measurement

```python
"""
Measure and track RAG pipeline latency.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class LatencyTrace:
    """Records latency for each pipeline stage."""
    stages: dict[str, float] = field(default_factory=dict)  # stage → ms

    @contextmanager
    def track(self, stage_name: str):
        """Context manager to time a pipeline stage."""
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stages[stage_name] = elapsed_ms

    @property
    def total_ms(self) -> float:
        return sum(self.stages.values())

    def report(self) -> str:
        lines = ["Stage Latency:"]
        for stage, ms in self.stages.items():
            pct = (ms / self.total_ms * 100) if self.total_ms > 0 else 0
            bar = "█" * int(pct / 2)
            lines.append(f"  {stage:20s} {ms:7.1f}ms ({pct:4.1f}%) {bar}")
        lines.append(f"  {'TOTAL':20s} {self.total_ms:7.1f}ms")
        return "\n".join(lines)


class LatencyTracker:
    """Aggregate latency traces across queries for percentile analysis."""

    def __init__(self):
        self.traces: list[LatencyTrace] = []

    def add(self, trace: LatencyTrace):
        self.traces.append(trace)

    def percentiles(self) -> dict:
        """Compute p50, p95, p99 for total latency."""
        if not self.traces:
            return {}

        import numpy as np
        totals = sorted(t.total_ms for t in self.traces)
        return {
            "p50": np.percentile(totals, 50),
            "p95": np.percentile(totals, 95),
            "p99": np.percentile(totals, 99),
            "mean": np.mean(totals),
            "max": max(totals),
        }

    def stage_percentiles(self, stage: str) -> dict:
        """Compute percentiles for a specific stage."""
        import numpy as np
        values = sorted(
            t.stages.get(stage, 0) for t in self.traces if stage in t.stages
        )
        if not values:
            return {}
        return {
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
        }


# ─── Usage ───
trace = LatencyTrace()

with trace.track("embed_query"):
    time.sleep(0.03)  # simulate 30ms embedding

with trace.track("vector_search"):
    time.sleep(0.02)  # simulate 20ms search

with trace.track("rerank"):
    time.sleep(0.15)  # simulate 150ms reranking

with trace.track("llm_generation"):
    time.sleep(0.8)   # simulate 800ms generation

print(trace.report())
# Stage Latency:
#   embed_query           30.1ms ( 3.0%) █
#   vector_search         20.0ms ( 2.0%) █
#   rerank               150.2ms (15.0%) ███████
#   llm_generation       800.5ms (80.0%) ████████████████████████████████████████
#   TOTAL               1000.8ms
```

---

## p95 / p99 Behavior

```
WHY PERCENTILES MATTER MORE THAN AVERAGES:

  Mean latency: 500ms ← looks great!

  But the distribution might be:
    90% of queries: 200-400ms   ← fast
    5% of queries:  1000-2000ms ← slow
    1% of queries:  5000-10000ms ← user gives up

  p50 = 350ms   (median user experience)
  p95 = 1500ms  (1 in 20 users waits 1.5 seconds)
  p99 = 5000ms  (1 in 100 users waits 5 seconds)

  At 10,000 queries/day:
    100 users experience 5+ second latency EVERY DAY.

COMMON CAUSES OF TAIL LATENCY:

  1. Cold starts (first query after idle → load model)
  2. Large context (20 chunks → more LLM tokens → slower)
  3. Complex re-ranking (many candidates × cross-encoder)
  4. Network retries (timeout + retry = 2× latency)
  5. Garbage collection pauses
  6. Rate limiting / throttling from API providers
```

---

## Failure Under Load

```python
"""
Simulate and test RAG behavior under load.
"""

import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass


@dataclass
class LoadTestResult:
    total_queries: int
    successful: int
    failed: int
    latencies_ms: list[float]
    errors: list[str]

    @property
    def success_rate(self) -> float:
        return self.successful / self.total_queries if self.total_queries > 0 else 0

    def summary(self) -> str:
        p50 = statistics.median(self.latencies_ms) if self.latencies_ms else 0
        p95 = (
            sorted(self.latencies_ms)[int(len(self.latencies_ms) * 0.95)]
            if self.latencies_ms else 0
        )
        return (
            f"Load Test Results:\n"
            f"  Queries: {self.total_queries}\n"
            f"  Success rate: {self.success_rate:.1%}\n"
            f"  Latency p50: {p50:.0f}ms\n"
            f"  Latency p95: {p95:.0f}ms\n"
            f"  Errors: {len(self.errors)}"
        )


def load_test_rag(
    query_fn,  # callable that takes a query string → returns answer
    queries: list[str],
    concurrency: int = 10,
    timeout_ms: int = 5000,
) -> LoadTestResult:
    """
    Run concurrent queries against a RAG system.

    Args:
        query_fn: Function that accepts a query string and returns an answer.
        queries: List of queries to send.
        concurrency: Number of concurrent requests.
        timeout_ms: Maximum allowed latency per query.
    """
    latencies = []
    errors = []
    successful = 0
    failed = 0

    def run_query(query: str) -> tuple[bool, float, str]:
        start = time.perf_counter()
        try:
            result = query_fn(query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            if elapsed_ms > timeout_ms:
                return False, elapsed_ms, f"Timeout: {elapsed_ms:.0f}ms > {timeout_ms}ms"
            return True, elapsed_ms, ""
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return False, elapsed_ms, str(e)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(run_query, q): q for q in queries}
        for future in as_completed(futures):
            success, latency, error = future.result()
            latencies.append(latency)
            if success:
                successful += 1
            else:
                failed += 1
                if error:
                    errors.append(error)

    return LoadTestResult(
        total_queries=len(queries),
        successful=successful,
        failed=failed,
        latencies_ms=latencies,
        errors=errors,
    )


# Example usage (with a mock function):
def mock_rag_query(query: str) -> str:
    """Simulate RAG with variable latency."""
    import random
    time.sleep(random.uniform(0.1, 0.5))
    return f"Answer to: {query}"

result = load_test_rag(
    query_fn=mock_rag_query,
    queries=[f"Query {i}" for i in range(50)],
    concurrency=10,
    timeout_ms=3000,
)
print(result.summary())
```

---

## Cost Optimization Strategies

```
KEY LEVERS FOR COST REDUCTION:

  ┌─────────────────────────────────────────────────────────┐
  │ Strategy              │ Savings │ Trade-off              │
  ├───────────────────────┼─────────┼────────────────────────┤
  │ Use gpt-4o-mini       │ 90%+    │ Slightly lower quality │
  │ instead of gpt-4o     │         │                        │
  ├───────────────────────┼─────────┼────────────────────────┤
  │ Reduce chunk count    │ 20-50%  │ May miss context       │
  │ (10 → 5 chunks)      │         │                        │
  ├───────────────────────┼─────────┼────────────────────────┤
  │ Cache common queries  │ 30-60%  │ Stale cache risk       │
  ├───────────────────────┼─────────┼────────────────────────┤
  │ Smaller chunks        │ 10-30%  │ More chunks needed     │
  │ (500 → 200 tokens)   │         │                        │
  ├───────────────────────┼─────────┼────────────────────────┤
  │ Skip re-ranker for    │ 15-25%  │ Slightly worse ranking │
  │ high-confidence       │         │                        │
  ├───────────────────────┼─────────┼────────────────────────┤
  │ Quantized embeddings  │ Storage │ Slight recall drop     │
  │                       │ -75%    │                        │
  └───────────────────────┴─────────┴────────────────────────┘

  BIGGEST WIN: Switch to smaller LLM (gpt-4o → gpt-4o-mini).
  LLM generation is typically 80%+ of total cost.
```

---

## Pitfalls & Common Mistakes

| Mistake                                         | Impact                                        | Fix                                                   |
| ----------------------------------------------- | --------------------------------------------- | ----------------------------------------------------- |
| **Not tracking per-query costs**                | Can't optimize or budget                      | Log token counts and costs per query                  |
| **Optimizing retrieval latency instead of LLM** | Saving 20ms on search, ignoring 2000ms on LLM | Start with LLM optimization (model choice, streaming) |
| **Using mean instead of percentiles**           | Hides tail latency issues                     | Track p95 and p99, not just averages                  |
| **No load testing**                             | System breaks at 10× normal traffic           | Test at 2×, 5×, 10× expected load                     |
| **Caching without invalidation**                | Stale answers served from cache               | TTL-based cache with drift-aware invalidation         |
| **Fixed budget per query**                      | Some queries need more tokens, some need less | Dynamic token budget based on query complexity        |

---

## Key Takeaways

1. **LLM generation dominates cost and latency** — optimize there first.
2. **Track percentiles (p95, p99)**, not averages — tail latency destroys user experience.
3. **Model the cost per query** — know your budget before scaling.
4. **Streaming reduces perceived latency** dramatically even if total latency is the same.
5. **Load test at 5-10× expected traffic** — find breaking points before users do.
6. **The biggest cost lever is model choice** — gpt-4o-mini is ~17× cheaper than gpt-4o.
