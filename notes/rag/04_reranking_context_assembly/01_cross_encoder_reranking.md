# Cross-Encoder Re-ranking

## Why It Matters

Bi-encoder retrieval (vector search) is fast but approximate — it independently encodes query and document, missing fine-grained interactions. **Cross-encoders** process query and document **together** through a transformer, capturing deep interactions between query terms and document terms. This gives dramatically better ranking but is too slow for initial retrieval. The pattern: **bi-encoder for recall, cross-encoder for precision**.

---

## Core Concept

```
BI-ENCODER (retrieval):
  Query  → Encoder → [q₁, q₂, ...] ─┐
                                       ├─ cosine similarity → 0.82
  Doc    → Encoder → [d₁, d₂, ...] ─┘

  ✅ Fast: encode once, search millions
  ❌ No interaction between query and doc tokens

CROSS-ENCODER (re-ranking):
  [CLS] Query tokens [SEP] Doc tokens [SEP] → Transformer → relevance score

  ✅ Deep interaction: each query token attends to each doc token
  ❌ Slow: must run for every (query, doc) pair

  ┌────────────────────────────────────────────────────┐
  │  HYBRID PIPELINE:                                   │
  │                                                      │
  │  Query → Bi-encoder → Top 100 docs (fast)           │
  │                         │                            │
  │                         ▼                            │
  │         Cross-encoder → Re-rank 100 → Top 5 (slow)  │
  │                                                      │
  │  Latency: ~10ms (retrieval) + ~50-200ms (re-rank)   │
  └────────────────────────────────────────────────────┘
```

---

## Simple Code — Cross-Encoder Re-ranking

```python
"""
Simple cross-encoder re-ranking using sentence-transformers.

Requirements: pip install sentence-transformers
"""

from sentence_transformers import CrossEncoder


def rerank(
    query: str,
    documents: list[str],
    top_k: int = 3,
) -> list[tuple[int, float, str]]:
    """
    Re-rank documents using a cross-encoder.
    Returns [(original_index, score, text)] sorted by relevance.
    """
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Score all (query, doc) pairs
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)

    # Sort by score descending
    ranked = sorted(
        enumerate(zip(scores, documents)),
        key=lambda x: x[1][0],
        reverse=True,
    )

    return [(idx, float(score), text) for idx, (score, text) in ranked[:top_k]]


# Example
query = "How to configure RBAC in Kubernetes?"
documents = [
    "Pod scheduling uses node selectors and affinity rules",
    "RBAC is configured using ClusterRole, Role, RoleBinding, and ClusterRoleBinding objects",
    "Kubernetes supports role-based access control for API authorization",
    "Docker containers run in isolated namespaces",
    "Configure kubectl credentials using kubeconfig files",
]

results = rerank(query, documents, top_k=3)

print(f"Query: {query}\n")
for idx, score, text in results:
    print(f"  [{score:.4f}] (was #{idx}) {text}")
```

---

## Production Code — Re-ranking Pipeline

```python
"""
Production re-ranking pipeline with batching, caching, and fallback.

Requirements: pip install sentence-transformers torch numpy
"""

import logging
import time
import hashlib
from dataclasses import dataclass
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    index: int
    text: str
    original_rank: int
    original_score: float
    reranked_score: float
    rank_change: int  # positive = moved up, negative = moved down


class ProductionReranker:
    """
    Cross-encoder re-ranker with batching and fallback.

    Models (speed vs quality):
    - cross-encoder/ms-marco-MiniLM-L-6-v2:  Fast, good quality
    - cross-encoder/ms-marco-MiniLM-L-12-v2: Slower, better quality
    - BAAI/bge-reranker-v2-m3:               Multilingual, best quality
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
    ):
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.batch_size = batch_size

    def rerank(
        self,
        query: str,
        documents: list[str],
        original_scores: list[float] | None = None,
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """
        Re-rank documents using cross-encoder.

        Args:
            query: User query
            documents: List of document texts to re-rank
            original_scores: Optional retrieval scores for comparison
            top_k: Number of results to return
        """
        if not documents:
            return []

        start = time.perf_counter()

        # Score all pairs in batches
        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        # Build ranked results
        indexed_scores = [(i, float(s)) for i, s in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for new_rank, (orig_idx, rerank_score) in enumerate(indexed_scores[:top_k]):
            results.append(RerankedResult(
                index=orig_idx,
                text=documents[orig_idx],
                original_rank=orig_idx + 1,  # 1-based
                original_score=original_scores[orig_idx] if original_scores else 0.0,
                reranked_score=rerank_score,
                rank_change=(orig_idx + 1) - (new_rank + 1),
            ))

        logger.info(
            f"Re-ranked {len(documents)} docs in {latency_ms:.0f}ms, "
            f"returned top {top_k}"
        )
        return results


# ─── Usage ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    reranker = ProductionReranker()

    query = "How to handle database connection pooling?"

    # Simulated retrieval results (in retrieval-score order)
    documents = [
        "Connection pool configuration requires setting max_connections",
        "Database backups should be performed nightly",
        "Use PgBouncer or pgpool-II for PostgreSQL connection pooling",
        "Pool size should match the number of application workers",
        "Monitor active connections with pg_stat_activity",
        "SQL injection can be prevented with parameterized queries",
        "Connection pooling reduces overhead by reusing established connections",
    ]
    retrieval_scores = [0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72]

    results = reranker.rerank(
        query=query,
        documents=documents,
        original_scores=retrieval_scores,
        top_k=5,
    )

    print(f"\nQuery: '{query}'\n")
    for r in results:
        direction = "↑" if r.rank_change > 0 else ("↓" if r.rank_change < 0 else "=")
        print(
            f"  [{r.reranked_score:+.4f}] "
            f"(was #{r.original_rank} {direction}{abs(r.rank_change)}) "
            f"{r.text}"
        )
```

---

## Cross-Encoder Models Comparison

```
┌──────────────────────────────────────────┬─────────┬────────┬──────┐
│ Model                                     │ Params  │ Speed  │ NDCG │
├──────────────────────────────────────────┼─────────┼────────┼──────┤
│ cross-encoder/ms-marco-MiniLM-L-6-v2     │ 22M     │ Fast   │ Good │
│ cross-encoder/ms-marco-MiniLM-L-12-v2    │ 33M     │ Medium │ Better│
│ BAAI/bge-reranker-v2-m3                  │ 568M    │ Slow   │ Best │
│ Cohere rerank-v3.5                       │ API     │ ~100ms │ Best │
│ Jina reranker-v2-base-multilingual       │ 278M    │ Medium │ Good │
└──────────────────────────────────────────┴─────────┴────────┴──────┘

Production recommendation:
  - Start with ms-marco-MiniLM-L-6-v2 (fast, free)
  - Move to Cohere/BGE-reranker if quality matters more than latency
```

---

## Pitfalls & Common Mistakes

| Mistake                                   | Impact                                   | Fix                                       |
| ----------------------------------------- | ---------------------------------------- | ----------------------------------------- |
| **Re-ranking all docs**                   | Too slow: O(n) cross-encoder calls       | Re-rank only top 20-100 from retrieval    |
| **Skipping re-ranking**                   | Retrieval noise enters the context       | Always re-rank before passing to LLM      |
| **Using re-ranker for initial retrieval** | ~1ms per doc ≈ 1000s for 1M docs         | Bi-encoder first, cross-encoder second    |
| **Ignoring max_length**                   | Long docs get truncated, losing key info | Ensure docs fit within model's max_length |
| **Not batching**                          | Sequential inference is slow             | Use batch_size=32+                        |

---

## Key Takeaways

1. **Cross-encoders dramatically improve ranking quality** — they see query-document interactions bi-encoders miss.
2. **Never use cross-encoders for initial retrieval** — too slow, O(n) complexity.
3. **The pattern: bi-encoder → top 50-100 → cross-encoder → top 5** is the production standard.
4. **Start with `ms-marco-MiniLM-L-6-v2`** — it's fast and free.
5. **Re-ranking typically adds 50-200ms** — budget for it in your latency SLA.
