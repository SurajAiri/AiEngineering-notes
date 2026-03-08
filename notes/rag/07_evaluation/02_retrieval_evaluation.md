# Retrieval Evaluation — Recall@k, MRR, nDCG

## Why It Matters

Retrieval is the **foundation** of RAG — if the right chunks aren't retrieved, no amount of prompt engineering or fine-tuning will fix the answer. Retrieval metrics tell you **objectively** whether your retriever is finding what it should.

```
THE THREE CORE RETRIEVAL METRICS:

  ┌───────────┬────────────────────────────────────────────┐
  │ Metric    │ Question It Answers                        │
  ├───────────┼────────────────────────────────────────────┤
  │ Recall@k  │ "Did we FIND all the relevant docs?"       │
  │           │ (completeness)                             │
  ├───────────┼────────────────────────────────────────────┤
  │ MRR       │ "How HIGH did the first relevant doc rank?"│
  │           │ (position of first hit)                    │
  ├───────────┼────────────────────────────────────────────┤
  │ nDCG      │ "Are ALL relevant docs ranked WELL?"       │
  │           │ (overall ranking quality)                  │
  └───────────┴────────────────────────────────────────────┘
```

---

## Recall@k

```
RECALL@k = (# relevant docs in top-k results) / (total # relevant docs)

EXAMPLE:
  3 relevant docs exist: [A, B, C]
  Retrieved top-5: [A, X, B, Y, Z]

  Recall@1 = 1/3 = 0.33  (only A found in top 1)
  Recall@3 = 2/3 = 0.67  (A and B found in top 3)
  Recall@5 = 2/3 = 0.67  (still only A and B; C not in top 5)

  INTERPRETATION:
  - Recall@5 = 0.67 means we're missing 1 of 3 relevant docs.
  - For RAG, Recall@10 ≥ 0.9 is a good target.
  - Low recall = LLM will miss information.
```

---

## Mean Reciprocal Rank (MRR)

```
MRR = Average of (1 / rank_of_first_relevant_doc) across queries.

EXAMPLE:
  Query 1: First relevant doc at position 1 → 1/1 = 1.0
  Query 2: First relevant doc at position 3 → 1/3 = 0.33
  Query 3: First relevant doc at position 2 → 1/2 = 0.5

  MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61

  INTERPRETATION:
  - MRR = 1.0 → First result is always relevant (perfect).
  - MRR = 0.5 → On average, first relevant result is around position 2.
  - MRR only cares about the FIRST relevant result, not all of them.

  MRR IS GOOD FOR:
  - "Is the top result useful?" questions
  - Single-answer factual queries

  MRR IS NOT GOOD FOR:
  - Queries needing MULTIPLE relevant docs (use nDCG or Recall)
```

---

## Normalized Discounted Cumulative Gain (nDCG)

```
nDCG considers BOTH relevance AND position.

STEP 1: Discounted Cumulative Gain (DCG)
  DCG@k = Σ (relevance_i / log₂(i + 1)) for i = 1 to k

  Position 1: relevance / log₂(2) = rel / 1.0   ← highest weight
  Position 2: relevance / log₂(3) = rel / 1.58
  Position 3: relevance / log₂(4) = rel / 2.0
  Position 5: relevance / log₂(6) = rel / 2.58  ← much lower weight

STEP 2: Ideal DCG (IDCG) — the perfect ranking
  Sort all relevant docs by relevance, compute DCG.

STEP 3: nDCG = DCG / IDCG  (normalized to 0-1 scale)

EXAMPLE with binary relevance (0 or 1):
  Retrieved:  [Relevant, Irrelevant, Relevant, Irrelevant, Relevant]
  Relevance:  [1,        0,          1,        0,          1        ]

  DCG@5 = 1/1.0 + 0/1.58 + 1/2.0 + 0/2.32 + 1/2.58
        = 1.0 + 0 + 0.5 + 0 + 0.39 = 1.89

  Ideal:      [Relevant, Relevant, Relevant, -, -]  (all relevant first)
  IDCG@5 = 1/1.0 + 1/1.58 + 1/2.0 = 1.0 + 0.63 + 0.5 = 2.13

  nDCG@5 = 1.89 / 2.13 = 0.89  (good, but not perfect ordering)
```

---

## Implementation — All Metrics

```python
"""
Retrieval evaluation metrics: Recall@k, MRR, nDCG.

Requirements: pip install numpy
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class EvalQuery:
    """A single evaluation query with ground truth."""
    query: str
    relevant_doc_ids: set[str]
    retrieved_doc_ids: list[str]  # ordered by rank


def recall_at_k(eval_query: EvalQuery, k: int) -> float:
    """
    Recall@k: fraction of relevant docs found in top-k.

    Returns 1.0 if all relevant docs are in top-k,
    0.0 if none are found.
    """
    if not eval_query.relevant_doc_ids:
        return 1.0  # no relevant docs to find

    top_k = set(eval_query.retrieved_doc_ids[:k])
    found = top_k & eval_query.relevant_doc_ids
    return len(found) / len(eval_query.relevant_doc_ids)


def precision_at_k(eval_query: EvalQuery, k: int) -> float:
    """
    Precision@k: fraction of top-k results that are relevant.
    """
    top_k = eval_query.retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    relevant_in_k = sum(1 for doc in top_k if doc in eval_query.relevant_doc_ids)
    return relevant_in_k / len(top_k)


def reciprocal_rank(eval_query: EvalQuery) -> float:
    """
    Reciprocal rank: 1/position of first relevant result.
    Returns 0 if no relevant result found.
    """
    for i, doc_id in enumerate(eval_query.retrieved_doc_ids):
        if doc_id in eval_query.relevant_doc_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(eval_query: EvalQuery, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ k.
    Uses binary relevance (0 or 1).
    """
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(eval_query.retrieved_doc_ids[:k]):
        rel = 1.0 if doc_id in eval_query.relevant_doc_ids else 0.0
        dcg += rel / np.log2(i + 2)  # i+2 because log₂(1) = 0

    # Ideal DCG: all relevant docs at top
    num_relevant = min(len(eval_query.relevant_doc_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg


# ─── Aggregate metrics across queries ───

def evaluate_retriever(eval_queries: list[EvalQuery], k: int = 5) -> dict:
    """Compute all metrics across a set of evaluation queries."""
    recalls = [recall_at_k(q, k) for q in eval_queries]
    precisions = [precision_at_k(q, k) for q in eval_queries]
    rrs = [reciprocal_rank(q) for q in eval_queries]
    ndcgs = [ndcg_at_k(q, k) for q in eval_queries]

    return {
        f"recall@{k}": np.mean(recalls),
        f"precision@{k}": np.mean(precisions),
        "mrr": np.mean(rrs),
        f"ndcg@{k}": np.mean(ndcgs),
        "num_queries": len(eval_queries),
    }


# ─── Example ───
if __name__ == "__main__":
    eval_set = [
        EvalQuery(
            query="What is the API rate limit?",
            relevant_doc_ids={"doc_42", "doc_87"},
            retrieved_doc_ids=["doc_42", "doc_15", "doc_87", "doc_33", "doc_9"],
        ),
        EvalQuery(
            query="How to configure SSL?",
            relevant_doc_ids={"doc_5", "doc_12", "doc_20"},
            retrieved_doc_ids=["doc_30", "doc_5", "doc_12", "doc_99", "doc_20"],
        ),
        EvalQuery(
            query="Default timeout value?",
            relevant_doc_ids={"doc_7"},
            retrieved_doc_ids=["doc_7", "doc_8", "doc_9", "doc_10", "doc_11"],
        ),
    ]

    # Evaluate at k=5
    results = evaluate_retriever(eval_set, k=5)
    print("=== Retrieval Evaluation (k=5) ===")
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.3f}")
        else:
            print(f"  {metric}: {value}")

    # Evaluate at multiple k values
    print("\n=== Recall at different k ===")
    for k in [1, 3, 5, 10]:
        recalls = [recall_at_k(q, k) for q in eval_set]
        print(f"  Recall@{k}: {np.mean(recalls):.3f}")
```

---

## Coverage vs Precision Tradeoffs

```
THE FUNDAMENTAL TENSION:

  Retrieve MORE (higher k):
    ✅ Higher recall — find more relevant docs
    ❌ Lower precision — more noise for LLM to handle
    ❌ Higher cost — more tokens in context
    ❌ Risk of over-retrieval hallucination

  Retrieve LESS (lower k):
    ✅ Higher precision — cleaner context
    ✅ Lower cost — fewer tokens
    ❌ Lower recall — may miss relevant docs
    ❌ Risk of missing context hallucination

  TYPICAL TRADEOFF CURVE:

  Recall@k
  1.0 ┤                  ●─────────●
      │              ●
      │          ●
  0.5 ┤      ●
      │   ●
      │ ●
  0.0 ┼─●───┬───┬───┬───┬───┬───┬──
      0   1   3   5  10  20  50  k

  Precision@k
  1.0 ┤●
      │  ●
      │    ●
  0.5 ┤      ●
      │          ●
      │              ●──────●─────●
  0.0 ┼───┬───┬───┬───┬───┬───┬──
      0   1   3   5  10  20  50  k

  SWEET SPOT: Usually k=5 to k=10 for most RAG systems.
```

---

## Hybrid Retrieval Attribution

When using hybrid retrieval (vector + BM25), you need to know which retrieval mode is contributing.

```python
"""
Evaluate hybrid retrieval: which strategy finds what.
"""


@dataclass
class HybridEvalQuery:
    query: str
    relevant_doc_ids: set[str]
    vector_retrieved: list[str]
    bm25_retrieved: list[str]
    hybrid_retrieved: list[str]  # after fusion


def hybrid_attribution(queries: list[HybridEvalQuery], k: int = 5) -> dict:
    """
    Analyze which retrieval strategy contributes relevant results.
    """
    vector_only_hits = 0      # found by vector, not BM25
    bm25_only_hits = 0        # found by BM25, not vector
    both_hits = 0             # found by both
    missed_by_both = 0        # relevant but not found by either
    total_relevant = 0

    for q in queries:
        top_vector = set(q.vector_retrieved[:k])
        top_bm25 = set(q.bm25_retrieved[:k])

        for doc_id in q.relevant_doc_ids:
            total_relevant += 1
            in_vector = doc_id in top_vector
            in_bm25 = doc_id in top_bm25

            if in_vector and in_bm25:
                both_hits += 1
            elif in_vector:
                vector_only_hits += 1
            elif in_bm25:
                bm25_only_hits += 1
            else:
                missed_by_both += 1

    return {
        "total_relevant_docs": total_relevant,
        "found_by_both": both_hits,
        "vector_only": vector_only_hits,
        "bm25_only": bm25_only_hits,
        "missed_by_both": missed_by_both,
        "vector_contribution": (vector_only_hits + both_hits) / max(total_relevant, 1),
        "bm25_contribution": (bm25_only_hits + both_hits) / max(total_relevant, 1),
        "hybrid_advantage": bm25_only_hits + vector_only_hits,
    }


# Example
queries = [
    HybridEvalQuery(
        query="API rate limit configuration",
        relevant_doc_ids={"doc_1", "doc_2"},
        vector_retrieved=["doc_1", "doc_5", "doc_9"],
        bm25_retrieved=["doc_2", "doc_1", "doc_7"],
        hybrid_retrieved=["doc_1", "doc_2", "doc_5"],
    ),
]

result = hybrid_attribution(queries, k=3)
print(result)
# Shows: doc_1 found by both, doc_2 found by BM25 only
# → BM25 adds unique value in this case
```

---

## Evaluation Reporting

```python
"""
Generate a readable evaluation report comparing retrieval strategies.
"""


def evaluation_report(
    strategy_name: str,
    eval_queries: list[EvalQuery],
    k_values: list[int] = [1, 3, 5, 10],
) -> str:
    """Generate a markdown-formatted evaluation report."""
    lines = [f"## Retrieval Evaluation: {strategy_name}", ""]
    lines.append(f"**Queries evaluated:** {len(eval_queries)}")
    lines.append("")

    # Table header
    lines.append("| Metric | " + " | ".join(f"k={k}" for k in k_values) + " |")
    lines.append("|--------|" + "|".join("------" for _ in k_values) + "|")

    # Recall
    recall_row = "| Recall | "
    for k in k_values:
        val = np.mean([recall_at_k(q, k) for q in eval_queries])
        recall_row += f"{val:.3f} | "
    lines.append(recall_row)

    # Precision
    prec_row = "| Precision | "
    for k in k_values:
        val = np.mean([precision_at_k(q, k) for q in eval_queries])
        prec_row += f"{val:.3f} | "
    lines.append(prec_row)

    # nDCG
    ndcg_row = "| nDCG | "
    for k in k_values:
        val = np.mean([ndcg_at_k(q, k) for q in eval_queries])
        ndcg_row += f"{val:.3f} | "
    lines.append(ndcg_row)

    # MRR (same across k)
    mrr_val = np.mean([reciprocal_rank(q) for q in eval_queries])
    lines.append(f"\n**MRR:** {mrr_val:.3f}")

    return "\n".join(lines)
```

---

## Pitfalls & Common Mistakes

| Mistake                                 | Impact                                              | Fix                                             |
| --------------------------------------- | --------------------------------------------------- | ----------------------------------------------- |
| **Only measuring Recall@5**             | Ignoring ranking quality                            | Also track MRR and nDCG                         |
| **Not tracking per-difficulty metrics** | High aggregate score hides failures on hard queries | Report metrics stratified by difficulty         |
| **Inconsistent k values**               | Can't compare experiments                           | Standardize on k=5 and k=10                     |
| **Binary relevance only**               | Treats "somewhat relevant" same as "perfect match"  | Consider graded relevance (0, 1, 2, 3) for nDCG |
| **Evaluating on too few queries**       | Noisy results, not statistically significant        | Use 50+ queries minimum                         |
| **Ignoring precision**                  | Recall is high but LLM gets noisy context           | Track Precision@k alongside Recall@k            |

---

## Key Takeaways

1. **Recall@k** = did we find all relevant docs? Most important for RAG coverage.
2. **MRR** = is the first relevant result near the top? Important for single-answer queries.
3. **nDCG** = is the overall ranking good? Best holistic metric.
4. **Use multiple metrics** — no single number tells the full story.
5. **Evaluate at multiple k** — Recall@5 vs Recall@10 shows retrieval depth.
6. **Track per-difficulty** — aggregate scores can hide failures on hard queries.
7. **The sweet spot for k is usually 5-10** — balance recall vs noise.

---

## Popular Libraries

### Quick Example — RAGAS Retrieval Metrics

```python
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from ragas import EvaluationDataset

# Prepare your evaluation data
eval_data = EvaluationDataset.from_list([
    {
        "user_input": "What is photosynthesis?",
        "retrieved_contexts": ["Photosynthesis converts sunlight to energy..."],
        "reference": "Photosynthesis is the process by which plants convert light into chemical energy.",
        "response": "Photosynthesis is how plants use sunlight to make food.",
    },
    # ... more examples
])

# Evaluate retrieval quality
results = evaluate(
    dataset=eval_data,
    metrics=[context_precision, context_recall],
)
print(results)  # {'context_precision': 0.85, 'context_recall': 0.92}
```

---

## Common Questions

### Q: Which retrieval metric should I prioritize?

**A:** **Recall@k** is the most important for RAG. If relevant documents don't appear in your top-k results, the LLM can't generate a correct answer no matter how good it is. After recall is acceptable (>0.8), optimize for **nDCG** (ranking quality) to reduce noise in the context window.

### Q: What's a good Recall@5 score?

**A:** >0.85 for production systems. <0.7 means your retrieval is actively hurting answer quality. Between 0.7-0.85, you should be investigating failure cases and adding hybrid search, query rewriting, or better embeddings.
