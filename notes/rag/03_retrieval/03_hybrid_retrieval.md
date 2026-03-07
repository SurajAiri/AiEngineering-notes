# Hybrid Retrieval

## Why It Matters

Neither vector search nor BM25 alone captures everything. **Hybrid retrieval** combines both to get the best of semantic understanding AND exact keyword matching. This is the **recommended default** for production RAG.

---

## Core Concept

```
                    ┌──────────────┐
       Query ──────►│   BM25       │──► keyword results + scores
         │         └──────────────┘
         │
         │         ┌──────────────┐
         └────────►│ Vector Search│──► semantic results + scores
                   └──────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │    FUSION    │ ← Combine & re-rank
                   └──────────────┘
                          │
                          ▼
                   Final ranked results
```

### Why You Need Both

```
Query: "How to fix ERR_CONNECTION_REFUSED in the auth service?"

BM25 finds:    "ERR_CONNECTION_REFUSED is raised when the auth service
                cannot reach the downstream identity provider on port 443."
                → Matched exact error code ✅

Vector finds:  "When the authentication module fails to connect to the
                identity provider, it typically indicates a network
                configuration issue or firewall rule blocking outbound traffic."
                → Matched meaning ✅, but missed the error code ❌

HYBRID:        Returns BOTH — exact match + semantic context.
               Together they give the LLM everything it needs.
```

---

## Fusion Strategies

```
┌──────────────────────────────────────────────────────────────────┐
│  STRATEGY              │ HOW                  │ WHEN TO USE      │
├────────────────────────┼──────────────────────┼──────────────────┤
│ Reciprocal Rank Fusion │ 1/(k+rank) per list  │ Default. Simple, │
│ (RRF)                  │ sum across lists      │ works well       │
├────────────────────────┼──────────────────────┼──────────────────┤
│ Weighted Linear        │ α×vec + (1-α)×bm25   │ When you have    │
│ Combination            │ after normalizing     │ tuned weights    │
├────────────────────────┼──────────────────────┼──────────────────┤
│ Re-ranker Fusion       │ Cross-encoder on      │ Best quality,    │
│                        │ union of results      │ highest latency  │
└────────────────────────┴──────────────────────┴──────────────────┘
```

---

## Simple Code — Reciprocal Rank Fusion (RRF)

```python
"""
Reciprocal Rank Fusion — the simplest and most effective
way to combine two ranked lists.

No external dependencies needed.
"""


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists using RRF.

    Each document gets score = Σ 1/(k + rank) across all lists.
    k=60 is the standard constant from the original paper.

    Args:
        ranked_lists: List of ranked document ID lists
        k: Smoothing constant (default 60)

    Returns:
        Sorted list of (doc_id, fused_score)
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Example
bm25_results = ["doc_4", "doc_1", "doc_7", "doc_3", "doc_9"]  # BM25 ranking
vector_results = ["doc_1", "doc_7", "doc_5", "doc_4", "doc_2"]  # Vector ranking

fused = reciprocal_rank_fusion([bm25_results, vector_results], k=60)

print("Fused Results:")
for doc_id, score in fused[:5]:
    bm25_rank = bm25_results.index(doc_id) + 1 if doc_id in bm25_results else "—"
    vec_rank = vector_results.index(doc_id) + 1 if doc_id in vector_results else "—"
    print(f"  {doc_id}: score={score:.6f}  (BM25 rank: {bm25_rank}, Vec rank: {vec_rank})")
```

---

## Production Code — Full Hybrid Retrieval Pipeline

```python
"""
Production hybrid retrieval combining BM25 + Vector Search with RRF fusion.

Requirements: pip install sentence-transformers rank_bm25 faiss-cpu numpy
"""

import re
import logging
import numpy as np
import faiss
from dataclasses import dataclass
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "but", "or", "not", "it", "its", "this", "that",
}


@dataclass
class HybridResult:
    """A single search result from hybrid retrieval."""
    doc_id: int
    text: str
    hybrid_score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    bm25_score: float = 0.0
    vector_score: float = 0.0


class HybridRetriever:
    """
    Hybrid retrieval engine combining BM25 + dense vector search.

    Fusion strategies:
    - "rrf": Reciprocal Rank Fusion (default, recommended)
    - "weighted": Weighted linear combination (requires score normalization)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        fusion: str = "rrf",
        rrf_k: int = 60,
        vector_weight: float = 0.5,  # for weighted fusion
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.fusion = fusion
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight

        self.texts: list[str] = []
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.IndexFlatIP | None = None
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

    def index(self, texts: list[str], batch_size: int = 64):
        """Build both BM25 and vector indexes."""
        self.texts = texts

        # Build BM25 index
        tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized, k1=self.bm25_k1, b=self.bm25_b)

        # Build FAISS index
        embeddings = self.model.encode(
            texts, batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        self.faiss_index = faiss.IndexFlatIP(self.dimension)
        self.faiss_index.add(embeddings)

        logger.info(
            f"Hybrid index built: {len(texts)} docs, "
            f"BM25 + FAISS({self.dimension}d)"
        )

    def search(
        self,
        query: str,
        k: int = 5,
        bm25_k: int = 20,
        vector_k: int = 20,
    ) -> list[HybridResult]:
        """
        Search with hybrid retrieval.

        Retrieves bm25_k and vector_k candidates respectively,
        then fuses to produce final top-k results.
        """
        if self.bm25 is None or self.faiss_index is None:
            raise RuntimeError("Index not built. Call index() first.")

        # BM25 search
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        bm25_top = np.argsort(bm25_scores)[::-1][:bm25_k]
        bm25_ranked = [(int(i), float(bm25_scores[i])) for i in bm25_top if bm25_scores[i] > 0]

        # Vector search
        query_vec = self.model.encode(query, normalize_embeddings=True)
        query_vec = np.array([query_vec], dtype=np.float32)
        distances, indices = self.faiss_index.search(query_vec, vector_k)
        vector_ranked = [
            (int(idx), float(dist))
            for dist, idx in zip(distances[0], indices[0])
            if idx != -1
        ]

        # Fuse
        if self.fusion == "rrf":
            return self._rrf_fusion(bm25_ranked, vector_ranked, k)
        elif self.fusion == "weighted":
            return self._weighted_fusion(bm25_ranked, vector_ranked, k)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion}")

    def _rrf_fusion(
        self,
        bm25_ranked: list[tuple[int, float]],
        vector_ranked: list[tuple[int, float]],
        k: int,
    ) -> list[HybridResult]:
        """Reciprocal Rank Fusion."""
        scores: dict[int, float] = {}
        bm25_rank_map: dict[int, int] = {}
        vector_rank_map: dict[int, int] = {}
        bm25_score_map: dict[int, float] = {}
        vector_score_map: dict[int, float] = {}

        for rank, (doc_id, score) in enumerate(bm25_ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            bm25_rank_map[doc_id] = rank
            bm25_score_map[doc_id] = score

        for rank, (doc_id, score) in enumerate(vector_ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            vector_rank_map[doc_id] = rank
            vector_score_map[doc_id] = score

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:k]

        return [
            HybridResult(
                doc_id=doc_id,
                text=self.texts[doc_id],
                hybrid_score=scores[doc_id],
                bm25_rank=bm25_rank_map.get(doc_id),
                vector_rank=vector_rank_map.get(doc_id),
                bm25_score=bm25_score_map.get(doc_id, 0.0),
                vector_score=vector_score_map.get(doc_id, 0.0),
            )
            for doc_id in sorted_ids
        ]

    def _weighted_fusion(
        self,
        bm25_ranked: list[tuple[int, float]],
        vector_ranked: list[tuple[int, float]],
        k: int,
    ) -> list[HybridResult]:
        """Weighted linear combination with min-max normalization."""
        # Normalize BM25 scores to [0, 1]
        bm25_scores = {doc_id: score for doc_id, score in bm25_ranked}
        vec_scores = {doc_id: score for doc_id, score in vector_ranked}

        def normalize(scores: dict[int, float]) -> dict[int, float]:
            if not scores:
                return {}
            min_s = min(scores.values())
            max_s = max(scores.values())
            rng = max_s - min_s
            if rng == 0:
                return {k: 1.0 for k in scores}
            return {k: (v - min_s) / rng for k, v in scores.items()}

        norm_bm25 = normalize(bm25_scores)
        norm_vec = normalize(vec_scores)

        all_ids = set(norm_bm25.keys()) | set(norm_vec.keys())
        combined = {}
        for doc_id in all_ids:
            combined[doc_id] = (
                (1 - self.vector_weight) * norm_bm25.get(doc_id, 0.0) +
                self.vector_weight * norm_vec.get(doc_id, 0.0)
            )

        sorted_ids = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)[:k]

        return [
            HybridResult(
                doc_id=doc_id,
                text=self.texts[doc_id],
                hybrid_score=combined[doc_id],
                bm25_score=bm25_scores.get(doc_id, 0.0),
                vector_score=vec_scores.get(doc_id, 0.0),
            )
            for doc_id in sorted_ids
        ]

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if t not in STOPWORDS]


# ─── Usage ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    retriever = HybridRetriever(
        model_name="all-MiniLM-L6-v2",
        fusion="rrf",
    )

    documents = [
        "RBAC (Role-Based Access Control) manages permissions in Kubernetes",
        "Container orchestration enables automatic scaling of workloads",
        "The ERR_CONNECTION_REFUSED error occurs when a service port is unreachable",
        "Horizontal Pod Autoscaler adjusts replica count based on CPU metrics",
        "Network policies control ingress and egress traffic between pods",
        "Set max_connections=100 in the database configuration file",
        "Service mesh provides mTLS encryption for inter-service communication",
        "Pod disruption budgets prevent too many pods from being terminated simultaneously",
    ]

    retriever.index(documents)

    # Query that benefits from both BM25 and vector search
    query = "ERR_CONNECTION_REFUSED when connecting between services"
    results = retriever.search(query, k=5)

    print(f"\nQuery: '{query}'\n")
    for r in results:
        print(f"  [{r.hybrid_score:.6f}] (BM25 rank: {r.bm25_rank or '—'}, "
              f"Vec rank: {r.vector_rank or '—'})")
        print(f"    {r.text}\n")
```

---

## RRF vs Weighted Fusion — When to Use Which

```
RRF (Reciprocal Rank Fusion)
────────────────────────────
✅ No score normalization needed (rank-based)
✅ Robust — works well without tuning
✅ Used by Elasticsearch, most vector DBs
❌ Ignores score magnitudes (rank 1 with score 0.99 ≈ rank 1 with 0.51)

Weighted Linear Combination
────────────────────────────
✅ Uses actual scores, more nuanced
✅ Tunable weight parameter (α)
❌ Requires score normalization (BM25 and cosine have different scales)
❌ Sensitive to normalization method
❌ Needs tuning per dataset

RECOMMENDATION: Start with RRF. Switch to weighted only if you have
evaluation data showing it performs better for your specific use case.
```

---

## Pitfalls & Common Mistakes

| Mistake                                      | Impact                               | Fix                                        |
| -------------------------------------------- | ------------------------------------ | ------------------------------------------ |
| **Using only vector search**                 | Miss exact matches for codes/IDs     | Always add BM25                            |
| **Not retrieving enough candidates**         | Fusion has too few docs to work with | Set bm25_k and vector_k to 20+             |
| **Mixing raw scores from different sources** | BM25 scores are 0-30+, cosine is 0-1 | Use RRF (rank-based) or normalize scores   |
| **Same k for retrieval and final output**    | No headroom for fusion to add value  | Retrieve 3-5× more candidates than final k |
| **Not logging which source contributed**     | Can't debug retrieval failures       | Track BM25 vs vector rank for each result  |

---

## Key Takeaways

1. **Hybrid retrieval is the production default** — not an optimization, a necessity.
2. **RRF is the safest fusion strategy** — rank-based, no normalization needed.
3. **Retrieve more candidates than you need** — fusion works best with a large pool.
4. **Track provenance** — log whether each result came from BM25, vector, or both.
5. **BM25 + Vector covers >95% of retrieval needs** — add re-ranking on top for the rest.
