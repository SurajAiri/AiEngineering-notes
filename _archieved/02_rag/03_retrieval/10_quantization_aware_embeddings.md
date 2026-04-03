# Quantization-Aware Embeddings

## Why It Matters

In production RAG with millions of vectors, **memory is the bottleneck**. A single `float32` embedding of dimension 768 takes 3,072 bytes. At 100M vectors, that's ~300GB just for vectors. **Quantization** compresses vectors to reduce memory by 4-32×, enabling larger indexes at lower cost — at the expense of some retrieval accuracy.

---

## Core Concept

```
FULL PRECISION (float32):
  [0.1234, -0.5678, 0.9012, ...]  × 768 dims × 4 bytes = 3,072 bytes/vector

BINARY QUANTIZATION:
  [1, 0, 1, ...]  × 768 dims × 1 bit = 96 bytes/vector  (32× smaller!)

PRODUCT QUANTIZATION (PQ):
  Split vector into subvectors, encode each as a code
  768 dims → 96 subvectors × 1 byte each = 96 bytes/vector

SCALAR QUANTIZATION (int8):
  [12, -56, 90, ...]  × 768 dims × 1 byte = 768 bytes/vector  (4× smaller)

┌────────────────────────────────────────────────────────────────┐
│  METHOD          │ SIZE     │ COMPRESSION │ RECALL LOSS        │
├──────────────────┼──────────┼─────────────┼────────────────────┤
│ float32          │ 3072 B   │ 1×          │ 0% (baseline)      │
│ float16          │ 1536 B   │ 2×          │ ~0%                │
│ int8 (scalar)    │ 768 B    │ 4×          │ 1-3%               │
│ Product (PQ-96)  │ 96 B     │ 32×         │ 3-10%              │
│ Binary           │ 96 B     │ 32×         │ 5-15% (varies)     │
└──────────────────┴──────────┴─────────────┴────────────────────┘
```

---

## How Product Quantization Works

```
Original vector (768 dims):
[v₁, v₂, v₃, ..., v₇₆₈]

Step 1: Split into M subvectors (e.g., M=96 → 8 dims each)
[v₁-v₈], [v₉-v₁₆], ..., [v₇₆₁-v₇₆₈]
  sub_1     sub_2            sub_96

Step 2: For each group, learn a codebook of 256 centroids (k-means)
  sub_1 → closest centroid ID: 42
  sub_2 → closest centroid ID: 187
  ...
  sub_96 → closest centroid ID: 5

Step 3: Store only the centroid IDs
  [42, 187, ..., 5] → 96 bytes instead of 3072!

At query time:
  Precompute distances between query subvectors and all centroids.
  Look up distances using centroid IDs → Approximate distance.
```

---

## Simple Code — Binary Quantization

```python
"""
Binary quantization: convert float embeddings to binary vectors.
32× compression, search using Hamming distance.
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def binary_quantize(embeddings: np.ndarray) -> np.ndarray:
    """Convert float vectors to binary: positive → 1, negative → 0."""
    return (embeddings > 0).astype(np.uint8)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Count differing bits between two binary vectors."""
    return int(np.sum(a != b))


def binary_search(
    query_bin: np.ndarray,
    doc_bins: np.ndarray,
    k: int = 5,
) -> list[tuple[int, int]]:
    """Search by Hamming distance (lower = more similar)."""
    distances = np.array([hamming_distance(query_bin, doc) for doc in doc_bins])
    top_k = np.argsort(distances)[:k]
    return [(int(idx), int(distances[idx])) for idx in top_k]


# Example
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "HNSW is a graph-based approximate nearest neighbor algorithm",
    "BM25 is a probabilistic ranking function for text retrieval",
    "Product quantization compresses vectors for efficient search",
    "Python is a popular programming language",
    "Vector databases store embeddings for similarity search",
]

# Full-precision embeddings
embeddings = model.encode(documents, normalize_embeddings=True)
query_emb = model.encode("approximate nearest neighbor search", normalize_embeddings=True)

# Quantize
doc_bins = binary_quantize(embeddings)
query_bin = binary_quantize(query_emb.reshape(1, -1))[0]

# Memory comparison
print(f"Float32 size: {embeddings.nbytes} bytes")
print(f"Binary size:  {doc_bins.nbytes} bytes")
print(f"Compression:  {embeddings.nbytes / doc_bins.nbytes:.0f}×\n")

# Search
results = binary_search(query_bin, doc_bins, k=3)
for idx, dist in results:
    print(f"  [hamming={dist}] {documents[idx]}")
```

---

## Production Code — Quantized Search with Rescoring

```python
"""
Production quantization pipeline:
1. Binary quantization for initial candidate selection (fast, cheap)
2. Full-precision rescoring of top candidates (accurate)

This is the standard pattern used by Vespa, Pinecone, and others.

Requirements: pip install faiss-cpu sentence-transformers numpy
"""

import logging
import time
import numpy as np
import faiss
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class QuantizedSearchResult:
    index: int
    text: str
    binary_rank: int     # rank from binary search
    final_score: float   # score after float32 rescoring


class TwoStageQuantizedSearch:
    """
    Two-stage retrieval:
    Stage 1: Fast binary/PQ search for candidate selection
    Stage 2: Float32 rescoring for precision

    This gives you the speed of quantized search with the
    accuracy of full-precision scoring.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        pq_segments: int = 96,    # number of PQ sub-quantizers
        pq_bits: int = 8,        # bits per code (256 centroids per segment)
        candidate_multiplier: int = 10,  # retrieve N×k candidates in stage 1
    ):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.pq_segments = pq_segments
        self.pq_bits = pq_bits
        self.candidate_multiplier = candidate_multiplier

        self.texts: list[str] = []
        self.full_embeddings: np.ndarray | None = None
        self.pq_index: faiss.IndexPQ | None = None

    def build_index(self, texts: list[str], batch_size: int = 64):
        """Build both PQ index (fast) and store full embeddings (accurate)."""
        self.texts = texts

        logger.info(f"Encoding {len(texts)} texts...")
        self.full_embeddings = self.model.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)

        # Build PQ index
        logger.info(f"Building PQ index (segments={self.pq_segments})...")
        self.pq_index = faiss.IndexPQ(
            self.dimension,
            self.pq_segments,
            self.pq_bits,
        )
        self.pq_index.train(self.full_embeddings)
        self.pq_index.add(self.full_embeddings)

        # Memory analysis
        full_size = self.full_embeddings.nbytes
        pq_size = self.pq_segments * len(texts)  # approximate
        logger.info(
            f"Memory: full={full_size / 1e6:.1f}MB, "
            f"PQ={pq_size / 1e6:.1f}MB, "
            f"compression={full_size / pq_size:.0f}×"
        )

    def search(self, query: str, k: int = 5) -> list[QuantizedSearchResult]:
        """
        Two-stage search:
        1. PQ search for candidates (fast, approximate)
        2. Full-precision rescoring (accurate)
        """
        if self.pq_index is None or self.full_embeddings is None:
            raise RuntimeError("Call build_index() first")

        query_vec = self.model.encode(
            query, normalize_embeddings=True
        ).astype(np.float32).reshape(1, -1)

        # Stage 1: PQ candidate selection
        n_candidates = min(k * self.candidate_multiplier, len(self.texts))
        start = time.perf_counter()
        _, candidate_indices = self.pq_index.search(query_vec, n_candidates)
        pq_time = (time.perf_counter() - start) * 1000

        candidates = candidate_indices[0]
        candidates = candidates[candidates != -1]  # remove padding

        # Stage 2: Full-precision rescoring
        start = time.perf_counter()
        candidate_embeddings = self.full_embeddings[candidates]
        scores = np.dot(candidate_embeddings, query_vec.T).squeeze()
        rescore_time = (time.perf_counter() - start) * 1000

        # Sort by rescored similarity
        sorted_order = np.argsort(scores)[::-1][:k]

        results = []
        for rank, sort_idx in enumerate(sorted_order):
            orig_idx = int(candidates[sort_idx])
            results.append(QuantizedSearchResult(
                index=orig_idx,
                text=self.texts[orig_idx],
                binary_rank=int(sort_idx) + 1,
                final_score=float(scores[sort_idx]),
            ))

        logger.debug(
            f"PQ search: {pq_time:.1f}ms, "
            f"Rescore: {rescore_time:.1f}ms, "
            f"Total: {pq_time + rescore_time:.1f}ms"
        )
        return results


# ─── Usage and Benchmark ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = TwoStageQuantizedSearch(pq_segments=48)

    # Generate test data
    np.random.seed(42)
    n = 50_000
    dim = 384  # all-MiniLM-L6-v2 dimension

    # Use random texts for benchmark (replace with real data in production)
    texts = [f"Document {i} about topic {i % 100}" for i in range(n)]

    engine.texts = texts
    engine.full_embeddings = np.random.randn(n, dim).astype(np.float32)
    engine.full_embeddings /= np.linalg.norm(
        engine.full_embeddings, axis=1, keepdims=True
    )

    # Build PQ index
    engine.pq_index = faiss.IndexPQ(dim, 48, 8)
    engine.pq_index.train(engine.full_embeddings)
    engine.pq_index.add(engine.full_embeddings)

    # Compare sizes
    full_mb = engine.full_embeddings.nbytes / 1e6
    pq_mb = (48 * n) / 1e6
    print(f"\nFull embeddings: {full_mb:.1f} MB")
    print(f"PQ index:        {pq_mb:.1f} MB")
    print(f"Compression:     {full_mb / pq_mb:.0f}×")
```

---

## Quantization Decision Chart

```
How important is recall?
│
├─ Critical (>99%) ─── Use float32 or float16 (no quantization)
│
├─ High (>97%) ─────── int8 scalar quantization (4× compression)
│
├─ Moderate (>93%) ──── Product quantization + rescoring (16-32× compression)
│
└─ Acceptable (>85%) ── Binary quantization (32× compression)
                         └── Good for initial filtering + rescore
```

---

## Pitfalls & Common Mistakes

| Mistake                                    | Impact                            | Fix                                               |
| ------------------------------------------ | --------------------------------- | ------------------------------------------------- |
| **Quantizing without measuring recall**    | Unknown accuracy loss             | Always benchmark recall@k vs exact search         |
| **Not rescoring after PQ/binary**          | Poor final ranking                | Two-stage: quantized candidates → float32 rescore |
| **Too few PQ segments**                    | Excessive accuracy loss           | Use at least dim/8 segments                       |
| **Quantizing before model selection**      | Quantizing a bad model is useless | Get a good embedding model first, then quantize   |
| **Binary quantization for small datasets** | Overkill, no benefit              | Only quantize when memory is actually a problem   |

---

## Key Takeaways

1. **Quantization trades accuracy for memory** — use it when scale demands it.
2. **Two-stage search** (quantized candidates → float32 rescore) is the production pattern.
3. **Binary quantization is simplest** — 32× compression, good for initial filtering.
4. **Product quantization is most flexible** — tunable compression vs accuracy.
5. **Always measure recall@k** after quantization — never assume it's "good enough."
