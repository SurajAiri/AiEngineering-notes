# Index Algorithms: HNSW vs IVF vs DiskANN

## Why It Matters

When you have millions of vectors, **brute-force search is too slow** (~100ms+ per query at 1M vectors). Approximate Nearest Neighbor (ANN) algorithms trade a tiny amount of accuracy for massive speed improvements. The three dominant approaches are **HNSW** (graph-based), **IVF** (cluster-based), and **DiskANN** (disk-optimized). Choosing the right one determines your latency, memory, and accuracy profile.

---

## Core Concept: Exact vs Approximate

```
EXACT SEARCH (brute-force):
  Compare query against ALL vectors: O(n)
  ✅ 100% accuracy
  ❌ 1M vectors × 768 dims × 4 bytes = ~3GB, ~100ms per query
  ❌ 100M vectors = ~300GB, ~10 seconds per query

APPROXIMATE SEARCH:
  Build a smart data structure, search a fraction: O(log n) or O(√n)
  ✅ 95-99.5% accuracy (recall@10)
  ✅ 1M vectors in ~1ms
  ✅ 100M vectors in ~5ms (with DiskANN)
```

---

## HNSW (Hierarchical Navigable Small World)

```
CONCEPT: Build a multi-layer graph. Each node connects to nearby nodes.
Search starts at the top (sparse) layer and descends to bottom (dense) layer.

Layer 3 (sparse):    A ─── F ─── K
                     │           │
Layer 2 (medium):    A ─── C ─── F ─── H ─── K
                     │     │     │     │     │
Layer 1 (dense):     A─B─C─D─E─F─G─H─I─J─K─L
                                 ▲
                              Query lands here
                              after traversing layers

PARAMETERS:
  M (connections per node): More = better recall, more memory
    M=16: Good for <1M vectors
    M=32: Good for 1-10M vectors (default)
    M=64: When you need max recall

  efConstruction (build-time search depth): Higher = better graph, slower build
    efConstruction=200: Standard
    efConstruction=400: When recall is critical

  efSearch (query-time search depth): Higher = better recall, slower query
    efSearch=64:  Fast, ~95% recall
    efSearch=128: Balanced, ~98% recall
    efSearch=256: High recall, ~99.5%
```

---

## IVF (Inverted File Index)

```
CONCEPT: Partition vectors into clusters (Voronoi cells).
At query time, search only the nearest clusters.

  Training phase:
  ┌─────────────────────────────────────┐
  │  Cluster vectors using k-means      │
  │                                      │
  │    ┌───┐   ┌───┐   ┌───┐   ┌───┐   │
  │    │ C1│   │ C2│   │ C3│   │ C4│   │
  │    │···│   │···│   │···│   │···│   │
  │    │···│   │···│   │···│   │···│   │
  │    └───┘   └───┘   └───┘   └───┘   │
  │  Each cluster has a centroid         │
  └─────────────────────────────────────┘

  Query phase:
  1. Find nearest nprobe centroids
  2. Search only those clusters
  3. Return top-k from searched clusters

PARAMETERS:
  nlist (number of clusters):
    Rule of thumb: nlist = √n  (e.g., 1000 for 1M vectors)

  nprobe (clusters to search at query time):
    nprobe=1:   Fastest, lowest recall
    nprobe=10:  Good balance
    nprobe=√nlist: High recall
    nprobe=nlist: Exact search (defeats the purpose)
```

---

## DiskANN

```
CONCEPT: Optimized for SSD storage. Keeps graph structure
partially on disk, so it can handle billions of vectors
without fitting everything in RAM.

  ┌─────────────┐     ┌─────────────────────┐
  │   IN RAM     │     │    ON SSD             │
  │  Compressed  │     │    Full-precision     │
  │  vectors +   │ ──► │    vectors for        │
  │  graph nav   │     │    re-ranking         │
  └─────────────┘     └─────────────────────┘

  Key innovation:
  - Graph search using compressed (PQ) vectors in RAM
  - Final re-ranking using full vectors from SSD
  - Can index 1B+ vectors with ~64GB RAM

WHEN TO USE:
  - Datasets too large for RAM (>100M vectors)
  - Cost-sensitive (SSD is cheaper than RAM)
  - Can tolerate ~5-10ms latency (vs ~1ms for in-memory HNSW)
```

---

## Comparison Code — FAISS Index Types

```python
"""
Compare HNSW, IVF, and Flat indexes in FAISS.
Measures build time, query latency, and recall.

Requirements: pip install faiss-cpu numpy
"""

import time
import numpy as np
import faiss


def benchmark_index(
    index: faiss.Index,
    data: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,  # true neighbors from exact search
    k: int = 10,
    index_name: str = "",
) -> dict:
    """Benchmark an index on build time, query time, and recall."""

    # Build
    start = time.perf_counter()
    if hasattr(index, 'train') and not index.is_trained:
        index.train(data)
    index.add(data)
    build_time = time.perf_counter() - start

    # Query
    n_queries = len(queries)
    start = time.perf_counter()
    _, predicted = index.search(queries, k)
    query_time = (time.perf_counter() - start) / n_queries * 1000  # ms per query

    # Recall@k
    recalls = []
    for pred_row, gt_row in zip(predicted, ground_truth):
        hit = len(set(pred_row) & set(gt_row))
        recalls.append(hit / k)
    recall = float(np.mean(recalls))

    result = {
        "name": index_name,
        "build_time_s": round(build_time, 2),
        "query_ms": round(query_time, 3),
        "recall@10": round(recall, 4),
        "memory_mb": round(index.ntotal * data.shape[1] * 4 / 1e6, 1),
    }
    print(f"  {index_name}: build={build_time:.1f}s, "
          f"query={query_time:.1f}ms, recall@10={recall:.3f}")
    return result


def main():
    np.random.seed(42)
    n_vectors = 100_000
    dim = 128
    n_queries = 100
    k = 10

    print(f"Dataset: {n_vectors} vectors, {dim} dimensions\n")

    # Generate data
    data = np.random.randn(n_vectors, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)

    # Ground truth from exact search
    exact_index = faiss.IndexFlatL2(dim)
    exact_index.add(data)
    _, ground_truth = exact_index.search(queries, k)

    # 1. Flat (exact) — baseline
    flat_index = faiss.IndexFlatL2(dim)
    benchmark_index(flat_index, data.copy(), queries, ground_truth, k, "Flat (exact)")

    # 2. HNSW
    hnsw_index = faiss.IndexHNSWFlat(dim, 32)  # M=32
    hnsw_index.hnsw.efConstruction = 200
    hnsw_index.hnsw.efSearch = 128
    benchmark_index(hnsw_index, data.copy(), queries, ground_truth, k, "HNSW (M=32)")

    # 3. IVF
    nlist = int(np.sqrt(n_vectors))  # ~316 clusters
    quantizer = faiss.IndexFlatL2(dim)
    ivf_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
    ivf_index.nprobe = 10
    benchmark_index(ivf_index, data.copy(), queries, ground_truth, k, f"IVF (nlist={nlist}, nprobe=10)")

    # 4. IVF + PQ (product quantization)
    ivfpq_index = faiss.IndexIVFPQ(quantizer, dim, nlist, 16, 8)
    ivfpq_index.nprobe = 10
    benchmark_index(ivfpq_index, data.copy(), queries, ground_truth, k, "IVF+PQ (compact)")


if __name__ == "__main__":
    main()
```

---

## Index Refresh for Dynamic Data

```
CHALLENGE: When documents are added/updated/deleted,
the index must reflect these changes.

┌─────────────────────────────────────────────────────────────────┐
│  STRATEGY         │ HOW                    │ WHEN               │
├───────────────────┼────────────────────────┼────────────────────┤
│ Append-only       │ Add new vectors to     │ Insert-heavy       │
│                   │ existing index         │ workloads          │
│                   │ Deletes = tombstone    │                    │
├───────────────────┼────────────────────────┼────────────────────┤
│ Periodic rebuild  │ Full rebuild on        │ Nightly/weekly     │
│                   │ schedule               │ batch updates      │
├───────────────────┼────────────────────────┼────────────────────┤
│ Blue-green        │ Build new index in     │ Zero-downtime      │
│ deployment        │ background, swap when  │ updates            │
│                   │ ready                  │                    │
├───────────────────┼────────────────────────┼────────────────────┤
│ Streaming         │ Real-time updates      │ Managed vector DBs │
│ (managed)         │ handled by the DB      │ (Pinecone, etc.)   │
└───────────────────┴────────────────────────┴────────────────────┘

For IVF: After adding >10% new vectors, re-train centroids
For HNSW: Supports online insertion, but deletions are expensive
For DiskANN: Supports streaming updates (DiskANN++)
```

---

## Decision Framework

```
                    How many vectors?
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
        <100K    100K-10M    >10M
          │         │          │
        FLAT      HNSW       ┌┴┐
       (exact)   (in-RAM)   ▼   ▼
                           Fits    Doesn't
                           RAM?    fit RAM?
                            │        │
                          HNSW    DiskANN
                          or IVF

Additional considerations:
  • Need online updates? → HNSW (easiest) or managed DB
  • Cost-sensitive? → IVF+PQ (lowest memory)
  • Highest recall? → HNSW with high efSearch
  • Lowest latency? → HNSW (sub-millisecond)
```

---

## Pitfalls & Common Mistakes

| Mistake                          | Impact                                  | Fix                                          |
| -------------------------------- | --------------------------------------- | -------------------------------------------- |
| **Using Flat for >100K vectors** | Unacceptable latency in production      | Switch to HNSW                               |
| **Low efSearch / nprobe**        | Missing relevant results                | Tune recall vs latency, benchmark            |
| **Not benchmarking recall**      | Silently losing accuracy                | Always measure recall@k against exact search |
| **Never rebuilding IVF**         | Cluster centroids drift as data changes | Retrain periodically                         |
| **Not considering delete cost**  | HNSW doesn't support true deletes       | Use tombstones + periodic rebuild            |

---

## Key Takeaways

1. **HNSW is the default choice** — best recall/latency tradeoff, works for most scales.
2. **IVF+PQ for memory-constrained** — compresses vectors significantly.
3. **DiskANN for billion-scale** — when data doesn't fit in RAM.
4. **Always benchmark recall** — approximate doesn't mean "good enough" by default.
5. **Plan for index refresh** — static indexes become stale as documents change.
