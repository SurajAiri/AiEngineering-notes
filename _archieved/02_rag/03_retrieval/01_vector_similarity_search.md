# Vector Similarity Search

## Why It Matters

Vector similarity search is the foundational retrieval mechanism in RAG. Documents and queries are converted to dense vectors (embeddings), and retrieval becomes a nearest-neighbor search in high-dimensional space.

---

## Core Concept

```
           INDEXING                              RETRIEVAL

Document → Embedding Model → [0.12, -0.34, ...]  → Store in Index
                                                          │
                                                          ▼
Query → Embedding Model → [0.15, -0.31, ...]     → Search Index
                                                          │
                                                          ▼
                                                   Top-k nearest
                                                   documents
```

### Similarity Metrics

```
┌─────────────────────────────────────────────────────────────────┐
│  METRIC         │ FORMULA              │ WHEN TO USE            │
├─────────────────┼──────────────────────┼────────────────────────┤
│ Cosine          │ A·B / (|A|×|B|)      │ Default. Direction     │
│ Similarity      │                      │ matters, not magnitude │
├─────────────────┼──────────────────────┼────────────────────────┤
│ Dot Product     │ A·B                  │ When embeddings are    │
│                 │                      │ already normalized     │
├─────────────────┼──────────────────────┼────────────────────────┤
│ Euclidean       │ √Σ(Aᵢ-Bᵢ)²          │ When magnitude matters │
│ (L2)            │                      │ (rare in NLP)          │
└─────────────────┴──────────────────────┴────────────────────────┘

Most embedding models output normalized vectors → cosine = dot product.
OpenAI, Cohere, sentence-transformers all normalize by default.
```

---

## Simple Code — Understanding Vector Search

```python
"""
Understand vector similarity search from scratch.
No libraries — just numpy.
"""

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def brute_force_search(
    query_vec: np.ndarray,
    document_vecs: np.ndarray,
    k: int = 3,
) -> list[tuple[int, float]]:
    """
    Search for k nearest neighbors by computing similarity
    against every document. O(n) — doesn't scale, but helps
    understand what's happening.
    """
    similarities = np.dot(document_vecs, query_vec) / (
        np.linalg.norm(document_vecs, axis=1) * np.linalg.norm(query_vec)
    )
    top_k_indices = np.argsort(similarities)[::-1][:k]
    return [(int(i), float(similarities[i])) for i in top_k_indices]


# Simulate with random vectors
np.random.seed(42)
docs = np.random.randn(1000, 384)  # 1000 documents, 384-dim embeddings
query = np.random.randn(384)

results = brute_force_search(query, docs, k=5)
for idx, score in results:
    print(f"Doc {idx}: similarity = {score:.4f}")
```

---

## Production Code — FAISS-Based Vector Search

```python
"""
Production vector search with FAISS.
Supports both exact (Flat) and approximate (HNSW) search.

Requirements: pip install faiss-cpu sentence-transformers numpy
"""

import logging
import time
import numpy as np
import faiss
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    index: int
    score: float
    text: str


@dataclass
class IndexStats:
    total_vectors: int
    dimension: int
    index_type: str
    memory_mb: float


class VectorSearchEngine:
    """
    Vector search engine wrapping FAISS.

    Supports:
    - Flat (exact) search for small datasets
    - HNSW (approximate) search for large datasets
    - L2 and inner product (cosine) metrics
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_type: str = "flat",       # "flat" or "hnsw"
        metric: str = "cosine",         # "cosine" or "l2"
        hnsw_m: int = 32,              # HNSW connections per node
        hnsw_ef_construction: int = 200, # HNSW build-time search depth
        hnsw_ef_search: int = 128,      # HNSW query-time search depth
    ):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index_type = index_type
        self.metric = metric
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.texts: list[str] = []
        self.index: faiss.Index | None = None

    def build_index(self, texts: list[str], batch_size: int = 64) -> IndexStats:
        """Encode texts and build FAISS index."""
        logger.info(f"Encoding {len(texts)} texts...")
        self.texts = texts

        # Batch encode
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=(self.metric == "cosine"),
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        # Build index
        if self.index_type == "flat":
            self.index = self._build_flat_index(embeddings)
        elif self.index_type == "hnsw":
            self.index = self._build_hnsw_index(embeddings)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        stats = IndexStats(
            total_vectors=self.index.ntotal,
            dimension=self.dimension,
            index_type=self.index_type,
            memory_mb=self._estimate_memory_mb(),
        )
        logger.info(f"Index built: {stats}")
        return stats

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Search the index for top-k similar documents."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_vec = self.model.encode(
            query,
            normalize_embeddings=(self.metric == "cosine"),
        )
        query_vec = np.array([query_vec], dtype=np.float32)

        start = time.perf_counter()
        distances, indices = self.index.search(query_vec, k)
        latency_ms = (time.perf_counter() - start) * 1000

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            # For inner product, distance IS the similarity score
            # For L2, convert distance to similarity
            score = float(dist) if self.metric == "cosine" else 1 / (1 + float(dist))
            results.append(SearchResult(
                index=int(idx),
                score=score,
                text=self.texts[int(idx)],
            ))

        logger.debug(f"Search completed in {latency_ms:.1f}ms, {len(results)} results")
        return results

    def _build_flat_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Exact search. O(n) per query but 100% accurate."""
        if self.metric == "cosine":
            index = faiss.IndexFlatIP(self.dimension)  # Inner Product
        else:
            index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)
        return index

    def _build_hnsw_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Approximate search. O(log n) per query, ~95-99% recall."""
        if self.metric == "cosine":
            index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)

        index.hnsw.efConstruction = self.hnsw_ef_construction
        index.hnsw.efSearch = self.hnsw_ef_search
        index.add(embeddings)
        return index

    def _estimate_memory_mb(self) -> float:
        if self.index is None:
            return 0.0
        n = self.index.ntotal
        # Flat: 4 bytes × dim × n
        # HNSW: ~(4 × dim + 4 × M × 2) × n
        if self.index_type == "flat":
            return (4 * self.dimension * n) / (1024 * 1024)
        else:
            return (4 * self.dimension + 4 * self.hnsw_m * 2) * n / (1024 * 1024)

    def save(self, path: str):
        """Save index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, path)
            logger.info(f"Index saved to {path}")

    def load(self, path: str):
        """Load index from disk."""
        self.index = faiss.read_index(path)
        logger.info(f"Index loaded from {path}, {self.index.ntotal} vectors")


# ─── Usage ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = VectorSearchEngine(
        model_name="all-MiniLM-L6-v2",
        index_type="flat",   # use "hnsw" for large datasets
        metric="cosine",
    )

    documents = [
        "Python is a high-level programming language.",
        "Machine learning models learn patterns from data.",
        "PostgreSQL is a powerful relational database.",
        "Vector databases store and search embeddings efficiently.",
        "FAISS is a library for approximate nearest neighbor search.",
        "The weather in San Francisco is foggy today.",
        "Neural networks are inspired by biological neurons.",
        "Docker containers provide isolated environments.",
        "Kubernetes orchestrates container deployments.",
        "RAG combines retrieval with language model generation.",
    ]

    engine.build_index(documents)

    query = "How do I search through embeddings?"
    results = engine.search(query, k=3)

    print(f"\nQuery: {query}\n")
    for r in results:
        print(f"  [{r.score:.4f}] {r.text}")
```

---

## How Similarity Search Really Works

```
EMBEDDING SPACE (simplified to 2D)

                    ↑ dimension 2
                    │
          "cat"  ·  │         · "database"
                    │
     "kitten"  ·    │    · "SQL"
                    │
        ────────────┼────────────→ dimension 1
                    │
                    │   · "vector search"    ← Query
                    │
                    │        · "FAISS"       ← Nearest neighbor!
                    │
                    │  · "embeddings"        ← 2nd nearest

In reality: 384-1536 dimensions, not 2.
The math is the same, just harder to visualize.
```

---

## Pitfalls & Common Mistakes

| Mistake                           | Impact                                                | Fix                                                   |
| --------------------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| **Not normalizing embeddings**    | Cosine similarity gives wrong results                 | Use `normalize_embeddings=True` or normalize manually |
| **Using L2 when you mean cosine** | Different ranking, especially for varied-length texts | Use inner product for normalized vectors              |
| **Setting k too high**            | Returns irrelevant results, wastes tokens             | Start with k=5, tune based on evaluation              |
| **Setting k too low**             | Misses relevant results                               | Retrieve more, then re-rank                           |
| **Mixing embedding models**       | Query and doc embeddings incompatible                 | Always use the same model for indexing and querying   |
| **Not batching encoding**         | Slow indexing for large datasets                      | Use `batch_size` param, GPU if available              |

---

## Trade-offs

```
┌─────────────────────────────────────────────────────────┐
│                FLAT vs HNSW vs IVF                       │
├──────────┬─────────┬──────────┬─────────────────────────┤
│          │ Flat    │ HNSW     │ IVF                     │
├──────────┼─────────┼──────────┼─────────────────────────┤
│ Speed    │ O(n)    │ O(log n) │ O(n/k) k=num clusters  │
│ Accuracy │ 100%    │ ~95-99%  │ ~90-98%                 │
│ Memory   │ n×d×4B  │ +graph   │ +centroids              │
│ Build    │ O(n)    │ O(n×M)   │ O(n×k×iter)             │
│ Best for │ <100K   │ <10M     │ >10M                    │
└──────────┴─────────┴──────────┴─────────────────────────┘
```

---

## Key Takeaways

1. **Vector search = nearest neighbor search in embedding space** — the core primitive of RAG retrieval.
2. **Always normalize embeddings** when using cosine similarity — most models do this by default.
3. **Flat index for small datasets** (<100K vectors), HNSW for larger ones.
4. **Same embedding model for indexing and querying** — this is non-negotiable.
5. **k is a tuning parameter** — start at 5, increase if recall is low, decrease if precision is low.

---

## Popular Libraries

| Library               | Best For                        | Install                             |
| --------------------- | ------------------------------- | ----------------------------------- |
| FAISS                 | High-performance local search   | `pip install faiss-cpu`             |
| sentence-transformers | Embedding generation            | `pip install sentence-transformers` |
| ChromaDB              | Simple vector store + retrieval | `pip install chromadb`              |
| Pinecone              | Managed cloud vector DB         | `pip install pinecone-client`       |
| Qdrant                | Self-hosted / cloud vector DB   | `pip install qdrant-client`         |

### Quick Example — FAISS + sentence-transformers

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Index documents
docs = ["Python is a programming language", "RAG uses retrieval for generation", "FAISS enables fast similarity search"]
doc_embeddings = model.encode(docs)

# Create FAISS index
dim = doc_embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatIP(dim)  # Inner product (use IndexFlatL2 for L2 distance)
faiss.normalize_L2(doc_embeddings)  # Normalize for cosine similarity
index.add(doc_embeddings)

# Query
query = "What is retrieval augmented generation?"
query_emb = model.encode([query])
faiss.normalize_L2(query_emb)

scores, indices = index.search(query_emb, k=2)  # Top 2 results
for score, idx in zip(scores[0], indices[0]):
    print(f"Score: {score:.3f} | {docs[idx]}")
```

### Quick Example — ChromaDB (Simplest Getting Started)

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("my_docs")

collection.add(
    documents=["Python is great", "RAG uses retrieval", "FAISS is fast"],
    ids=["doc1", "doc2", "doc3"],
)

results = collection.query(query_texts=["What is RAG?"], n_results=2)
print(results["documents"])
```

---

## Common Questions

### Q: FAISS vs ChromaDB vs Pinecone — which should I use?

**A:** **ChromaDB** to prototype fast (5 min setup, in-memory). **FAISS** for production local search (fast, battle-tested, handles millions of vectors). **Pinecone/Qdrant** for managed production (auto-scaling, no infra management). Start with ChromaDB, graduate to FAISS or a managed DB.

### Q: How many vectors can I store in memory?

**A:** A 384-dim float32 vector = 1.5 KB. So 1M vectors ≈ 1.5 GB RAM. For 1536-dim (OpenAI) = 6 GB for 1M vectors. If you exceed RAM, use disk-backed indexes (FAISS IVF) or a vector database.

### Q: Do I need to re-embed everything if I switch embedding models?

**A:** Yes. Vectors from different models live in different spaces and are **not compatible**. If you switch from MiniLM to OpenAI embeddings, you must re-embed all documents. This is why choosing your embedding model early matters.
