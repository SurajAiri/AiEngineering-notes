# Chunk Size, Overlap & Trade-offs

## Why This Matters

Chunk size is the single most impactful parameter in a RAG system. Too small and chunks lack context. Too large and the semantic signal gets diluted. There's no universal "right" answer — it depends on your data, queries, and embedding model.

---

## Chunk Size Spectrum

```
Tokens:  50      128     256     512     1024    2048
         │        │       │       │        │       │
         ▼        ▼       ▼       ▼        ▼       ▼
    ┌─────────────────────────────────────────────────┐
    │  SMALL                         LARGE            │
    │  ┌────────┐          ┌──────────────────────┐   │
    │  │ Precise│          │ Rich context          │   │
    │  │ but no │          │ but diluted signal    │   │
    │  │ context│          │                       │   │
    │  └────────┘          └──────────────────────┘   │
    │                                                  │
    │  ✅ Good for:        ✅ Good for:                │
    │  • Fact lookup       • Complex explanations     │
    │  • Entity details    • Multi-step reasoning     │
    │  • Definitions       • Summarization            │
    │                                                  │
    │  ❌ Bad for:         ❌ Bad for:                 │
    │  • Complex Qs        • Precise fact lookup      │
    │  • "Why" questions   • Multiple topics mixed    │
    └─────────────────────────────────────────────────┘
```

---

## The Impact on RAG Quality

### Small Chunks (50-128 tokens)

```python
# Chunk: "HNSW uses a multi-layer graph structure."
# Query: "How does HNSW work?"
#
# Result: ✅ High precision match
# But: ❌ Insufficient context for a complete answer
#         "It uses a multi-layer graph structure." ← not helpful alone
```

### Large Chunks (512-1024 tokens)

```python
# Chunk: [Full 800-token section about HNSW, including algorithm details,
#         complexity analysis, comparison with IVF, and implementation notes]
#
# Query: "What is the time complexity of HNSW?"
#
# Result: Answer is IN the chunk, but...
# ❌ Embedding of 800 tokens averages across many concepts
# ❌ The time complexity sentence gets diluted
# ❌ Chunk might score lower than a 100-token chunk focused on complexity
```

---

## Empirical Recommendations

```
┌─────────────────────────────────────────────────────────────┐
│                 STARTING POINTS BY USE CASE                  │
│                                                              │
│  Use Case                    Chunk Size    Overlap           │
│  ─────────────────────────   ──────────    ───────           │
│  Chatbot FAQ                 100-200       20-40             │
│  Technical docs search       256-512       50-100            │
│  Legal document analysis     512-1024      100-200           │
│  Code search                 200-400       50-100            │
│  Academic paper search       256-512       50-100            │
│  Customer support            128-256       20-50             │
│                                                              │
│  ⚠️ These are STARTING POINTS. Always measure and adjust.   │
└─────────────────────────────────────────────────────────────┘
```

---

## Measuring Chunk Size Impact

```python
"""
Framework for testing different chunk sizes on YOUR data.
The only way to find the right size is to measure.

Requirements: pip install sentence-transformers tiktoken numpy
"""

import re
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer


class ChunkSizeExperiment:
    """
    Test different chunk sizes against a set of queries
    to find the optimal size for your use case.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def create_chunks(
        self, text: str, chunk_size: int, overlap: int = 0
    ) -> list[str]:
        """Create fixed-size chunks with optional overlap."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent))
            if current_tokens + sent_tokens > chunk_size and current:
                chunks.append(' '.join(current))

                # Handle overlap
                if overlap > 0:
                    overlap_sents = []
                    overlap_count = 0
                    for s in reversed(current):
                        s_toks = len(self.tokenizer.encode(s))
                        if overlap_count + s_toks <= overlap:
                            overlap_sents.insert(0, s)
                            overlap_count += s_toks
                        else:
                            break
                    current = overlap_sents + [sent]
                    current_tokens = overlap_count + sent_tokens
                else:
                    current = [sent]
                    current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(' '.join(current))

        return chunks

    def evaluate_chunk_size(
        self,
        document: str,
        queries: list[str],
        relevant_passages: list[str],
        chunk_sizes: list[int],
        overlap_ratio: float = 0.1,
        k: int = 3,
    ) -> dict[int, dict]:
        """
        Test different chunk sizes and report retrieval quality.

        Args:
            document: Source document text
            queries: Test queries
            relevant_passages: Expected relevant text for each query
            chunk_sizes: List of chunk sizes to test
            overlap_ratio: Overlap as fraction of chunk size
            k: Number of top results to consider
        """
        results = {}

        for size in chunk_sizes:
            overlap = int(size * overlap_ratio)
            chunks = self.create_chunks(document, size, overlap)
            chunk_embeddings = self.model.encode(chunks, normalize_embeddings=True)

            hits = 0
            total_queries = len(queries)
            avg_rank = []

            for query, relevant in zip(queries, relevant_passages):
                query_emb = self.model.encode(query, normalize_embeddings=True)

                # Compute similarities
                sims = np.dot(chunk_embeddings, query_emb)
                top_k_indices = np.argsort(sims)[-k:][::-1]
                top_k_chunks = [chunks[i] for i in top_k_indices]

                # Check if relevant passage appears in top-k
                found = False
                for rank, chunk in enumerate(top_k_chunks, 1):
                    if relevant.lower() in chunk.lower():
                        hits += 1
                        avg_rank.append(rank)
                        found = True
                        break

                if not found:
                    avg_rank.append(k + 1)  # not found

            results[size] = {
                "chunk_count": len(chunks),
                "recall_at_k": hits / total_queries,
                "avg_rank": np.mean(avg_rank),
                "avg_tokens": np.mean([
                    len(self.tokenizer.encode(c)) for c in chunks
                ]),
            }

        return results


# ─── Usage ───
if __name__ == "__main__":
    exp = ChunkSizeExperiment()

    document = """
    HNSW (Hierarchical Navigable Small World) is a graph-based algorithm
    for approximate nearest neighbor search. It constructs a multi-layer
    graph where each layer is a navigable small world graph. The algorithm
    has O(log n) search complexity.

    The construction process adds points one by one. For each new point,
    it finds its nearest neighbors in the existing graph and creates
    bidirectional connections. The number of connections per node is
    controlled by the parameter M.

    IVF (Inverted File Index) takes a different approach. It partitions
    the vector space into clusters using k-means clustering. At query
    time, only the nearest clusters are searched, controlled by the
    nprobe parameter.

    DiskANN is designed for datasets too large to fit in memory. It
    stores the graph on disk and uses a pruning strategy called
    Vamana to create a compact graph structure. It achieves high
    recall while using minimal memory.
    """

    queries = [
        "What is the search complexity of HNSW?",
        "How does IVF partition vectors?",
        "What is special about DiskANN?",
    ]

    relevant = [
        "O(log n) search complexity",
        "partitions the vector space into clusters using k-means",
        "datasets too large to fit in memory",
    ]

    results = exp.evaluate_chunk_size(
        document, queries, relevant,
        chunk_sizes=[64, 128, 256, 512],
        k=3,
    )

    print(f"{'Size':>6} {'Chunks':>7} {'Recall@3':>10} {'Avg Rank':>10} {'Avg Tokens':>11}")
    print("-" * 50)
    for size, metrics in sorted(results.items()):
        print(
            f"{size:>6} {metrics['chunk_count']:>7} "
            f"{metrics['recall_at_k']:>10.2f} {metrics['avg_rank']:>10.2f} "
            f"{metrics['avg_tokens']:>11.0f}"
        )
```

---

## Overlap — How Much and Why

```
No overlap (0%):
┌──────┐ ┌──────┐ ┌──────┐
│  A   │ │  B   │ │  C   │    Sentence at boundary → lost
└──────┘ └──────┘ └──────┘

10% overlap:
┌────────┐
│   A    │
│      ┌─┤────────┐
│      │ │   B    │            Boundary sentence → preserved
└──────┤ │      ┌─┤────────┐
       └─┤      │ │   C    │
         │      │ │        │
         └──────┤ │        │
                └─┴────────┘

Cost of overlap:
  chunk_size=512, overlap=0   → N chunks
  chunk_size=512, overlap=50  → ~1.1N chunks   (+10%)
  chunk_size=512, overlap=100 → ~1.24N chunks  (+24%)
  chunk_size=512, overlap=200 → ~1.64N chunks  (+64%)
```

### Overlap Best Practices

```python
"""
Calculate the storage/compute impact of different overlap settings.
"""

def chunk_math(
    doc_tokens: int,
    chunk_size: int,
    overlap: int,
) -> dict:
    """Calculate chunks needed and overhead."""
    step = chunk_size - overlap
    num_chunks_overlap = max(1, -(-((doc_tokens - overlap)) // step))  # ceiling division
    num_chunks_no_overlap = max(1, -(-doc_tokens // chunk_size))

    overhead = (num_chunks_overlap - num_chunks_no_overlap) / num_chunks_no_overlap

    return {
        "chunks_no_overlap": num_chunks_no_overlap,
        "chunks_with_overlap": num_chunks_overlap,
        "overhead_pct": overhead * 100,
        "total_tokens_stored": num_chunks_overlap * chunk_size,
    }


# Compare overlap settings for a 10,000 token document
doc_tokens = 10_000
print(f"Document: {doc_tokens} tokens, chunk_size=512\n")

for overlap in [0, 25, 50, 100, 200]:
    result = chunk_math(doc_tokens, 512, overlap)
    print(f"Overlap={overlap:3d}: {result['chunks_with_overlap']:3d} chunks "
          f"(+{result['overhead_pct']:.1f}% overhead)")
```

---

## Chunk Size vs Retrieval Quality vs Cost

```
                     ┌─────────────────────────────┐
                     │     THE FUNDAMENTAL           │
                     │     TRADEOFF                   │
                     └─────────────────────────────┘

    Retrieval                              Retrieval
    Precision ▲                            Context  ▲
              │  ╲                                  │       ╱
              │    ╲                                │     ╱
              │      ╲                              │   ╱
              │        ╲___                         │ ╱___
              │            ╲                        │╱
              └──────────────▶                      └──────────────▶
              Small      Large                     Small      Large
              Chunk Size                           Chunk Size


    Embedding          Storage             Query
    Cost     ▲         Cost     ▲          Latency  ▲
             │ ╲                │ ╲                  │ ╲
             │   ╲              │   ╲                │   ╲
             │     ╲            │     ╲              │     ╲
             │       ╲          │       ╲__          │       ╲__
             └────────▶         └──────────▶         └──────────▶
             Small   Large     Small   Large        Small   Large
             Chunk Size        Chunk Size           Chunk Size

Small chunks: MORE chunks to embed & store, but LESS data per embedding call
Large chunks: FEWER chunks, but each embedding captures more (diluted) content
```

---

## Chunk-Query Alignment

The most overlooked aspect of chunk sizing:

```
Your chunk size should match the GRANULARITY of your queries.

Query type:              Best chunk size:
─────────────────────    ──────────────────
"What is X?"            Small (100-200 tokens) — definition-level
"How does X work?"      Medium (256-512 tokens) — explanation-level
"Compare X and Y"       Large (512-1024 tokens) — multi-concept level
"Summarize section Z"   Very large or hierarchical
```

```python
"""
Demonstrate chunk-query alignment.
Same document, different chunk sizes work for different queries.
"""

document = """
HNSW stands for Hierarchical Navigable Small World. It is an algorithm
for approximate nearest neighbor search in high-dimensional spaces.

The algorithm works by constructing a multi-layer graph. Each layer is
a navigable small world graph. The top layer has the fewest nodes with
long-range connections. Each subsequent layer has more nodes with
shorter connections. The bottom layer contains all data points.

During search, the algorithm starts at the top layer and greedily
navigates to the nearest node. It then descends to the next layer and
repeats the process. This hierarchical approach provides O(log n)
search time complexity.

The key parameters are M (max connections per node), ef_construction
(search depth during build), and ef_search (search depth during query).
Higher M values improve recall but increase memory usage. Higher ef
values improve accuracy but increase latency.
"""

# Query 1: Definition → small chunk would be enough
# "What is HNSW?"
# → First paragraph is perfect

# Query 2: Process → medium chunk
# "How does HNSW search work?"
# → Second + third paragraphs needed

# Query 3: Configuration → needs specific detail
# "What does the M parameter do in HNSW?"
# → Last paragraph, specifically about M
```

---

## Pitfalls & Common Mistakes

| Mistake                                    | Impact                                   | Fix                                                |
| ------------------------------------------ | ---------------------------------------- | -------------------------------------------------- |
| **One size for all documents**             | Different doc types need different sizes | Profile your document types and adjust             |
| **Never testing chunk size**               | Guessing instead of measuring            | Run the experiment framework above                 |
| **Ignoring embedding model limits**        | Chunks exceed model's max tokens         | Check model's max_seq_length (512 for many models) |
| **Overlap on homogeneous text**            | Wasted storage for no benefit            | Use overlap mainly for diverse content             |
| **Chunk size not aligned with query type** | Systematic retrieval failures            | Analyze your queries and match granularity         |
| **Optimizing chunk size in isolation**     | Other parameters compensate              | Test chunk size together with k, reranking, etc.   |

---

## Decision Framework

```
START HERE
    │
    ▼
What are your queries mostly about?
    │
    ├── Simple facts / definitions
    │   → 100-200 tokens, low overlap
    │
    ├── Explanations / how-things-work
    │   → 256-512 tokens, 10-15% overlap
    │
    ├── Complex analysis / comparisons
    │   → 512-1024 tokens, 15-20% overlap
    │
    └── Mixed query types
        → Consider hierarchical chunking
        → Or test multiple sizes and pick best average
```

---

## Key Takeaways

1. **There is no universal optimal chunk size** — it depends on your data and queries.
2. **Measure, don't guess** — use the experiment framework to test chunk sizes on your data.
3. **Chunk size should match query granularity** — small queries need small chunks.
4. **Overlap of 10-15%** is generally a good starting point.
5. **The embedding model matters** — most models have a max_seq_length (often 512 tokens).
6. **Test chunk size with your retrieval pipeline** — not in isolation.
