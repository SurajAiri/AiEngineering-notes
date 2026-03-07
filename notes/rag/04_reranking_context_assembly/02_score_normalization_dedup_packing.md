# Score Normalization, Context Deduplication & Token Budget

## Part 1: Score Normalization

### Why It Matters

Different retrieval sources produce scores on completely different scales. BM25 scores might range from 0 to 30+, cosine similarity from -1 to 1, and cross-encoder scores from -10 to 10. To combine, compare, or threshold these scores, you must **normalize** them to a common scale.

```
RAW SCORES (uncomparable):
  BM25:          [15.2, 12.8, 8.4, 3.1]
  Vector cosine: [0.92, 0.87, 0.73, 0.45]
  Cross-encoder: [4.7,  3.2,  1.1, -2.3]

  Is BM25's 15.2 "better" than cosine's 0.92? Impossible to tell!

NORMALIZED (comparable):
  BM25:          [1.00, 0.80, 0.44, 0.00]
  Vector cosine: [1.00, 0.89, 0.60, 0.00]
  Cross-encoder: [1.00, 0.79, 0.49, 0.00]

  Now: combine with weights, apply thresholds, compare across sources.
```

### Normalization Methods

```python
"""
Common score normalization methods.
"""

import numpy as np


def min_max_normalize(scores: list[float]) -> list[float]:
    """Scale to [0, 1]. Simple, but sensitive to outliers."""
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def z_score_normalize(scores: list[float]) -> list[float]:
    """Mean=0, std=1. Good for normally distributed scores."""
    mean = np.mean(scores)
    std = np.std(scores)
    if std == 0:
        return [0.0] * len(scores)
    return [(s - mean) / std for s in scores]


def sigmoid_normalize(scores: list[float], temp: float = 1.0) -> list[float]:
    """Map to (0, 1) via sigmoid. Good for cross-encoder scores."""
    return [1 / (1 + np.exp(-s / temp)) for s in scores]


def rank_normalize(scores: list[float]) -> list[float]:
    """Convert to rank-based scores. Immune to score distribution."""
    n = len(scores)
    ranked_indices = np.argsort(np.argsort(scores)[::-1])
    return [1.0 - (rank / (n - 1)) if n > 1 else 1.0 for rank in ranked_indices]


# Compare methods
raw_scores = [15.2, 12.8, 8.4, 3.1, 1.0]
print("Raw:     ", [f"{s:.2f}" for s in raw_scores])
print("MinMax:  ", [f"{s:.2f}" for s in min_max_normalize(raw_scores)])
print("Sigmoid: ", [f"{s:.2f}" for s in sigmoid_normalize(raw_scores)])
print("Rank:    ", [f"{s:.2f}" for s in rank_normalize(raw_scores)])
```

**Recommendation**: Use **min-max** for combining same-system scores, **rank-based (RRF)** for combining across systems.

---

## Part 2: Context Deduplication

### Why It Matters

When retrieving from multiple sources (hybrid search, multi-query expansion), you often get **duplicate or near-duplicate chunks**. Sending duplicates to the LLM wastes tokens and can cause repetitive answers.

```
BEFORE DEDUP (5 chunks, 2 are duplicates):
  1. "RBAC uses ClusterRole and RoleBinding..."     ←
  2. "Configure RBAC with ClusterRole objects..."    ← Near-duplicate of #1
  3. "Network policies control pod traffic..."
  4. "RBAC uses ClusterRole and RoleBinding..."      ← Exact duplicate of #1
  5. "Service accounts provide pod identity..."

AFTER DEDUP (3 unique chunks):
  1. "RBAC uses ClusterRole and RoleBinding..."
  2. "Network policies control pod traffic..."
  3. "Service accounts provide pod identity..."

Saved ~40% of token budget!
```

### Dedup Implementation

```python
"""
Context deduplication: remove exact and near-duplicate chunks
before passing to LLM.

Requirements: pip install sentence-transformers numpy
"""

import hashlib
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


@dataclass
class DedupResult:
    text: str
    score: float
    source: str
    is_duplicate: bool = False
    duplicate_of: int | None = None


class ContextDeduplicator:
    """Remove duplicate and near-duplicate chunks from retrieval results."""

    def __init__(
        self,
        semantic_threshold: float = 0.92,  # cosine sim above this = duplicate
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.threshold = semantic_threshold
        self.model = SentenceTransformer(model_name)

    def deduplicate(self, results: list[DedupResult]) -> list[DedupResult]:
        """
        Remove duplicates in order of priority (first occurrence kept).

        Three levels:
        1. Exact match (hash)
        2. Near-duplicate (normalized text match)
        3. Semantic duplicate (embedding similarity)
        """
        if not results:
            return []

        # Level 1: Exact hash dedup
        seen_hashes = set()
        unique_after_exact = []
        for r in results:
            h = hashlib.sha256(r.text.encode()).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_after_exact.append(r)
            else:
                r.is_duplicate = True

        # Level 2: Normalized text dedup
        def normalize(text: str) -> str:
            return " ".join(text.lower().split())

        seen_normalized = {}
        unique_after_norm = []
        for i, r in enumerate(unique_after_exact):
            norm = normalize(r.text)
            if norm not in seen_normalized:
                seen_normalized[norm] = i
                unique_after_norm.append(r)
            else:
                r.is_duplicate = True
                r.duplicate_of = seen_normalized[norm]

        # Level 3: Semantic dedup
        if len(unique_after_norm) <= 1:
            return unique_after_norm

        texts = [r.text for r in unique_after_norm]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        keep = [True] * len(unique_after_norm)
        for i in range(len(unique_after_norm)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(unique_after_norm)):
                if not keep[j]:
                    continue
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim > self.threshold:
                    keep[j] = False
                    unique_after_norm[j].is_duplicate = True
                    unique_after_norm[j].duplicate_of = i

        return [r for r, k in zip(unique_after_norm, keep) if k]


# Example
deduplicator = ContextDeduplicator(semantic_threshold=0.90)

results = [
    DedupResult(text="RBAC uses ClusterRole and RoleBinding for access control", score=0.9, source="bm25"),
    DedupResult(text="Configure RBAC using ClusterRole and RoleBinding objects", score=0.85, source="vector"),
    DedupResult(text="Network policies control pod-to-pod traffic in Kubernetes", score=0.82, source="vector"),
    DedupResult(text="RBAC uses ClusterRole and RoleBinding for access control", score=0.78, source="vector"),
    DedupResult(text="Service accounts provide identity for pod processes", score=0.75, source="bm25"),
]

deduplicated = deduplicator.deduplicate(results)
print(f"Before: {len(results)} chunks")
print(f"After:  {len(deduplicated)} chunks\n")
for r in deduplicated:
    print(f"  [{r.score:.2f}] ({r.source}) {r.text}")
```

---

## Part 3: Token Budget & Context Packing

### Why It Matters

LLMs have context windows (4K-128K tokens). You must **pack the most useful information within a fixed token budget**, prioritizing relevance and avoiding wasting space on low-value chunks.

```
CONTEXT WINDOW BUDGET:
┌────────────────────────────────────────────────────┐
│ System prompt           │ ~500 tokens               │
│ Retrieved context       │ ~3000 tokens  ← BUDGET    │
│ User query              │ ~100 tokens               │
│ Reserved for answer     │ ~400 tokens               │
│ ─────────────────────── │ ────────────              │
│ Total                   │ ~4000 tokens              │
└────────────────────────────────────────────────────┘

PACKING STRATEGIES:

Breadth-first:              Depth-first:
  chunk1[200tok]              chunk1[800tok]   ← Full context
  chunk2[200tok]              chunk2[800tok]
  chunk3[200tok]
  chunk4[200tok]
  ...15 chunks

  ✅ More sources             ✅ Deeper per source
  ❌ Less context each        ❌ Fewer sources
  Best for: multi-topic Q     Best for: detailed Q
```

### Context Packing Code

```python
"""
Token budget management and context packing.

Requirements: pip install tiktoken
"""

import tiktoken
from dataclasses import dataclass


@dataclass
class PackedChunk:
    text: str
    tokens: int
    score: float
    index: int
    source_doc: str = ""


class ContextPacker:
    """
    Pack chunks into a fixed token budget.

    Strategies:
    - greedy: Take highest-scoring chunks until budget exhausted
    - diverse: Ensure coverage across different source documents
    - proportional: Allocate budget proportionally to score
    """

    def __init__(
        self,
        max_tokens: int = 3000,
        encoding: str = "cl100k_base",
        separator: str = "\n\n---\n\n",
    ):
        self.max_tokens = max_tokens
        self.tokenizer = tiktoken.get_encoding(encoding)
        self.separator = separator
        self.separator_tokens = len(self.tokenizer.encode(separator))

    def pack_greedy(
        self,
        chunks: list[dict],  # {"text": str, "score": float, "source": str}
    ) -> list[PackedChunk]:
        """
        Greedily pack highest-scoring chunks until budget is exhausted.
        """
        # Sort by score descending
        sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)

        packed = []
        used_tokens = 0

        for i, chunk in enumerate(sorted_chunks):
            chunk_tokens = len(self.tokenizer.encode(chunk["text"]))
            needed = chunk_tokens + (self.separator_tokens if packed else 0)

            if used_tokens + needed > self.max_tokens:
                # Try truncating this chunk to fit remaining budget
                remaining = self.max_tokens - used_tokens - self.separator_tokens
                if remaining > 50:  # worth including if >50 tokens
                    truncated = self._truncate_to_tokens(chunk["text"], remaining)
                    packed.append(PackedChunk(
                        text=truncated,
                        tokens=remaining,
                        score=chunk["score"],
                        index=i,
                        source_doc=chunk.get("source", ""),
                    ))
                break

            packed.append(PackedChunk(
                text=chunk["text"],
                tokens=chunk_tokens,
                score=chunk["score"],
                index=i,
                source_doc=chunk.get("source", ""),
            ))
            used_tokens += needed

        return packed

    def pack_diverse(
        self,
        chunks: list[dict],
    ) -> list[PackedChunk]:
        """
        Pack chunks ensuring diversity across source documents.
        Round-robin across sources, taking the best from each.
        """
        # Group by source
        by_source: dict[str, list] = {}
        for chunk in chunks:
            source = chunk.get("source", "unknown")
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(chunk)

        # Sort each source group by score
        for source in by_source:
            by_source[source].sort(key=lambda c: c["score"], reverse=True)

        # Round-robin selection
        packed = []
        used_tokens = 0
        source_ptrs = {source: 0 for source in by_source}

        while True:
            added_any = False
            for source in by_source:
                ptr = source_ptrs[source]
                if ptr >= len(by_source[source]):
                    continue

                chunk = by_source[source][ptr]
                chunk_tokens = len(self.tokenizer.encode(chunk["text"]))
                needed = chunk_tokens + (self.separator_tokens if packed else 0)

                if used_tokens + needed > self.max_tokens:
                    continue

                packed.append(PackedChunk(
                    text=chunk["text"],
                    tokens=chunk_tokens,
                    score=chunk["score"],
                    index=len(packed),
                    source_doc=source,
                ))
                used_tokens += needed
                source_ptrs[source] += 1
                added_any = True

            if not added_any:
                break

        return packed

    def format_context(self, packed: list[PackedChunk]) -> str:
        """Format packed chunks into a context string for LLM."""
        parts = []
        for i, chunk in enumerate(packed, 1):
            citation = f"[Source {i}: {chunk.source_doc}]" if chunk.source_doc else f"[Chunk {i}]"
            parts.append(f"{citation}\n{chunk.text}")
        return self.separator.join(parts)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens) + "..."

    def stats(self, packed: list[PackedChunk]) -> dict:
        total_tokens = sum(c.tokens for c in packed)
        return {
            "chunks_packed": len(packed),
            "tokens_used": total_tokens,
            "budget_utilization": f"{total_tokens / self.max_tokens:.1%}",
            "sources": list(set(c.source_doc for c in packed)),
        }


# ─── Usage ───
if __name__ == "__main__":
    packer = ContextPacker(max_tokens=500)  # small budget for demo

    chunks = [
        {"text": "RBAC uses ClusterRole and RoleBinding objects to manage permissions. "
                 "ClusterRole defines a set of permissions, while RoleBinding grants those "
                 "permissions to users or service accounts within a namespace.",
         "score": 0.95, "source": "k8s-security-guide"},
        {"text": "Network policies are Kubernetes resources that control traffic between pods. "
                 "They use label selectors to define ingress and egress rules.",
         "score": 0.88, "source": "k8s-networking-guide"},
        {"text": "Service accounts provide identity for processes running in pods. "
                 "Each namespace has a default service account.",
         "score": 0.82, "source": "k8s-security-guide"},
        {"text": "Pod security standards define three levels: Privileged, Baseline, and Restricted. "
                 "These replace the deprecated PodSecurityPolicy.",
         "score": 0.78, "source": "k8s-security-guide"},
    ]

    # Greedy packing
    print("=== GREEDY ===")
    packed = packer.pack_greedy(chunks)
    print(packer.format_context(packed))
    print(f"\nStats: {packer.stats(packed)}\n")

    # Diverse packing
    print("=== DIVERSE ===")
    packed = packer.pack_diverse(chunks)
    print(packer.format_context(packed))
    print(f"\nStats: {packer.stats(packed)}")
```

---

## Ordering Context for Answerability

```
WHERE TO PUT THE MOST RELEVANT CHUNKS:

  ┌──────────────────────────────────────┐
  │  Research says: "Lost in the Middle"  │
  │                                        │
  │  LLMs attend best to:                 │
  │    1. BEGINNING of context  ← Best    │
  │    2. END of context        ← Good    │
  │    3. MIDDLE of context     ← Worst   │
  │                                        │
  │  Recommended order:                    │
  │    [Most relevant] [Least] [2nd most]  │
  │     ↑ beginning     middle    end ↑    │
  └──────────────────────────────────────┘

  In practice: Sort by relevance descending is usually fine.
  The "lost in the middle" effect matters most with >10 chunks.
```

---

## Pitfalls & Common Mistakes

| Mistake                         | Impact                                                   | Fix                                       |
| ------------------------------- | -------------------------------------------------------- | ----------------------------------------- |
| **Not deduplicating**           | Wasted tokens, repetitive answers                        | Dedup before packing                      |
| **Ignoring token budget**       | Context exceeds window, gets truncated                   | Always count tokens before sending        |
| **Fixed context size**          | Simple queries get noise, complex queries get too little | Adaptive budget based on query complexity |
| **Not adding source citations** | Can't trace answers to sources                           | Add `[Source N]` markers                  |
| **Packing without re-ranking**  | Low-quality chunks waste budget                          | Re-rank first, then pack top results      |

---

## Key Takeaways

1. **Normalize scores** before combining results from different sources — use RRF or min-max.
2. **Deduplicate context** — exact, normalized, and semantic levels.
3. **Token budget is finite** — pack the highest-value chunks first.
4. **Diverse packing** beats greedy when questions span multiple topics.
5. **Put most relevant chunks first** — LLMs attend to beginnings best.
