# RAG Failure Modes — Build-Time Awareness

## Why It Matters

Most RAG failures are **silent** — the system returns an answer that looks plausible but is wrong. Understanding failure modes before they happen lets you design defenses proactively rather than debugging production incidents reactively.

```
THE 6 RAG FAILURE MODES:

  ┌──────────────────────────────────────────────┐
  │                                              │
  │  1. OVER-RETRIEVAL HALLUCINATION             │
  │     Too many chunks → LLM invents            │
  │     connections between unrelated info        │
  │                                              │
  │  2. MISSING CONTEXT HALLUCINATION            │
  │     Answer exists in corpus but retrieval     │
  │     didn't find it → LLM fills the gap       │
  │                                              │
  │  3. RETRIEVAL NOISE AMPLIFICATION            │
  │     Irrelevant chunks dilute good ones        │
  │     → LLM gets confused by noise             │
  │                                              │
  │  4. STALE DATA ERRORS                        │
  │     Index has outdated info → LLM gives       │
  │     correct-but-old answers                   │
  │                                              │
  │  5. CHUNK BOUNDARY HALLUCINATIONS            │
  │     Answer split across chunks → incomplete   │
  │     context → LLM makes up the rest           │
  │                                              │
  │  6. AUTHORITY INVERSION                      │
  │     Low-quality sources outrank high-quality   │
  │     ones → LLM trusts the wrong source        │
  │                                              │
  └──────────────────────────────────────────────┘
```

---

## 1. Over-Retrieval Hallucination

### What Happens

Too many retrieved chunks overwhelm the LLM. When presented with 10-20 chunks covering different aspects, the LLM may synthesize connections that don't exist in any single source.

```
EXAMPLE:
  Query: "What's the timeout for API calls?"

  Retrieved (k=10):
    Chunk 1: "API gateway timeout: 30 seconds"
    Chunk 2: "Database connection timeout: 5 seconds"
    Chunk 3: "Retry timeout: 60 seconds"
    Chunk 4: "Healthcheck timeout: 10 seconds"
    ...6 more chunks about different timeouts...

  LLM answer: "The API call timeout is 30 seconds, but it retries
  after 60 seconds with a total deadline of 10 seconds for health
  checks..."

  → LLM merged multiple unrelated timeouts into one confused answer.
```

### Detection & Prevention

```python
"""
Detect and prevent over-retrieval hallucination.
"""


def detect_over_retrieval(
    chunks: list[str],
    scores: list[float],
    max_good_chunks: int = 5,
    score_cliff_threshold: float = 0.15,
) -> dict:
    """
    Check if we're retrieving too many chunks, diluting the good ones.

    Returns:
        {"risk": "low"|"medium"|"high", "recommended_k": int, "reason": str}
    """
    if len(chunks) <= 3:
        return {"risk": "low", "recommended_k": len(chunks), "reason": "Few chunks"}

    # Find the score cliff — where scores drop significantly
    cliff_index = len(scores)
    for i in range(1, len(scores)):
        if scores[i - 1] - scores[i] > score_cliff_threshold:
            cliff_index = i
            break

    # If we're sending more chunks than the cliff suggests
    if len(chunks) > cliff_index + 2:
        return {
            "risk": "high",
            "recommended_k": cliff_index,
            "reason": f"Score cliff at position {cliff_index}. "
                      f"Chunks after this are likely noise.",
        }

    # Check topic diversity — too many topics = confusion
    if cliff_index > max_good_chunks:
        return {
            "risk": "medium",
            "recommended_k": max_good_chunks,
            "reason": f"Many high-scoring chunks. Consider limiting to {max_good_chunks}.",
        }

    return {
        "risk": "low",
        "recommended_k": len(chunks),
        "reason": "Retrieval looks clean",
    }


# Example
result = detect_over_retrieval(
    chunks=["chunk1", "chunk2", "chunk3", "chunk4", "chunk5",
            "chunk6", "chunk7", "chunk8"],
    scores=[0.92, 0.88, 0.85, 0.82, 0.50, 0.45, 0.40, 0.35],
    #                                    ↑ cliff here (0.82 → 0.50)
)
print(result)
# {'risk': 'high', 'recommended_k': 4, 'reason': 'Score cliff at position 4...'}
```

---

## 2. Missing Context Hallucination

### What Happens

The answer exists in the corpus but the retriever failed to find it — due to vocabulary mismatch, poor chunking, or inadequate query. The LLM, not knowing the answer is available, fabricates one.

```
EXAMPLE:
  Corpus contains: "Rate limiting is configured via X-RateLimit-Policy header"

  Query: "How do I throttle API requests?"

  Retriever: Finds nothing (no embedding match for "throttle" → "rate limit")

  LLM: "You can throttle API requests by setting the maxRequests
  parameter in the config file."  ← COMPLETELY FABRICATED

  The right answer was in the corpus, but retrieval missed it.
```

### Prevention

```
STRATEGIES TO PREVENT MISSING CONTEXT:

  1. QUERY EXPANSION
     "throttle" → also search for "rate limit", "request limit"

  2. HYBRID RETRIEVAL
     Vector search misses vocabulary mismatch,
     BM25 catches exact keyword matches.

  3. SYNONYM MAPPING
     Maintain domain-specific synonym lists:
     throttle ↔ rate limit ↔ request limiting

  4. ABSTENTION
     If retrieval scores are low, SAY "I don't know"
     instead of generating a fabricated answer.

  5. CHUNK OVERLAP
     Ensure overlapping windows so key sentences
     appear in multiple chunks.
```

---

## 3. Retrieval Noise Amplification

### What Happens

Irrelevant chunks are included alongside relevant ones. The LLM tries to use everything, and the noise corrupts the answer.

```
EXAMPLE:
  Query: "What database does the payment service use?"

  Retrieved:
    [0.85] "Payment service stores transactions in PostgreSQL 15"  ← CORRECT
    [0.60] "The analytics dashboard queries Elasticsearch"         ← NOISE
    [0.58] "MongoDB is used for session storage"                   ← NOISE
    [0.55] "Redis caches frequently accessed payment tokens"       ← PARTIAL

  LLM: "The payment service uses PostgreSQL 15 for transactions,
  Elasticsearch for analytics, and Redis for caching."

  → LLM attributed Elasticsearch to payment service (WRONG —
    that's the analytics dashboard's database).
```

### Prevention

```python
"""
Filter noise from retrieval results before sending to LLM.
"""


def filter_retrieval_noise(
    chunks: list[str],
    scores: list[float],
    min_score: float = 0.5,
    max_score_ratio: float = 0.65,
) -> list[tuple[str, float]]:
    """
    Remove noisy chunks using two filters:
    1. Absolute score threshold
    2. Relative score threshold (must be >= X% of top score)

    Returns:
        Filtered list of (chunk, score) tuples.
    """
    if not scores:
        return []

    top_score = scores[0]
    relative_threshold = top_score * max_score_ratio

    filtered = []
    for chunk, score in zip(chunks, scores):
        if score >= min_score and score >= relative_threshold:
            filtered.append((chunk, score))

    return filtered


# Example
filtered = filter_retrieval_noise(
    chunks=[
        "Payment service stores transactions in PostgreSQL 15",
        "The analytics dashboard queries Elasticsearch",
        "MongoDB is used for session storage",
        "Redis caches frequently accessed payment tokens",
    ],
    scores=[0.85, 0.60, 0.58, 0.55],
    min_score=0.5,
    max_score_ratio=0.65,
)
# relative_threshold = 0.85 * 0.65 = 0.55
# Keeps: PostgreSQL (0.85), Elasticsearch (0.60), Redis (0.55)
# Drops: MongoDB (0.58) — wait, 0.58 > 0.55, so it stays too!
#
# Need tighter ratio or domain-specific filtering for better noise removal.

for chunk, score in filtered:
    print(f"  [{score:.2f}] {chunk[:60]}")
```

---

## 4. Stale Data Errors

### What Happens

The index contains outdated information. The LLM provides a correct answer **based on old data** — which is effectively wrong.

```
EXAMPLE:
  Index was last updated: January 2024

  Query: "What's the latest Python version?"

  Chunk (indexed Jan 2024): "Python 3.12 was released in October 2023"

  LLM: "The latest Python version is 3.12."

  Actual answer (today): Python 3.13 is out.

  → Answer was correct when indexed, but STALE now.
```

### Prevention

```
STRATEGIES FOR STALE DATA:

  1. TIMESTAMP METADATA
     Store indexed_at and source_updated_at on every chunk.
     Flag chunks older than a threshold.

  2. FRESHNESS SCORING
     Penalize old chunks in ranking:
     final_score = relevance_score × freshness_weight

  3. TEMPORAL DISCLAIMERS
     Add "Information current as of {date}" to context.

  4. INCREMENTAL REINDEXING
     Schedule periodic re-crawl of source documents.
     Diff-based updates, not full rebuilds.

  5. VERSION-AWARE RETRIEVAL
     For versioned docs (APIs, changelogs), always prefer
     latest version chunks.
```

```python
"""Simple freshness-aware scoring."""

from datetime import datetime, timedelta


def freshness_adjusted_score(
    relevance_score: float,
    indexed_at: datetime,
    half_life_days: int = 90,
) -> float:
    """
    Decay relevance score based on age.
    Score halves every `half_life_days` days.
    """
    age_days = (datetime.now() - indexed_at).days
    decay = 0.5 ** (age_days / half_life_days)
    return relevance_score * decay


# Example: same relevance, different ages
now = datetime.now()
print(freshness_adjusted_score(0.90, now - timedelta(days=7)))    # ~0.89 (fresh)
print(freshness_adjusted_score(0.90, now - timedelta(days=90)))   # ~0.45 (half-life)
print(freshness_adjusted_score(0.90, now - timedelta(days=365)))  # ~0.06 (very stale)
```

---

## 5. Chunk Boundary Hallucinations

### What Happens

A complete answer spans two chunks, but the retriever only returns one. The LLM gets an incomplete context and fills in the missing part with fabricated content.

```
EXAMPLE:
  Original document:
  "The maximum file upload size is 100MB. Files larger than this
  must be uploaded using the multi-part upload API, which supports
  files up to 5GB in 10MB segments."

  Chunk 1: "The maximum file upload size is 100MB. Files larger than this"
  Chunk 2: "must be uploaded using the multi-part upload API, which supports
            files up to 5GB in 10MB segments."

  Retriever returns only Chunk 1.

  LLM: "The maximum file upload size is 100MB. Files larger than this
  will be rejected by the server."  ← FABRICATED ENDING

  The real answer (in Chunk 2) was about multi-part upload,
  not rejection.
```

### Prevention

```
STRATEGIES:

  1. CHUNK OVERLAP
     Use 10-20% overlap so split sentences appear in both chunks.

  2. SENTENCE-BOUNDARY CHUNKING
     Never split in the middle of a sentence.

  3. PARENT DOCUMENT RETRIEVAL
     Retrieve the chunk, but send the parent (larger section)
     to the LLM for context.

  4. ADJACENT CHUNK EXPANSION
     When a chunk is retrieved, also include chunk N-1 and N+1.
```

```python
"""
Adjacent chunk expansion: include surrounding chunks for context.
"""


def expand_chunks(
    retrieved_indices: list[int],
    all_chunks: list[str],
    window: int = 1,
) -> list[str]:
    """
    For each retrieved chunk, also include adjacent chunks.

    Args:
        retrieved_indices: Indices of retrieved chunks
        all_chunks: All chunks in order
        window: Number of adjacent chunks to include on each side

    Returns:
        Expanded list of chunks (deduplicated, in order)
    """
    expanded_indices = set()
    for idx in retrieved_indices:
        for offset in range(-window, window + 1):
            neighbor = idx + offset
            if 0 <= neighbor < len(all_chunks):
                expanded_indices.add(neighbor)

    # Return in document order
    return [all_chunks[i] for i in sorted(expanded_indices)]


# Example
all_chunks = [
    "Chapter 1: Introduction to the system.",
    "The maximum upload size is 100MB. Files larger than this",
    "must use multi-part upload API, supporting up to 5GB.",
    "Chapter 2: Authentication and authorization.",
    "OAuth2 is required for all API endpoints.",
]

# Retriever found chunk index 1
expanded = expand_chunks([1], all_chunks, window=1)
for chunk in expanded:
    print(f"  {chunk}")
# Includes chunks 0, 1, and 2 — now the LLM sees the full upload info.
```

---

## 6. Authority Inversion

### What Happens

Low-quality or unreliable sources rank higher than authoritative ones. The LLM trusts the wrong document.

```
EXAMPLE:
  Query: "What's the recommended dosage for Medication X?"

  Retrieved:
    [0.91] Blog post: "I take 200mg of Medication X daily" ← UNRELIABLE
    [0.87] FDA label: "Recommended dosage: 50-100mg daily" ← AUTHORITATIVE
    [0.83] Forum: "My doctor said 150mg works best"        ← ANECDOTAL

  LLM: "The recommended dosage is 200mg daily."

  → Blog post ranked higher than the FDA label!
    Authority inversion caused a dangerous error.
```

### Prevention

```python
"""
Authority-based score boosting.
Ensure high-quality sources rank above low-quality ones.
"""

from dataclasses import dataclass
from enum import Enum


class SourceTier(Enum):
    """Source authority tiers with boost factors."""
    OFFICIAL = 1.3      # Official docs, regulatory filings
    VERIFIED = 1.15     # Peer-reviewed, internal docs
    COMMUNITY = 1.0     # Stack Overflow, tutorials
    UNVERIFIED = 0.8    # Blog posts, forums
    USER_GENERATED = 0.6  # Comments, social media


@dataclass
class AuthorityChunk:
    text: str
    raw_score: float
    source_tier: SourceTier
    source_name: str

    @property
    def authority_adjusted_score(self) -> float:
        return self.raw_score * self.source_tier.value


def authority_rerank(chunks: list[AuthorityChunk]) -> list[AuthorityChunk]:
    """Re-rank chunks by authority-adjusted score."""
    return sorted(chunks, key=lambda c: c.authority_adjusted_score, reverse=True)


# Example: Authority inversion fixed
chunks = [
    AuthorityChunk(
        text="I take 200mg of Medication X daily",
        raw_score=0.91,
        source_tier=SourceTier.UNVERIFIED,
        source_name="health-blog.com",
    ),
    AuthorityChunk(
        text="Recommended dosage: 50-100mg daily",
        raw_score=0.87,
        source_tier=SourceTier.OFFICIAL,
        source_name="FDA Label",
    ),
    AuthorityChunk(
        text="My doctor said 150mg works best",
        raw_score=0.83,
        source_tier=SourceTier.USER_GENERATED,
        source_name="health-forum.com",
    ),
]

print("Before authority re-ranking:")
for c in chunks:
    print(f"  [{c.raw_score:.2f}] ({c.source_tier.name}) {c.text[:50]}")

reranked = authority_rerank(chunks)
print("\nAfter authority re-ranking:")
for c in reranked:
    print(f"  [{c.authority_adjusted_score:.2f}] ({c.source_tier.name}) {c.text[:50]}")
# FDA Label now ranks first: 0.87 × 1.3 = 1.13
# Blog drops: 0.91 × 0.8 = 0.73
```

---

## Summary: Detection & Prevention Matrix

```
┌─────────────────────────┬──────────────────────┬──────────────────────┐
│ Failure Mode            │ Detection Signal     │ Prevention           │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Over-retrieval          │ Many chunks, score   │ Dynamic k, score     │
│ hallucination           │ cliff, topic spread  │ cliff detection      │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Missing context         │ Low retrieval scores │ Hybrid retrieval,    │
│ hallucination           │ across all chunks    │ query expansion      │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Noise amplification     │ High variance in     │ Relative score       │
│                         │ chunk relevance      │ filtering            │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Stale data              │ Old timestamp on     │ Freshness decay,     │
│                         │ top chunks           │ incremental reindex  │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Chunk boundary          │ Chunks end mid-      │ Overlap, adjacent    │
│ hallucination           │ sentence             │ chunk expansion      │
├─────────────────────────┼──────────────────────┼──────────────────────┤
│ Authority inversion     │ Low-tier source      │ Authority-based      │
│                         │ ranks first          │ score boosting       │
└─────────────────────────┴──────────────────────┴──────────────────────┘
```

---

## Key Takeaways

1. **Most RAG failures are silent** — the system returns plausible-looking wrong answers.
2. **Over-retrieval is as dangerous as under-retrieval** — more chunks ≠ better answers.
3. **Missing context causes fabrication** — if retrieval misses the right chunk, the LLM fills the gap.
4. **Noise needs filtering** — don't send low-relevance chunks to the LLM.
5. **Stale data is a time bomb** — always track when documents were last updated.
6. **Chunk boundaries destroy meaning** — use overlap and adjacent chunk expansion.
7. **Authority matters** — not all sources are equal; boost official/verified sources.
