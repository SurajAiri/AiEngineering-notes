# Semantic Caching for RAG

## 🟢 How to Approach This Topic

> **Why this matters for your job:** RAG systems often see repeated or similar queries. Without caching, you pay full embedding + retrieval + LLM cost every time. Semantic caching recognizes that "How do I reset my password?" and "How can I change my password?" should return the same answer — saving cost and latency.

**Prerequisites:** Understand the basic RAG pipeline (retrieval → generation).

**Reading order:**

1. Why caching matters (concept) — 5 min
2. Exact vs semantic caching — 5 min
3. Simple implementation — 15 min
4. Library-based solutions — 15 min
5. When NOT to cache — 10 min

**⏱️ Core concept: 30 min | Full exploration: 1.5 hours**

---

## Why Cache RAG Results?

```
WITHOUT CACHING:
User A: "How do I reset my password?" → Embed → Search → Rerank → LLM → $0.05, 2s
User B: "How can I change my password?" → Embed → Search → Rerank → LLM → $0.05, 2s
User C: "Password reset instructions"   → Embed → Search → Rerank → LLM → $0.05, 2s
                                                            Total: $0.15, 6s

WITH SEMANTIC CACHING:
User A: "How do I reset my password?" → Embed → Search → Rerank → LLM → $0.05, 2s
User B: "How can I change my password?" → Cache HIT (similar query) → $0.001, 50ms
User C: "Password reset instructions"   → Cache HIT (similar query) → $0.001, 50ms
                                                            Total: $0.052, 2.1s
```

### Cost Impact at Scale

| Daily Queries | Cache Hit Rate | Monthly Cost (No Cache) | Monthly Cost (With Cache) | Savings |
| ------------- | -------------- | ----------------------- | ------------------------- | ------- |
| 1,000         | 30%            | ~$1,500                 | ~$1,060                   | 29%     |
| 10,000        | 40%            | ~$15,000                | ~$9,150                   | 39%     |
| 100,000       | 50%            | ~$150,000               | ~$76,500                  | 49%     |

---

## Exact vs Semantic Caching

```
EXACT CACHE (dictionary lookup):
  "How do I reset my password?" → HIT  ✅
  "How can I change my password?" → MISS ❌  (different string)

SEMANTIC CACHE (embedding similarity):
  "How do I reset my password?" → HIT  ✅
  "How can I change my password?" → HIT  ✅  (similar meaning!)
  "Tell me about quantum physics" → MISS ❌  (different topic)
```

| Feature             | Exact Cache                   | Semantic Cache                  |
| ------------------- | ----------------------------- | ------------------------------- |
| **Accuracy**        | 100% (identical queries only) | ~95% (similar meaning)          |
| **Hit rate**        | Low (10-20% typical)          | Higher (30-50% typical)         |
| **Lookup speed**    | O(1) hash lookup              | O(n) or ANN similarity search   |
| **False positives** | None                          | Possible (threshold-dependent)  |
| **Storage**         | Low (string → response)       | Higher (embeddings + responses) |

---

## Simple Code — In-Memory Semantic Cache

```python
"""
Minimal semantic cache using embeddings.
Good for understanding the concept. Not for production.
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field
import time


@dataclass
class CacheEntry:
    query: str
    query_embedding: np.ndarray
    response: str
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.85, max_entries: int = 1000):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []

    def get(self, query: str) -> str | None:
        """Look up query in cache. Returns cached response or None."""
        query_embedding = self.model.encode(query, normalize_embeddings=True)

        best_score = -1
        best_entry = None

        for entry in self.entries:
            score = np.dot(query_embedding, entry.query_embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold and best_entry:
            best_entry.hit_count += 1
            return best_entry.response

        return None  # Cache miss

    def put(self, query: str, response: str):
        """Store query-response pair in cache."""
        embedding = self.model.encode(query, normalize_embeddings=True)

        # Evict oldest if full
        if len(self.entries) >= self.max_entries:
            self.entries.sort(key=lambda e: e.created_at)
            self.entries.pop(0)

        self.entries.append(CacheEntry(
            query=query,
            query_embedding=embedding,
            response=response,
        ))

    def stats(self) -> dict:
        total_hits = sum(e.hit_count for e in self.entries)
        return {
            "entries": len(self.entries),
            "total_hits": total_hits,
        }


# Usage:
cache = SemanticCache(similarity_threshold=0.85)

# First query — cache miss
result = cache.get("How do I reset my password?")
if result is None:
    # Run full RAG pipeline
    answer = "Go to Settings > Security > Reset Password..."
    cache.put("How do I reset my password?", answer)
    print(f"Cache MISS → {answer}")

# Similar query — cache hit!
result = cache.get("How can I change my password?")
if result:
    print(f"Cache HIT → {result}")
```

---

## Production Code — TTL and FAISS-backed Cache

```python
"""
Production semantic cache with:
- FAISS for fast similarity search
- TTL (time-to-live) for cache expiry
- Thread-safe operations
"""
import numpy as np
import faiss
import time
import threading
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass, field


@dataclass
class CachedResponse:
    query: str
    response: str
    metadata: dict
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0


class ProductionSemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        ttl_seconds: int = 3600,       # 1 hour default
        max_entries: int = 10000,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.threshold = similarity_threshold
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self.lock = threading.Lock()

        # FAISS index for fast similarity search
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine on normalized)
        self.entries: list[CachedResponse] = []

    def get(self, query: str) -> tuple[str | None, dict]:
        """
        Look up query. Returns (response, metadata) or (None, {}).
        """
        query_embedding = self.model.encode(
            query, normalize_embeddings=True
        ).reshape(1, -1).astype("float32")

        with self.lock:
            if self.index.ntotal == 0:
                return None, {}

            # Search
            scores, indices = self.index.search(query_embedding, 1)
            best_score = scores[0][0]
            best_idx = indices[0][0]

            if best_score >= self.threshold and best_idx >= 0:
                entry = self.entries[best_idx]

                # Check TTL
                if time.time() - entry.created_at > self.ttl:
                    return None, {}  # Expired

                entry.hit_count += 1
                return entry.response, {
                    "cached_query": entry.query,
                    "similarity": float(best_score),
                    "cache_age_seconds": time.time() - entry.created_at,
                }

        return None, {}

    def put(self, query: str, response: str, metadata: dict | None = None):
        """Store response in cache."""
        embedding = self.model.encode(
            query, normalize_embeddings=True
        ).reshape(1, -1).astype("float32")

        with self.lock:
            # Evict expired entries periodically
            if len(self.entries) >= self.max_entries:
                self._evict_expired()

            self.index.add(embedding)
            self.entries.append(CachedResponse(
                query=query,
                response=response,
                metadata=metadata or {},
            ))

    def _evict_expired(self):
        """Remove expired entries. Rebuilds FAISS index."""
        now = time.time()
        valid = [(i, e) for i, e in enumerate(self.entries) if now - e.created_at < self.ttl]

        if len(valid) == len(self.entries):
            # Nothing to evict, remove oldest by hit count
            self.entries.sort(key=lambda e: e.hit_count)
            self.entries = self.entries[len(self.entries) // 4:]  # Keep top 75%
        else:
            self.entries = [e for _, e in valid]

        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dimension)
        if self.entries:
            embeddings = np.array([
                self.model.encode(e.query, normalize_embeddings=True)
                for e in self.entries
            ]).astype("float32")
            self.index.add(embeddings)

    def clear(self):
        with self.lock:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.entries.clear()
```

---

## With Libraries

### LangChain Caching

```python
"""
LangChain has built-in caching for LLM calls.
This caches the LLM response, not retrieval results.
"""
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache

# Exact string match cache (simplest)
set_llm_cache(InMemoryCache())

# Now any LLM call with identical input is cached
llm = ChatOpenAI(model="gpt-4o-mini")
response1 = llm.invoke("What is RAG?")   # Calls API
response2 = llm.invoke("What is RAG?")   # Returns cached (instant)
```

```python
# Redis-backed cache (persistent, production)
from langchain_community.cache import RedisCache
import redis

set_llm_cache(RedisCache(redis_client=redis.Redis(host="localhost", port=6379)))

# SQLite cache (persistent, no Redis needed)
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain_cache.db"))
```

### GPTCache (Semantic Caching Library)

```python
"""
GPTCache: dedicated semantic caching library.
Handles similarity matching, eviction, and persistence.
pip install gptcache
"""
from gptcache import Cache
from gptcache.embedding import Onnx
from gptcache.manager import manager_factory
from gptcache.similarity_evaluation.np import NumpyNormEvaluation
from gptcache.processor.pre import get_prompt

cache = Cache()

# Configure with ONNX embeddings + FAISS index
onnx = Onnx()
cache_manager = manager_factory(
    "sqlite,faiss",
    data_dir="cache_data",
    vector_params={"dimension": onnx.dimension},
)

cache.init(
    embedding_func=onnx.to_embeddings,
    data_manager=cache_manager,
    similarity_evaluation=NumpyNormEvaluation(),
    pre_embedding_func=get_prompt,
)

# Use with LangChain
from langchain_community.cache import GPTCache

set_llm_cache(GPTCache(init_gptcache_func=lambda _: cache))
```

---

## When NOT to Cache

| Situation                      | Why Caching Fails                              |
| ------------------------------ | ---------------------------------------------- |
| **Data changes frequently**    | Cached answers become stale                    |
| **User-specific answers**      | User A's answer ≠ User B's answer              |
| **Very diverse queries**       | Low hit rate, wasted storage                   |
| **Security-sensitive content** | Cached responses might leak across users       |
| **Real-time data**             | Stock prices, weather — always need fresh data |

### Cache Invalidation Strategies

```python
# 1. TTL-based (simplest)
cache = ProductionSemanticCache(ttl_seconds=3600)  # expire after 1 hour

# 2. Event-based (when data changes)
def on_document_updated(doc_id: str):
    """Clear cache entries that might reference updated document."""
    cache.clear()  # nuclear option
    # Better: clear only entries whose response references doc_id

# 3. Version-based
cache_key = f"{query}:v{data_version}"
```

---

## Common Pitfalls

| Pitfall                         | Impact                                        | Fix                                                     |
| ------------------------------- | --------------------------------------------- | ------------------------------------------------------- |
| Threshold too low (0.7)         | Wrong cache hits — returns irrelevant answers | Start at 0.85-0.90, tune down carefully                 |
| Threshold too high (0.95)       | Almost no cache hits                          | 0.85 is a good default for most use cases               |
| No TTL                          | Stale answers served forever                  | Always set TTL (1-24 hours typical)                     |
| Caching user-specific responses | Privacy leak across users                     | Scope cache by user_id or session                       |
| Not measuring hit rate          | Don't know if caching helps                   | Log hits/misses, target 30%+ hit rate                   |
| Caching before measuring        | Premature optimization                        | Profile costs first, cache if retrieval is >30% of cost |

---

## Syllabus Mapping

Not explicitly in `p2_rag_depth.md` but critical for **2.10 Cost, Latency & Stability**. Caching is a first-line defense against cost and latency problems in production RAG.
