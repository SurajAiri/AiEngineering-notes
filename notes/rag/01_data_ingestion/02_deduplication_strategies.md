# Deduplication Strategies

## Why This Matters

Duplicate documents in a RAG system cause:

- **Wasted embedding compute** — you embed the same information multiple times
- **Biased retrieval** — duplicates get higher aggregate scores, crowding out unique content
- **Token waste** — the LLM receives redundant context chunks
- **Contradictory answers** — slightly different versions of the same doc can confuse generation

---

## The Three Levels of Deduplication

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Level 1: EXACT DEDUP                                         │
│   "Are these byte-for-byte identical?"                          │
│   → Hash comparison (MD5, SHA-256)                              │
│   → Cheapest, fastest, catches copy-paste duplicates           │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Level 2: NEAR-DUPLICATE                                       │
│   "Are these almost the same with minor differences?"           │
│   → MinHash / SimHash / edit distance                           │
│   → Catches reformatted, slightly edited versions              │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Level 3: SEMANTIC DEDUP                                       │
│   "Do these say the same thing in different words?"             │
│   → Embedding similarity comparison                             │
│   → Catches paraphrases, summaries, rewritten content          │
│   → Most expensive, most nuanced                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Level 1: Exact Deduplication

### Concept

Compare cryptographic hashes of documents. If two documents produce the same hash, they're identical.

### Simple Code

```python
import hashlib

def get_hash(text: str) -> str:
    """Get SHA-256 hash of text content."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

documents = [
    "Machine learning is a subset of AI.",
    "Deep learning uses neural networks.",
    "Machine learning is a subset of AI.",   # exact duplicate
    "Natural language processing is an AI field.",
]

seen_hashes = set()
unique_docs = []

for doc in documents:
    h = get_hash(doc)
    if h not in seen_hashes:
        seen_hashes.add(h)
        unique_docs.append(doc)
    else:
        print(f"DUPLICATE: {doc[:50]}...")

print(f"\n{len(documents)} docs → {len(unique_docs)} unique")
# DUPLICATE: Machine learning is a subset of AI....
# 4 docs → 3 unique
```

### Production Code — Exact Dedup with Tracking

```python
"""
Exact deduplication with source tracking and reporting.
Handles documents at ingestion time and keeps an audit trail.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Document:
    content: str
    source: str
    doc_id: str


@dataclass
class DedupResult:
    unique: list[Document]
    duplicates: list[tuple[Document, str]]  # (duplicate_doc, original_doc_id)
    total_input: int
    total_unique: int


class ExactDeduplicator:
    """Tracks document hashes across ingestion batches."""

    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self._hash_to_doc_id: dict[str, str] = {}

    def _compute_hash(self, text: str) -> str:
        if self.normalize:
            # Normalize before hashing so that whitespace-only changes
            # are treated as duplicates
            text = ' '.join(text.split())
            text = text.lower().strip()
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def deduplicate(self, documents: list[Document]) -> DedupResult:
        unique = []
        duplicates = []

        for doc in documents:
            h = self._compute_hash(doc.content)
            if h not in self._hash_to_doc_id:
                self._hash_to_doc_id[h] = doc.doc_id
                unique.append(doc)
            else:
                original_id = self._hash_to_doc_id[h]
                duplicates.append((doc, original_id))
                logger.info(
                    f"Exact duplicate: {doc.doc_id} (source={doc.source}) "
                    f"== {original_id}"
                )

        return DedupResult(
            unique=unique,
            duplicates=duplicates,
            total_input=len(documents),
            total_unique=len(unique),
        )


# Usage
if __name__ == "__main__":
    dedup = ExactDeduplicator(normalize=True)

    docs = [
        Document("Vector databases store embeddings.", "file1.pdf", "doc_001"),
        Document("Vector  databases  store  embeddings.", "file2.pdf", "doc_002"),  # whitespace diff
        Document("Graph databases store relationships.", "file3.pdf", "doc_003"),
    ]

    result = dedup.deduplicate(docs)
    print(f"Input: {result.total_input}, Unique: {result.total_unique}")
    for dup_doc, orig_id in result.duplicates:
        print(f"  '{dup_doc.doc_id}' is a duplicate of '{orig_id}'")
```

---

## Level 2: Near-Duplicate Detection

### Concept

Near-duplicates are documents that differ only slightly — maybe different formatting, a typo fix, or a minor edit. Exact hashing won't catch these.

**MinHash + LSH (Locality-Sensitive Hashing)** is the standard approach:

```
Document → Shingling → MinHash Signature → LSH Buckets
                │              │                 │
        Break text into    Create compact     Group similar
        overlapping         fingerprint       docs into same
        n-grams            (fixed-size         bucket for
                            hash array)        comparison
```

### How MinHash Works (Intuitive)

```
Doc A: "the cat sat on the mat"
Doc B: "the cat sat on a mat"

Shingles (3-grams):
Doc A: {"the cat sat", "cat sat on", "sat on the", "on the mat"}
Doc B: {"the cat sat", "cat sat on", "sat on a",   "on a mat"}

Jaccard Similarity = |A ∩ B| / |A ∪ B| = 2 / 6 = 0.33

MinHash creates a compact signature that ESTIMATES Jaccard similarity
without computing all pairwise shingle comparisons.
```

### Simple Code — Shingling + Jaccard

```python
def get_shingles(text: str, k: int = 3) -> set[str]:
    """Create character or word shingles from text."""
    words = text.lower().split()
    return {' '.join(words[i:i+k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union)


# Example
doc_a = "The vector database stores high-dimensional embeddings for similarity search"
doc_b = "The vector database stores high-dimensional embeddings for nearest neighbor search"
doc_c = "Kubernetes orchestrates containerized applications across clusters"

shingles_a = get_shingles(doc_a)
shingles_b = get_shingles(doc_b)
shingles_c = get_shingles(doc_c)

print(f"A vs B: {jaccard_similarity(shingles_a, shingles_b):.2f}")  # ~0.67 (near-dup)
print(f"A vs C: {jaccard_similarity(shingles_a, shingles_c):.2f}")  # ~0.00 (different)
```

### Production Code — MinHash Deduplication

```python
"""
Near-duplicate detection using MinHash + LSH.
Scales to millions of documents efficiently.

Requirements: pip install datasketch
"""

from datasketch import MinHash, MinHashLSH
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Document:
    content: str
    source: str
    doc_id: str


class NearDuplicateDetector:
    """
    Detect near-duplicate documents using MinHash LSH.

    Parameters:
        threshold: Jaccard similarity threshold for duplicates (0.0 to 1.0)
        num_perm: Number of permutations for MinHash (higher = more accurate, slower)
        shingle_size: Number of words per shingle
    """

    def __init__(
        self,
        threshold: float = 0.8,
        num_perm: int = 128,
        shingle_size: int = 3,
    ):
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._minhashes: dict[str, MinHash] = {}

    def _text_to_minhash(self, text: str) -> MinHash:
        words = text.lower().split()
        shingles = [
            ' '.join(words[i:i + self.shingle_size])
            for i in range(len(words) - self.shingle_size + 1)
        ]
        mh = MinHash(num_perm=self.num_perm)
        for s in shingles:
            mh.update(s.encode('utf-8'))
        return mh

    def deduplicate(self, documents: list[Document]) -> tuple[list[Document], list[tuple[Document, list[str]]]]:
        """
        Returns:
            (unique_docs, duplicate_pairs)
            where duplicate_pairs = [(doc, [list of similar doc_ids]), ...]
        """
        unique = []
        duplicates = []

        for doc in documents:
            mh = self._text_to_minhash(doc.content)

            # Query LSH for existing near-duplicates
            similar_ids = self.lsh.query(mh)

            if similar_ids:
                duplicates.append((doc, similar_ids))
                logger.info(
                    f"Near-duplicate: {doc.doc_id} similar to {similar_ids}"
                )
            else:
                # No near-duplicates found, add to index
                self.lsh.insert(doc.doc_id, mh)
                self._minhashes[doc.doc_id] = mh
                unique.append(doc)

        return unique, duplicates

    def get_similarity(self, doc_id_a: str, doc_id_b: str) -> float:
        """Get estimated Jaccard similarity between two indexed documents."""
        mh_a = self._minhashes.get(doc_id_a)
        mh_b = self._minhashes.get(doc_id_b)
        if mh_a is None or mh_b is None:
            raise ValueError("One or both doc_ids not in index")
        return mh_a.jaccard(mh_b)


# Usage
if __name__ == "__main__":
    detector = NearDuplicateDetector(threshold=0.5, num_perm=128)

    docs = [
        Document(
            "A vector database is designed to store and query high-dimensional "
            "vector embeddings efficiently using approximate nearest neighbor search.",
            "blog_v1.html", "doc_001"
        ),
        Document(
            "A vector database is designed to store and query high-dimensional "
            "vector embeddings efficiently using approximate nearest neighbor algorithms.",
            "blog_v2.html", "doc_002"  # near-duplicate (one word changed)
        ),
        Document(
            "Kubernetes is an open-source container orchestration platform that "
            "automates deployment, scaling, and management of containerized apps.",
            "k8s_intro.html", "doc_003"  # completely different
        ),
    ]

    unique, duplicates = detector.deduplicate(docs)
    print(f"Unique: {len(unique)}, Duplicates: {len(duplicates)}")
    for dup_doc, similar_ids in duplicates:
        print(f"  '{dup_doc.doc_id}' is near-duplicate of {similar_ids}")
```

---

## Level 3: Semantic Deduplication

### Concept

Two documents can say the same thing in completely different words. Semantic dedup uses embedding similarity to detect these.

```
"The server crashed due to OOM"          ──┐
                                            ├── Cosine similarity: 0.92 → DUPLICATE
"The machine went down because it ran    ──┘
 out of memory"

"The server handles 10k requests/sec"    ──── Different topic → similarity: 0.31
```

### Simple Code

```python
"""
Semantic dedup using sentence embeddings.
Requirements: pip install sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "The application crashed because it ran out of memory.",
    "An out-of-memory error caused the app to crash.",  # semantic duplicate
    "We need to increase the pod memory limits in Kubernetes.",  # related but different
    "The deployment pipeline uses GitHub Actions for CI/CD.",  # unrelated
]

embeddings = model.encode(documents)

# Compute pairwise cosine similarity
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

threshold = 0.85
seen = []
unique_indices = []

for i, emb in enumerate(embeddings):
    is_dup = False
    for j in seen:
        sim = cosine_sim(emb, embeddings[j])
        if sim > threshold:
            print(f"SEMANTIC DUP: [{i}] similar to [{j}] (sim={sim:.3f})")
            is_dup = True
            break
    if not is_dup:
        seen.append(i)
        unique_indices.append(i)

print(f"\nUnique docs: {[documents[i][:50] for i in unique_indices]}")
```

### Production Code — Semantic Dedup at Scale

```python
"""
Semantic deduplication using embeddings with batch processing.
For large corpora, uses FAISS for efficient similarity search.

Requirements: pip install sentence-transformers faiss-cpu numpy
"""

import numpy as np
import faiss
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class Document:
    content: str
    source: str
    doc_id: str


class SemanticDeduplicator:
    """
    Embedding-based semantic deduplication.
    Uses FAISS for efficient nearest-neighbor lookup.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.90,
        batch_size: int = 256,
    ):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.dimension = self.model.get_sentence_embedding_dimension()

        # Use inner product index (for normalized vectors, IP = cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        self._doc_ids: list[str] = []

    def deduplicate(
        self, documents: list[Document]
    ) -> tuple[list[Document], list[tuple[Document, str, float]]]:
        """
        Returns:
            unique_docs: documents that are not semantic duplicates
            duplicates: list of (doc, matched_doc_id, similarity_score)
        """
        unique = []
        duplicates = []

        # Process in batches for memory efficiency
        for batch_start in range(0, len(documents), self.batch_size):
            batch = documents[batch_start:batch_start + self.batch_size]
            texts = [doc.content for doc in batch]

            # Encode batch
            embeddings = self.model.encode(
                texts,
                normalize_embeddings=True,  # L2 normalize for cosine similarity
                show_progress_bar=False,
                batch_size=self.batch_size,
            )
            embeddings = np.array(embeddings, dtype=np.float32)

            for i, (doc, emb) in enumerate(zip(batch, embeddings)):
                emb_reshaped = emb.reshape(1, -1)

                if self.index.ntotal > 0:
                    # Search for similar existing documents
                    scores, indices = self.index.search(emb_reshaped, k=1)
                    best_score = float(scores[0][0])
                    best_idx = int(indices[0][0])

                    if best_score >= self.similarity_threshold:
                        matched_id = self._doc_ids[best_idx]
                        duplicates.append((doc, matched_id, best_score))
                        logger.info(
                            f"Semantic duplicate: {doc.doc_id} ≈ {matched_id} "
                            f"(similarity={best_score:.3f})"
                        )
                        continue

                # Not a duplicate — add to index
                self.index.add(emb_reshaped)
                self._doc_ids.append(doc.doc_id)
                unique.append(doc)

        logger.info(
            f"Semantic dedup: {len(documents)} input → "
            f"{len(unique)} unique, {len(duplicates)} duplicates"
        )
        return unique, duplicates


# Usage
if __name__ == "__main__":
    dedup = SemanticDeduplicator(similarity_threshold=0.85)

    docs = [
        Document(
            "Machine learning algorithms learn patterns from training data "
            "to make predictions on new, unseen data.",
            "ml_intro.pdf", "doc_001"
        ),
        Document(
            "ML models are trained on datasets to recognize patterns and "
            "then apply those patterns to predict outcomes on fresh data.",
            "ml_guide.pdf", "doc_002"  # semantic duplicate
        ),
        Document(
            "Docker containers package applications with their dependencies "
            "for consistent deployment across environments.",
            "docker_basics.pdf", "doc_003"  # different topic
        ),
    ]

    unique, duplicates = dedup.deduplicate(docs)
    print(f"Unique: {len(unique)}")
    for dup, matched_id, score in duplicates:
        print(f"  '{dup.doc_id}' ≈ '{matched_id}' (score={score:.3f})")
```

---

## Combining All Three Levels

The best production systems use a layered approach:

```
Documents
    │
    ▼
┌──────────────┐   Cheapest, fastest
│ Exact Dedup  │   Hash comparison — O(1) per doc
│ (SHA-256)    │   Catches: copy-paste duplicates, re-downloads
└──────┬───────┘
       │ unique docs only
       ▼
┌──────────────┐   Moderate cost
│ Near-Dup     │   MinHash + LSH — O(1) amortized per doc
│ (MinHash)    │   Catches: reformatted, slightly edited versions
└──────┬───────┘
       │ unique docs only
       ▼
┌──────────────┐   Most expensive
│ Semantic Dup │   Embedding + FAISS — O(log n) per doc
│ (Embeddings) │   Catches: paraphrases, rewrites, summaries
└──────┬───────┘
       │
       ▼
  Deduplicated Corpus
```

---

## Where to Deduplicate

```
                    DOCUMENT LEVEL              CHUNK LEVEL

When:               Before chunking             After chunking
Catches:            Whole-doc duplicates         Cross-doc chunk overlap
Cost:               Lower (fewer items)          Higher (many chunks)
Importance:         Essential                    Nice-to-have

Example:            Same PDF uploaded twice      Two docs sharing an
                                                 identical introduction
                                                 paragraph
```

**Do both when possible** — document-level dedup first (cheaper), then optionally chunk-level dedup.

---

## Pitfalls & Common Mistakes

| Mistake                             | Impact                                             | Fix                                                                 |
| ----------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------- |
| **Only doing exact dedup**          | Misses reformatted duplicates, version-edited docs | Add MinHash layer                                                   |
| **Threshold too aggressive**        | Removes documents that are related but distinct    | Start at 0.9 for semantic, 0.8 for MinHash; tune with manual review |
| **Dedup after embedding**           | Wasted compute on embedding duplicates             | Dedup before embedding                                              |
| **No normalization before hashing** | Whitespace-only changes create "unique" hashes     | Normalize text before hashing                                       |
| **Not tracking dedup decisions**    | Can't debug why a document was dropped             | Log every dedup decision with source IDs                            |
| **Deduplicating across domains**    | Different contexts may legitimately repeat text    | Scope dedup within document collections                             |

---

## Trade-offs

| Method                | Speed      | Accuracy          | Catches                |
| --------------------- | ---------- | ----------------- | ---------------------- |
| Exact (hash)          | ⚡ Fastest | 100% for exact    | Only byte-identical    |
| Near-dup (MinHash)    | 🚀 Fast    | ~95% at threshold | Reformats, minor edits |
| Semantic (embeddings) | 🐢 Slowest | ~85-95%           | Paraphrases, rewrites  |

**Cost reality:**

- 1M documents × exact dedup = seconds
- 1M documents × MinHash = minutes
- 1M documents × semantic dedup = hours (depends on GPU)

---

## Key Takeaways

1. **Layer your dedup** — exact first, then near-dup, then semantic.
2. **Dedup before chunking and embedding** to save compute.
3. **Always track what was deduplicated** and why — you will need to debug this.
4. **Tune thresholds conservatively** — it's better to keep a near-duplicate than to lose a unique document.
5. **Normalize before comparing** — whitespace, casing, and encoding differences create false negatives.
