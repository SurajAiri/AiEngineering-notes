# Advanced Embedding Strategies

## Why It Matters

Standard embedding models (like `all-MiniLM-L6-v2`) produce a single vector per chunk. This works for most use cases, but advanced embeddings unlock better retrieval quality, flexible storage, and domain-specific precision.

```
EVOLUTION OF EMBEDDING STRATEGIES:

  Level 1: Single dense vector per document
           all-MiniLM-L6-v2 → 384-dim vector
           ✅ Simple, fast
           ❌ One vector must capture ALL semantics

  Level 2: Late interaction (ColBERT)
           Token-level embeddings → compare per-token
           ✅ Much better matching quality
           ❌ More storage, more compute

  Level 3: Matryoshka embeddings
           Variable-dimension vectors from same model
           ✅ Use 64-dim for speed, 768-dim for precision
           ❌ Not all models support it

  Level 4: Fine-tuned embeddings
           Train on YOUR domain data
           ✅ Best quality for your specific use case
           ❌ Requires training data and compute

  Level 5: Multi-vector representations
           Multiple vectors per document (aspects, sentences)
           ✅ Captures multiple facets
           ❌ Complex indexing, more storage
```

---

## 1. Late-Interaction Models (ColBERT / ColPali)

### Concept

```
STANDARD EMBEDDING (bi-encoder):
  Document → [single 384-dim vector]
  Query    → [single 384-dim vector]
  Score = cosine(doc_vec, query_vec)

  Problem: One vector must represent EVERYTHING about the document.
  Nuanced queries get a coarse match.

ColBERT (late interaction):
  Document → [token₁_vec, token₂_vec, ..., tokenₙ_vec]  (N vectors)
  Query    → [token₁_vec, token₂_vec, ..., tokenₘ_vec]  (M vectors)

  Score = Σ max_j(sim(query_tokenᵢ, doc_tokenⱼ)) for each query token

  Each query token finds its BEST matching document token.
  Then scores are summed.

  ┌────────────────────────────────────┐
  │ Query: "FastAPI async performance" │
  │                                    │
  │ "FastAPI" ──────max sim──▶ "FastAPI"  (doc token)   = 0.95
  │ "async"   ──────max sim──▶ "asyncio"  (doc token)   = 0.88
  │ "perfor.." ─────max sim──▶ "speed"    (doc token)   = 0.72
  │                                    │
  │ Total = 0.95 + 0.88 + 0.72 = 2.55 │
  └────────────────────────────────────┘

WHY IT'S BETTER:
  Bi-encoder: "FastAPI async performance" → one vector → might miss "speed"
  ColBERT: Each query word matched independently → catches "speed" ↔ "performance"
```

### ColBERT Usage

```python
"""
ColBERT-style late interaction scoring.

Requirements: pip install colbert-ai torch
(Note: full ColBERT setup requires more dependencies)
"""

# ─── Simplified ColBERT scoring concept ───
import numpy as np


def colbert_score(
    query_token_embeddings: np.ndarray,   # (M, dim)
    doc_token_embeddings: np.ndarray,     # (N, dim)
) -> float:
    """
    Late-interaction scoring: for each query token,
    find the max-similarity document token. Sum the scores.

    This is the core ColBERT scoring mechanism.
    """
    # Similarity matrix: (M, N)
    sim_matrix = np.dot(query_token_embeddings, doc_token_embeddings.T)

    # For each query token, take max similarity across doc tokens
    max_sims = np.max(sim_matrix, axis=1)  # (M,)

    # Sum over query tokens
    return float(np.sum(max_sims))


# Example with random embeddings
np.random.seed(42)
query_tokens = np.random.randn(3, 128)   # 3 query tokens, 128-dim
doc_tokens = np.random.randn(10, 128)    # 10 doc tokens, 128-dim

# Normalize
query_tokens = query_tokens / np.linalg.norm(query_tokens, axis=1, keepdims=True)
doc_tokens = doc_tokens / np.linalg.norm(doc_tokens, axis=1, keepdims=True)

score = colbert_score(query_tokens, doc_tokens)
print(f"ColBERT score: {score:.3f}")
```

### ColPali — For Documents with Visual Content

```
ColPali = ColBERT + Vision Language Model

Instead of OCR → text → embed,
ColPali embeds DOCUMENT IMAGES directly.

  PDF Page (image) → Vision encoder → token-level embeddings
  Query (text) → Text encoder → token-level embeddings

  Scoring: Same late-interaction as ColBERT.

USE CASE:
  - Documents with complex layouts (tables, diagrams, forms)
  - Scanned documents where OCR is unreliable
  - PDFs where layout carries meaning (financial reports, etc.)
```

---

## 2. Matryoshka Embeddings (Variable Dimensions)

### Concept

```
STANDARD EMBEDDINGS:
  Model produces 768-dim vector → ALWAYS 768 dimensions.
  Want smaller? Too bad, retrain a different model.

MATRYOSHKA EMBEDDINGS:
  Same model, but first N dimensions are independently useful!

  768-dim: [v₁, v₂, v₃, ..., v₆₄, ..., v₂₅₆, ..., v₇₆₈]
            ├── 64-dim ──┤
            ├────── 256-dim ─────────┤
            ├──────────── 768-dim (full) ───────────────────┤

  The model is trained so that:
  - First 64 dims capture the most important semantics
  - First 256 dims capture more nuance
  - Full 768 dims capture everything

  NAMED AFTER: Russian nesting dolls (smaller versions inside bigger)

BENEFITS:
  ┌──────────────┬─────────┬──────────┬───────────┐
  │ Dimensions   │ Storage │ Speed    │ Quality   │
  ├──────────────┼─────────┼──────────┼───────────┤
  │ 64           │ 16×     │ ~10×     │ ~90-93%   │
  │              │ smaller │ faster   │ of full   │
  ├──────────────┼─────────┼──────────┼───────────┤
  │ 256          │ 3×      │ ~3×      │ ~97-98%   │
  │              │ smaller │ faster   │ of full   │
  ├──────────────┼─────────┼──────────┼───────────┤
  │ 768          │ full    │ baseline │ 100%      │
  └──────────────┴─────────┴──────────┴───────────┘
```

### Usage with Matryoshka Models

```python
"""
Matryoshka embeddings: use variable-dimension vectors from the same model.

Requirements: pip install sentence-transformers numpy faiss-cpu
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def matryoshka_embed(
    model: SentenceTransformer,
    texts: list[str],
    dimensions: int = 256,
) -> np.ndarray:
    """
    Generate Matryoshka embeddings truncated to `dimensions`.

    Works with models trained with Matryoshka loss:
    - nomic-embed-text-v1.5
    - mxbai-embed-large-v1
    - text-embedding-3-small (OpenAI)
    - text-embedding-3-large (OpenAI)
    """
    # Get full embeddings
    full_embeddings = model.encode(texts, normalize_embeddings=False)

    # Truncate to desired dimensions
    truncated = full_embeddings[:, :dimensions]

    # Re-normalize after truncation
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    normalized = truncated / norms

    return normalized


def compare_dimensions(
    model: SentenceTransformer,
    queries: list[str],
    documents: list[str],
    dim_options: list[int] = [64, 128, 256, 384],
):
    """
    Compare retrieval quality at different Matryoshka dimensions.
    """
    print(f"Comparing {len(dim_options)} dimension options:")
    print(f"  Queries: {len(queries)}, Documents: {len(documents)}")
    print()

    full_dim = model.get_sentence_embedding_dimension()

    # Get full embeddings as reference
    query_full = model.encode(queries, normalize_embeddings=True)
    doc_full = model.encode(documents, normalize_embeddings=True)
    reference_sims = np.dot(query_full, doc_full.T)

    for dim in dim_options:
        if dim > full_dim:
            continue

        query_emb = matryoshka_embed(model, queries, dimensions=dim)
        doc_emb = matryoshka_embed(model, documents, dimensions=dim)

        sims = np.dot(query_emb, doc_emb.T)

        # Compare rankings to full-dimension reference
        rank_agreement = 0
        for i in range(len(queries)):
            ref_top = np.argsort(reference_sims[i])[::-1][:3]
            trunc_top = np.argsort(sims[i])[::-1][:3]
            rank_agreement += len(set(ref_top) & set(trunc_top)) / 3

        rank_agreement /= len(queries)
        storage = dim * 4  # bytes per vector (float32)

        print(f"  {dim:4d} dims: rank agreement={rank_agreement:.1%}, "
              f"storage={storage:,} bytes/vec, "
              f"ratio={dim/full_dim:.1%} of full")


# ─── Two-Stage Search with Matryoshka ───
class MatryoshkaTwoStage:
    """
    Use small dimensions for fast initial search,
    then full dimensions for precise re-scoring.
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.documents: list[str] = []
        self.small_index = None  # for fast initial search
        self.full_embeddings = None  # for re-scoring

    def ingest(
        self,
        documents: list[str],
        small_dim: int = 64,
    ):
        """Index documents at two resolutions."""
        self.documents = documents

        # Full embeddings (stored for re-scoring, not indexed)
        self.full_embeddings = self.model.encode(
            documents, normalize_embeddings=True
        )

        # Small embeddings (indexed for fast search)
        small_emb = matryoshka_embed(self.model, documents, dimensions=small_dim)
        self.small_index = faiss.IndexFlatIP(small_dim)
        self.small_index.add(small_emb.astype(np.float32))
        self.small_dim = small_dim

    def search(
        self,
        query: str,
        initial_k: int = 50,
        final_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Two-stage search:
        1. Fast search with small dimensions → get candidates
        2. Re-score candidates with full dimensions → precise top-k
        """
        # Stage 1: Fast candidate retrieval
        query_small = matryoshka_embed(
            self.model, [query], dimensions=self.small_dim
        )
        _, candidate_indices = self.small_index.search(
            query_small.astype(np.float32),
            min(initial_k, len(self.documents)),
        )
        candidates = candidate_indices[0]

        # Stage 2: Re-score with full embeddings
        query_full = self.model.encode(query, normalize_embeddings=True)
        scores = []
        for idx in candidates:
            if idx < len(self.documents):
                score = float(np.dot(query_full, self.full_embeddings[idx]))
                scores.append((idx, score))

        # Sort by full-dimension score
        scores.sort(key=lambda x: x[1], reverse=True)

        return [
            (self.documents[idx], score)
            for idx, score in scores[:final_k]
        ]


# ─── Example ───
if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

    queries = ["How does authentication work?", "What is the timeout setting?"]
    docs = [
        "OAuth2 is used for API authentication with JWT tokens.",
        "The default request timeout is 30 seconds.",
        "Docker containers run in isolated namespaces.",
        "Rate limiting prevents API abuse.",
    ]

    compare_dimensions(model, queries, docs, [64, 128, 256, 384])
```

---

## 3. Domain-Specific Embedding Fine-Tuning

### Why Fine-Tune?

```
GENERIC EMBEDDINGS (pre-trained):
  "myocardial infarction" ←→ "heart attack"    sim = 0.65 (ok)
  "PE" ←→ "pulmonary embolism"                 sim = 0.30 (bad!)
  "stat" ←→ "immediately"                       sim = 0.20 (terrible!)

FINE-TUNED ON MEDICAL DATA:
  "myocardial infarction" ←→ "heart attack"    sim = 0.95
  "PE" ←→ "pulmonary embolism"                 sim = 0.90
  "stat" ←→ "immediately"                       sim = 0.85

WHEN TO FINE-TUNE:
  ✅ Domain has specialized vocabulary (medical, legal, finance)
  ✅ Abbreviations and jargon are common
  ✅ Generic models score <0.7 on your evaluation set
  ✅ You have enough training data (>1000 pairs)

WHEN NOT TO:
  ❌ Generic vocabulary (most tech documentation)
  ❌ Less than 100 training pairs
  ❌ Rapid iteration phase (fine-tuning is slow)
```

### Fine-Tuning with Sentence Transformers

```python
"""
Fine-tune an embedding model on domain-specific data.

Requirements: pip install sentence-transformers datasets
"""

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from datasets import Dataset


def prepare_training_data() -> Dataset:
    """
    Create training pairs from your domain.

    Three approaches:
    1. MANUAL: Domain experts create query-passage pairs
    2. SYNTHETIC: LLM generates queries for your passages
    3. MINING: Use existing search logs (query, clicked doc)
    """
    # Example: synthetic training data for API documentation
    training_pairs = [
        # (query, positive_passage)
        ("How to authenticate API requests",
         "OAuth2 authentication requires a valid JWT token in the Authorization header."),
        ("rate limit exceeded error",
         "When the rate limit of 1000 req/min is exceeded, the API returns HTTP 429."),
        ("configure database connection pooling",
         "Set POOL_SIZE=20 and MAX_OVERFLOW=10 in the database configuration."),
        ("deploy to kubernetes",
         "Use the provided Helm chart to deploy the application to a Kubernetes cluster."),
        # ... hundreds more pairs for real fine-tuning
    ]

    return Dataset.from_dict({
        "anchor": [p[0] for p in training_pairs],
        "positive": [p[1] for p in training_pairs],
    })


def fine_tune_embeddings(
    base_model: str = "all-MiniLM-L6-v2",
    output_dir: str = "fine-tuned-embeddings",
    epochs: int = 3,
    batch_size: int = 16,
):
    """Fine-tune an embedding model on domain data."""

    # Load base model
    model = SentenceTransformer(base_model)

    # Prepare data
    train_dataset = prepare_training_data()

    # Loss function: Multiple Negatives Ranking Loss
    # Best for (query, positive) pairs without explicit negatives
    loss = losses.MultipleNegativesRankingLoss(model)

    # Training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,
        save_total_limit=2,
    )

    # Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    trainer.train()
    model.save(output_dir)
    print(f"Model saved to {output_dir}")

    return model


def generate_synthetic_pairs(
    passages: list[str],
    num_queries_per_passage: int = 3,
) -> list[tuple[str, str]]:
    """
    Generate synthetic training data using an LLM.
    For each passage, generate queries that would retrieve it.
    """
    from openai import OpenAI
    import json

    client = OpenAI()
    pairs = []

    for passage in passages:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Generate {num_queries_per_passage} diverse search queries
that a user might type to find this passage:

Passage: {passage}

Return JSON: {{"queries": ["q1", "q2", "q3"]}}"""
            }],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        result = json.loads(response.choices[0].message.content)
        for query in result.get("queries", []):
            pairs.append((query, passage))

    return pairs


# ─── Evaluation ───
def evaluate_fine_tuned(
    base_model_name: str,
    fine_tuned_path: str,
    eval_queries: list[str],
    eval_docs: list[str],
    relevant_pairs: list[tuple[int, int]],  # (query_idx, doc_idx)
):
    """Compare base vs fine-tuned model on retrieval quality."""
    import numpy as np

    base = SentenceTransformer(base_model_name)
    fine_tuned = SentenceTransformer(fine_tuned_path)

    for name, model in [("Base", base), ("Fine-tuned", fine_tuned)]:
        q_emb = model.encode(eval_queries, normalize_embeddings=True)
        d_emb = model.encode(eval_docs, normalize_embeddings=True)
        sims = np.dot(q_emb, d_emb.T)

        # MRR
        mrr = 0
        for q_idx, d_idx in relevant_pairs:
            ranking = np.argsort(sims[q_idx])[::-1]
            rank = np.where(ranking == d_idx)[0][0] + 1
            mrr += 1.0 / rank
        mrr /= len(relevant_pairs)

        print(f"  {name}: MRR = {mrr:.3f}")
```

---

## 4. Multi-Vector Representations

### Concept

```
SINGLE VECTOR:
  "FastAPI supports async handlers and uses Pydantic for validation."
  → ONE 384-dim vector (must represent everything)

MULTI-VECTOR:
  Same document → MULTIPLE vectors, one per aspect:

  Vector 1: "FastAPI async handlers"    → captures async topic
  Vector 2: "Pydantic validation"       → captures validation topic
  Vector 3: "Python web framework"      → captures general category

  Retrieval: Query matches ANY of the vectors.
  If query is about async → Vector 1 scores highest.
  If query is about validation → Vector 2 scores highest.

APPROACHES:
  1. Sentence-level: One vector per sentence in the chunk
  2. Aspect-level: LLM extracts key topics, embed each
  3. Passage + summary: Embed both the passage and its summary
```

### Implementation

```python
"""
Multi-vector representations: multiple embeddings per document.

Requirements: pip install sentence-transformers faiss-cpu numpy
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass


@dataclass
class MultiVectorDoc:
    doc_id: str
    text: str
    sub_texts: list[str]  # sentence-level or aspect-level splits
    vector_ids: list[int]  # indices in the FAISS index


class MultiVectorIndex:
    """
    Index multiple vectors per document.
    For retrieval: find the best matching vector,
    then return the parent document.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.docs: list[MultiVectorDoc] = []
        self.vector_to_doc: dict[int, int] = {}  # vector_idx → doc_idx
        self.index = None
        self._next_vector_id = 0

    def add_document(
        self,
        doc_id: str,
        text: str,
        split_method: str = "sentence",
    ):
        """Add a document with multi-vector representation."""
        # Split into sub-texts
        if split_method == "sentence":
            import re
            sub_texts = [
                s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                if len(s.strip()) > 20
            ]
        else:
            # Could be aspect-level, paragraph-level, etc.
            sub_texts = [text]

        # Also include the full text as one vector
        sub_texts.append(text)

        # Track vector IDs
        vector_ids = list(range(
            self._next_vector_id,
            self._next_vector_id + len(sub_texts),
        ))

        doc = MultiVectorDoc(
            doc_id=doc_id,
            text=text,
            sub_texts=sub_texts,
            vector_ids=vector_ids,
        )

        doc_idx = len(self.docs)
        self.docs.append(doc)

        for vid in vector_ids:
            self.vector_to_doc[vid] = doc_idx

        self._next_vector_id += len(sub_texts)

    def build_index(self):
        """Build FAISS index from all sub-text vectors."""
        all_texts = []
        for doc in self.docs:
            all_texts.extend(doc.sub_texts)

        embeddings = self.model.encode(all_texts, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[tuple[str, float, str]]:
        """
        Search over all vectors, deduplicate to document level.

        Returns:
            [(doc_id, best_score, matched_sub_text)]
        """
        if self.index is None:
            self.build_index()

        query_emb = self.model.encode(query, normalize_embeddings=True)

        # Search more candidates (since multiple vectors per doc)
        search_k = min(k * 5, self.index.ntotal)
        scores, indices = self.index.search(
            query_emb.reshape(1, -1).astype(np.float32), search_k
        )

        # Deduplicate: keep best score per document
        seen_docs = {}
        for score, vec_idx in zip(scores[0], indices[0]):
            if vec_idx < 0:
                continue
            doc_idx = self.vector_to_doc.get(int(vec_idx))
            if doc_idx is None:
                continue
            doc = self.docs[doc_idx]

            if doc.doc_id not in seen_docs or score > seen_docs[doc.doc_id][0]:
                # Find which sub-text matched
                local_idx = int(vec_idx) - doc.vector_ids[0]
                matched_text = doc.sub_texts[local_idx] if local_idx < len(doc.sub_texts) else ""
                seen_docs[doc.doc_id] = (float(score), matched_text)

        # Sort by score and return top-k
        results = [
            (doc_id, score, matched)
            for doc_id, (score, matched) in seen_docs.items()
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# ─── Usage ───
if __name__ == "__main__":
    idx = MultiVectorIndex()

    idx.add_document(
        "doc_1",
        "FastAPI supports async request handling with Python's asyncio. "
        "It uses Pydantic for data validation. "
        "OpenAPI documentation is generated automatically at /docs.",
    )
    idx.add_document(
        "doc_2",
        "Django provides an ORM for database access. "
        "It includes a built-in admin panel. "
        "Template rendering uses the Jinja2-like Django template language.",
    )

    idx.build_index()

    # Query that matches a specific aspect of doc_1
    results = idx.search("how does data validation work?")
    for doc_id, score, matched in results:
        print(f"  [{doc_id}] score={score:.3f}")
        print(f"    Matched: {matched[:80]}")
```

---

## Comparison Matrix

```
┌─────────────────────┬──────────┬──────────┬──────────┬───────────┐
│ Strategy            │ Quality  │ Storage  │ Speed    │ Complexity│
├─────────────────────┼──────────┼──────────┼──────────┼───────────┤
│ Bi-encoder (base)   │ Good     │ Low      │ Fast     │ Low       │
│ ColBERT             │ Best     │ High     │ Medium   │ High      │
│ Matryoshka          │ Good     │ Flexible │ Flexible │ Low       │
│ Fine-tuned          │ Best for │ Low      │ Fast     │ Medium    │
│                     │ domain   │          │          │           │
│ Multi-vector        │ Very Good│ Medium   │ Medium   │ Medium    │
└─────────────────────┴──────────┴──────────┴──────────┴───────────┘

DECISION GUIDE:
  Start with → bi-encoder (all-MiniLM-L6-v2)
  Need better quality → fine-tune on your data
  Need flexibility → Matryoshka embeddings
  Need best quality → ColBERT (if you can afford storage)
  Complex documents → Multi-vector representations
```

---

## Pitfalls & Common Mistakes

| Mistake                                          | Impact                                   | Fix                                                                    |
| ------------------------------------------------ | ---------------------------------------- | ---------------------------------------------------------------------- |
| **Fine-tuning with <100 pairs**                  | Overfitting, worse than base model       | Use 1000+ pairs; generate synthetic data if needed                     |
| **Not evaluating after fine-tuning**             | May have made things worse               | Always compare base vs fine-tuned on held-out eval set                 |
| **Truncating Matryoshka without re-normalizing** | Similarity scores are wrong              | Always L2-normalize after truncation                                   |
| **ColBERT storage explosion**                    | 128 vectors per doc × millions of docs   | Use compression; consider hybrid (ColBERT re-rank only)                |
| **Multi-vector without dedup**                   | Same doc returned multiple times         | Deduplicate results at document level                                  |
| **Choosing advanced embeddings prematurely**     | Over-engineering before basics are solid | Start with bi-encoder; switch only when eval shows it's the bottleneck |

---

## Key Takeaways

1. **Start simple** — `all-MiniLM-L6-v2` or OpenAI `text-embedding-3-small` handles most cases.
2. **Fine-tune when domain vocabulary is specialized** — biggest quality gain for domain-specific RAG.
3. **Matryoshka gives storage/speed flexibility** — use small dims for fast search, full dims for precision.
4. **ColBERT is the quality ceiling** — best retrieval quality, but storage-expensive.
5. **Multi-vector captures multiple aspects** — useful for long, multi-topic documents.
6. **Always evaluate** — advanced ≠ better for your specific use case.
