# Embedding Basics

> **Prerequisite for RAG retrieval.** If you don't understand embeddings, vector search won't make sense.

## What Are Embeddings?

An embedding is a **numerical representation** of text — an array of numbers (a vector) that captures the meaning of the text. Similar texts produce similar vectors.

```
Text:       "The cat sat on the mat"
Embedding:  [0.12, -0.34, 0.56, 0.78, -0.11, ...]   ← 384 to 3072 numbers

Text:       "A kitten rested on a rug"
Embedding:  [0.11, -0.32, 0.58, 0.75, -0.13, ...]   ← SIMILAR numbers!

Text:       "Stock prices fell sharply today"
Embedding:  [-0.45, 0.67, -0.23, 0.12, 0.89, ...]   ← DIFFERENT numbers
```

---

## Why Embeddings Work for RAG

```
TRADITIONAL SEARCH (keyword matching):
  Query: "How to fix authentication errors?"
  Doc:   "Resolving login failures in the SSO module"
  Match: ❌ No shared keywords!

EMBEDDING SEARCH (semantic matching):
  Query embedding: [0.45, 0.12, -0.33, ...]
  Doc embedding:   [0.43, 0.14, -0.31, ...]
  Cosine similarity: 0.94 → HIGH MATCH ✅

Embeddings understand that "authentication errors" ≈ "login failures"
```

---

## How Embedding Models Work (Simplified)

```
                    Embedding Model
                    (Neural Network)
                         │
    "Machine learning"  ──►  [0.12, -0.34, 0.56, ...]
                         │
                    The model was trained on
                    billions of text pairs to
                    learn that similar meanings
                    → similar vectors
```

You don't need to understand the internals. Just know:

1. **Input:** A string of text (sentence, paragraph, or chunk)
2. **Output:** A fixed-size array of floats (the "embedding")
3. **Property:** Semantically similar inputs → similar outputs

---

## Similarity Metrics

```
Given two vectors A and B:

COSINE SIMILARITY (most common for text):
  sim = A·B / (|A| × |B|)
  Range: -1 to 1  (1 = identical direction, 0 = unrelated)
  Use this when: You care about meaning, not length

DOT PRODUCT:
  sim = A·B
  Range: -∞ to +∞
  Use this when: Vectors are already normalized (then = cosine)

EUCLIDEAN DISTANCE (L2):
  dist = √Σ(Aᵢ - Bᵢ)²
  Range: 0 to +∞ (0 = identical)
  Use this when: Magnitude matters (rare in NLP)
```

**In practice:** Most embedding models output normalized vectors, so cosine similarity = dot product. Just use cosine similarity as your default.

---

## Simple Code — Creating and Comparing Embeddings

```python
"""
Create embeddings and compute similarity.
Requirements: pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Load a free, open-source embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions

# Create embeddings for some texts
texts = [
    "How to reset my password",
    "Steps to change login credentials",
    "The weather forecast for tomorrow",
]

embeddings = model.encode(texts)

print(f"Shape: {embeddings.shape}")  # (3, 384) — 3 texts, 384 dimensions each

# Compute cosine similarity between texts
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"'password reset' vs 'login credentials': {cosine_sim(embeddings[0], embeddings[1]):.3f}")
# ~0.75 — HIGH similarity (same topic)

print(f"'password reset' vs 'weather forecast':  {cosine_sim(embeddings[0], embeddings[2]):.3f}")
# ~0.10 — LOW similarity (different topics)
```

### Using OpenAI Embeddings

```python
"""
OpenAI's embedding API — paid but high quality.
Requirements: pip install openai
"""

from openai import OpenAI

client = OpenAI()  # needs OPENAI_API_KEY env var

response = client.embeddings.create(
    model="text-embedding-3-small",  # 1536 dimensions, cheap
    input=["How to reset my password", "Steps to change login credentials"]
)

embedding_1 = response.data[0].embedding  # list of 1536 floats
embedding_2 = response.data[1].embedding

# Compute similarity
import numpy as np
sim = np.dot(embedding_1, embedding_2)  # OpenAI embeddings are normalized
print(f"Similarity: {sim:.3f}")
```

---

## Popular Embedding Models (2024-2026)

```
┌────────────────────────────┬───────┬────────┬───────────┬──────────┐
│ Model                      │ Dims  │ Free?  │ Quality   │ Speed    │
├────────────────────────────┼───────┼────────┼───────────┼──────────┤
│ all-MiniLM-L6-v2           │ 384   │ ✅ Yes │ Good      │ Fast     │
│ all-mpnet-base-v2          │ 768   │ ✅ Yes │ Better    │ Medium   │
│ text-embedding-3-small     │ 1536  │ ❌ Paid│ Very Good │ Fast     │
│ text-embedding-3-large     │ 3072  │ ❌ Paid│ Best(OAI) │ Medium   │
│ embed-v3 (Cohere)          │ 1024  │ ❌ Paid│ Very Good │ Fast     │
│ voyage-large-2 (Voyage)    │ 1536  │ ❌ Paid│ Excellent │ Medium   │
│ BGE-large-en-v1.5          │ 1024  │ ✅ Yes │ Very Good │ Medium   │
│ E5-mistral-7b-instruct     │ 4096  │ ✅ Yes │ Excellent │ Slow     │
└────────────────────────────┴───────┴────────┴───────────┴──────────┘

RECOMMENDATION:
  Start with: all-MiniLM-L6-v2 (free, fast, good enough)
  Production: text-embedding-3-small (low cost, high quality)
  Best quality: voyage-large-2 or E5-mistral-7b-instruct
```

---

## Key Concepts for RAG

### 1. Same Model for Queries and Documents

You MUST use the **same embedding model** for both documents and queries. Mixing models produces vectors in different spaces — similarity scores become meaningless.

### 2. Chunk Size Affects Embedding Quality

- Short text → precise embedding of that specific idea
- Long text → averaged embedding of many ideas (diluted)
- This is why chunking matters — you're choosing what each vector represents

### 3. Embeddings Are Not Perfect

```
THINGS EMBEDDINGS HANDLE WELL:
  ✅ Synonyms:     "car" ≈ "automobile"
  ✅ Paraphrases:  "How to fix X" ≈ "Steps to resolve X"
  ✅ Related topics: "Python" ≈ "programming"

THINGS EMBEDDINGS STRUGGLE WITH:
  ❌ Negation:     "is fast" ≈ "is not fast" (similar embeddings!)
  ❌ Exact values:  "API key: abc123" — exact match is better
  ❌ Acronyms:     "HNSW" may not match "Hierarchical Navigable Small World"
  ❌ Domain jargon: Medical/legal terms may not embed well
```

This is why production RAG uses **hybrid search** (vectors + keyword matching).

---

## Common Questions

### Q: Do I need a GPU to create embeddings?

**A:** For `all-MiniLM-L6-v2` and similar small models — no, CPU is fine. For larger models or millions of documents, a GPU speeds things up significantly, but it's not required to get started.

### Q: How much do embeddings cost?

**A:** Open-source models (sentence-transformers) = free. OpenAI `text-embedding-3-small` = ~$0.02 per 1M tokens. For 10,000 document chunks of ~200 tokens each, that's ~$0.04.

### Q: Can I change embedding models later?

**A:** Yes, but you'll need to **re-embed all your documents**. Embeddings from different models live in different vector spaces and can't be mixed. Plan for this in your architecture.

### Q: What dimensions should I use?

**A:** Higher dimensions = more expressive but more storage/compute. 384-1536 is the sweet spot for most use cases. Start with whatever your chosen model outputs — don't overthink this early on.

---

## What's Next?

Now that you understand embeddings:

- **[End-to-End RAG Example](./03_rag_pipeline_end_to_end.md)** — See embeddings in action in a full pipeline
- **[Vector Similarity Search](../03_retrieval/01_vector_similarity_search.md)** — Deep dive into how search works with embeddings
