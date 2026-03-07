# Hypothetical Document Embeddings (HyDE)

## Why It Matters

There's a fundamental asymmetry in RAG: **queries are short questions**, but **documents are long answers**. Their embeddings live in different regions of the vector space. HyDE bridges this gap by generating a **hypothetical answer** to the query, then using that answer's embedding to search — because an answer is more similar to other answers than a question is.

---

## Core Concept

```
STANDARD RETRIEVAL:
  Query: "How does HNSW work?"
  → Embed the question  → Search → Results may be poor
    (question ≠ answer in embedding space)

HYDE RETRIEVAL:
  Query: "How does HNSW work?"
  → LLM generates a hypothetical answer:
    "HNSW (Hierarchical Navigable Small World) is a graph-based
     approximate nearest neighbor algorithm. It builds a multi-layer
     graph where each layer is a navigable small world graph..."
  → Embed the hypothetical answer → Search → Much better results!
    (hypothetical answer ≈ real answer in embedding space)

┌────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────┐
│   Query    │ ──► │    LLM      │ ──► │   Embed     │ ──► │Search │
│            │     │ (generate   │     │ hypothetical│     │       │
│ "How does  │     │  hypothetical│    │  answer     │     │       │
│  HNSW      │     │  answer)    │     │             │     │       │
│  work?"    │     └─────────────┘     └─────────────┘     └───────┘
└────────────┘                                                │
                                                              ▼
                                                        Real documents
                                                        about HNSW
```

### Why It Works

```
In embedding space:

  "How does HNSW work?"  ·                    ← Question embedding
                                                (far from answers)

                                    · Real doc about HNSW

                               · Hypothetical answer about HNSW  ← Close to real doc!

  The hypothetical answer shares vocabulary, structure, and
  concepts with the real documents, even if it's factually wrong.

  KEY INSIGHT: The hypothetical document doesn't need to be
  correct — it just needs to be in the right neighborhood
  of the embedding space.
```

---

## Simple Code — Basic HyDE

```python
"""
Minimal HyDE implementation to understand the concept.
"""

from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


def hyde_search(
    query: str,
    documents: list[str],
    model: SentenceTransformer,
    llm_client: OpenAI,
    k: int = 3,
) -> list[tuple[int, float]]:
    """
    HyDE: Generate a hypothetical answer, embed it, then search.
    """
    # Step 1: Generate hypothetical document
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": (
                "Write a short passage that answers the question. "
                "Write as if it were a paragraph from a technical document. "
                "Do not say 'I think' or 'I believe'. Just state facts."
            )},
            {"role": "user", "content": query},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    hypothetical_doc = response.choices[0].message.content
    print(f"Hypothetical doc: {hypothetical_doc[:100]}...")

    # Step 2: Embed the hypothetical document (NOT the query)
    hyde_embedding = model.encode(hypothetical_doc, normalize_embeddings=True)
    hyde_embedding = np.array([hyde_embedding], dtype=np.float32)

    # Step 3: Search with the hypothetical doc embedding
    doc_embeddings = model.encode(documents, normalize_embeddings=True)
    doc_embeddings = np.array(doc_embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)

    distances, indices = index.search(hyde_embedding, k)
    return [(int(idx), float(dist)) for idx, dist in zip(indices[0], distances[0])]


# Example
model = SentenceTransformer("all-MiniLM-L6-v2")
client = OpenAI()

documents = [
    "HNSW builds a hierarchical graph where upper layers have fewer nodes for fast coarse search",
    "BM25 scores documents using term frequency and inverse document frequency",
    "The HNSW algorithm uses skip connections to enable logarithmic search complexity",
    "Python is a popular language for data science",
    "Approximate nearest neighbor search trades accuracy for speed",
]

results = hyde_search("How does HNSW work?", documents, model, client)
for idx, score in results:
    print(f"  [{score:.4f}] {documents[idx]}")
```

---

## Production Code — HyDE with Caching and Fallback

```python
"""
Production HyDE implementation with:
- Caching to avoid redundant LLM calls
- Fallback to standard search if LLM fails
- Configurable generation prompts per domain
- Optional multi-hypothesis (generate N hypothetical docs)

Requirements: pip install openai sentence-transformers faiss-cpu numpy
"""

import hashlib
import json
import logging
import numpy as np
import faiss
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class HyDEResult:
    index: int
    score: float
    text: str
    used_hyde: bool


class HyDERetriever:
    """
    Hypothetical Document Embeddings retriever.

    Generates hypothetical answers and uses their embeddings
    for retrieval. Falls back to standard search on LLM failure.
    """

    SYSTEM_PROMPTS = {
        "general": (
            "Write a short passage (3-5 sentences) that directly answers "
            "the question. Write as if this is an excerpt from a technical "
            "document or manual. State facts only, no hedging."
        ),
        "code": (
            "Write a short code example with brief explanation that answers "
            "the question. Write as if this is from documentation."
        ),
        "medical": (
            "Write a short clinical passage that addresses the question. "
            "Write as if from a medical reference. Include relevant "
            "terminology and standard treatments."
        ),
    }

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        domain: str = "general",
        num_hypotheses: int = 1,   # generate N hypothetical docs
        temperature: float = 0.0,
    ):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(embedding_model)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.llm = OpenAI()
        self.llm_model = llm_model
        self.domain = domain
        self.num_hypotheses = num_hypotheses
        self.temperature = temperature

        self.texts: list[str] = []
        self.index: faiss.IndexFlatIP | None = None
        self._cache: dict[str, list[str]] = {}

    def build_index(self, texts: list[str], batch_size: int = 64):
        """Build FAISS index from document texts."""
        self.texts = texts
        embeddings = self.encoder.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        logger.info(f"Index built with {len(texts)} documents")

    def search(self, query: str, k: int = 5) -> list[HyDEResult]:
        """
        Search using HyDE. Falls back to standard search on failure.
        """
        if self.index is None:
            raise RuntimeError("Call build_index() first")

        try:
            hypothetical_docs = self._generate_hypothetical(query)
            hyde_embedding = self._embed_hypothetical(hypothetical_docs)
            used_hyde = True
            logger.info(f"HyDE generated {len(hypothetical_docs)} hypothetical docs")
        except Exception as e:
            logger.warning(f"HyDE generation failed, falling back: {e}")
            hyde_embedding = self.encoder.encode(
                query, normalize_embeddings=True
            )
            hyde_embedding = np.array([hyde_embedding], dtype=np.float32)
            used_hyde = False

        distances, indices = self.index.search(hyde_embedding, k)

        return [
            HyDEResult(
                index=int(idx),
                score=float(dist),
                text=self.texts[int(idx)],
                used_hyde=used_hyde,
            )
            for idx, dist in zip(indices[0], distances[0])
            if idx != -1
        ]

    def _generate_hypothetical(self, query: str) -> list[str]:
        """Generate hypothetical document(s) for the query."""
        cache_key = hashlib.sha256(query.encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        system_prompt = self.SYSTEM_PROMPTS.get(
            self.domain, self.SYSTEM_PROMPTS["general"]
        )

        hypotheticals = []
        for i in range(self.num_hypotheses):
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature + (i * 0.2),  # diversity for multi
                max_tokens=250,
            )
            hypotheticals.append(response.choices[0].message.content)

        self._cache[cache_key] = hypotheticals
        return hypotheticals

    def _embed_hypothetical(self, docs: list[str]) -> np.ndarray:
        """
        Embed hypothetical documents and average.
        If multiple hypotheses, the average embedding gives a
        more robust search vector.
        """
        embeddings = self.encoder.encode(docs, normalize_embeddings=True)
        avg_embedding = np.mean(embeddings, axis=0)
        # Re-normalize after averaging
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        return np.array([avg_embedding], dtype=np.float32)


# ─── Usage ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    retriever = HyDERetriever(
        domain="general",
        num_hypotheses=1,
    )

    documents = [
        "HNSW constructs a multi-layer proximity graph for approximate nearest neighbor search",
        "The algorithm starts at the top layer with few nodes and descends to denser layers",
        "BM25 is a probabilistic ranking function based on term frequency",
        "Product quantization compresses vectors into compact codes for efficient search",
        "Graph-based indexes like HNSW achieve O(log N) query time complexity",
        "The Faiss library provides efficient implementations of various ANN algorithms",
        "Dense retrieval encodes queries and documents into a shared vector space",
        "Inverted file indexes partition the vector space into Voronoi cells",
    ]

    retriever.build_index(documents)

    # Compare standard vs HyDE
    query = "How does HNSW search work internally?"

    results = retriever.search(query, k=3)
    print(f"\nQuery: '{query}'\n")
    for r in results:
        print(f"  [{r.score:.4f}] (HyDE={r.used_hyde}) {r.text}")
```

---

## When HyDE Helps vs Hurts

```
HyDE HELPS ✅:
  • Questions about technical concepts ("How does X work?")
  • Factual queries with clear expected answers
  • Domain-specific questions where vocabulary matters
  • When query-document asymmetry is high

HyDE HURTS ❌:
  • Keyword/exact-match queries ("error ERR_0x4A2F")
  • Questions about very recent events (LLM doesn't know)
  • Highly ambiguous queries (LLM may pick wrong interpretation)
  • Simple lookups that don't need semantic matching
  • Domains where the LLM hallucinates confidently

                ┌──────────────────────────────────┐
                │  HyDE adds ONE LLM call latency  │
                │  (~200-500ms for gpt-4o-mini)     │
                │  Consider if that budget exists   │
                └──────────────────────────────────┘
```

---

## Pitfalls & Common Mistakes

| Mistake                           | Impact                                | Fix                                                  |
| --------------------------------- | ------------------------------------- | ---------------------------------------------------- |
| **Treating HyDE output as truth** | Users see halllucinated "answers"     | HyDE is for retrieval only — show real docs to users |
| **Using HyDE for all queries**    | Unnecessary latency for exact matches | Route: keyword queries → BM25, semantic → HyDE       |
| **Long hypothetical docs**        | Embedding gets diluted                | Cap at 3-5 sentences / 250 tokens                    |
| **No fallback**                   | Pipeline breaks on LLM errors         | Always fall back to standard search                  |
| **Ignoring LLM cost**             | Every query costs tokens              | Cache hypotheticals for repeated/similar queries     |

---

## Key Takeaways

1. **HyDE solves the query-document asymmetry problem** — hypothetical answers are closer to real answers in embedding space.
2. **The hypothetical doc doesn't need to be factually correct** — it just needs to be in the right neighborhood.
3. **Adds latency** (one LLM call) — use selectively, not for every query.
4. **Multi-hypothesis HyDE** (averaging N hypothetical embeddings) is more robust but costs more.
5. **Never show the hypothetical doc to the user** — it's a retrieval trick, not an answer.
