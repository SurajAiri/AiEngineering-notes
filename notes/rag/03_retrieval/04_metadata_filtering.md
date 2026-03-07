# Metadata Filtering

## Why It Matters

Vector search returns documents that are **semantically similar**, but similarity alone isn't enough. You often need results from a **specific time range, source, category, or access level**. Metadata filtering narrows the search space before or during vector search — it's how you answer "What does the **2024 Q3 earnings report** say about revenue?" instead of getting results from 2019.

---

## Core Concept

```
Without filtering:
  Query: "revenue growth" → Searches ALL 500K documents → Gets results from 2019-2024

With filtering:
  Query: "revenue growth"
  Filter: year=2024 AND source="earnings_report" AND quarter="Q3"
  → Searches only ~50 matching documents → Precise results ✅

┌─────────────────────────────────────────────────┐
│      TWO STRATEGIES FOR METADATA FILTERING      │
├─────────────────────────────────────────────────┤
│                                                  │
│  PRE-FILTERING    Filter first, then search      │
│  ┌──────┐    ┌──────────┐    ┌────────┐         │
│  │Filter│ →  │Candidates│ →  │Search  │         │
│  └──────┘    └──────────┘    └────────┘         │
│  ✅ Fast if filter is selective                  │
│  ❌ May filter out relevant docs                 │
│                                                  │
│  POST-FILTERING   Search first, then filter      │
│  ┌──────┐    ┌────────┐    ┌──────┐             │
│  │Search│ →  │Results │ →  │Filter│             │
│  └──────┘    └────────┘    └──────┘             │
│  ✅ Doesn't miss anything in search              │
│  ❌ May return fewer than k results              │
│                                                  │
│  IN-SEARCH FILTERING (best — most vector DBs)   │
│  ┌──────────────────────────┐                    │
│  │ Filter + Search together │                    │
│  └──────────────────────────┘                    │
│  ✅ Best of both worlds                          │
│  ✅ What Pinecone/Weaviate/Qdrant do by default  │
└─────────────────────────────────────────────────┘
```

---

## What Metadata to Store

```
Useful metadata fields for RAG chunks:

IDENTITY          SOURCE             TEMPORAL
─────────         ──────             ────────
doc_id            source_url         created_at
chunk_id          file_path          updated_at
doc_type          author             version
                  department         effective_date

STRUCTURAL        ACCESS             SEMANTIC
──────────        ──────             ────────
section_title     access_level       language
page_number       team               topic
heading_path      visibility         confidence_score
chunk_index       sensitivity        entity_tags
```

---

## Simple Code — Filtering with FAISS

```python
"""
Metadata filtering with FAISS.
FAISS doesn't support filtering natively, so we implement
post-filtering and pre-filtering manually.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def search_with_metadata_filter(
    query: str,
    documents: list[dict],
    metadata_filter: dict,
    model: SentenceTransformer,
    k: int = 3,
) -> list[dict]:
    """
    Pre-filter documents by metadata, then do vector search
    on the filtered subset.
    """
    # Step 1: Filter by metadata
    filtered = []
    for doc in documents:
        match = all(
            doc["metadata"].get(key) == value
            for key, value in metadata_filter.items()
        )
        if match:
            filtered.append(doc)

    if not filtered:
        return []

    # Step 2: Vector search on filtered subset
    texts = [d["text"] for d in filtered]
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    query_vec = model.encode(query, normalize_embeddings=True)
    query_vec = np.array([query_vec], dtype=np.float32)

    distances, indices = index.search(query_vec, min(k, len(filtered)))

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx != -1:
            doc = filtered[int(idx)]
            doc["score"] = float(dist)
            results.append(doc)

    return results


# Example
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    {"text": "Q3 2024 revenue grew 15% YoY", "metadata": {"year": 2024, "quarter": "Q3", "type": "earnings"}},
    {"text": "Q3 2023 revenue declined 2%", "metadata": {"year": 2023, "quarter": "Q3", "type": "earnings"}},
    {"text": "New product launched in Q3 2024", "metadata": {"year": 2024, "quarter": "Q3", "type": "press_release"}},
    {"text": "Annual forecast predicts 20% growth", "metadata": {"year": 2024, "quarter": "Q4", "type": "forecast"}},
]

results = search_with_metadata_filter(
    query="revenue performance",
    documents=documents,
    metadata_filter={"year": 2024, "quarter": "Q3"},
    model=model,
)

for r in results:
    print(f"[{r['score']:.4f}] {r['text']}")
```

---

## Production Code — Metadata Filtering with Qdrant

```python
"""
Production metadata filtering with Qdrant.
Qdrant supports rich filtering with AND/OR/NOT, range queries,
and geo queries natively within vector search.

Requirements: pip install qdrant-client sentence-transformers
"""

import logging
from dataclasses import dataclass
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
    FilterSelector,
)
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class FilteredResult:
    doc_id: str
    text: str
    score: float
    metadata: dict


class MetadataFilteredSearch:
    """
    Vector search with metadata filtering using Qdrant.
    Supports complex filter expressions.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        model_name: str = "all-MiniLM-L6-v2",
        qdrant_url: str = "http://localhost:6333",
    ):
        self.collection_name = collection_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.client = QdrantClient(url=qdrant_url)

    def create_collection(self):
        """Create Qdrant collection with vector config."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info(f"Collection '{self.collection_name}' created")

    def ingest(self, documents: list[dict]):
        """
        Ingest documents with metadata.

        Each document: {"id": str, "text": str, "metadata": dict}
        """
        points = []
        texts = [doc["text"] for doc in documents]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        for doc, embedding in zip(documents, embeddings):
            points.append(PointStruct(
                id=hash(doc["id"]) % (2**63),  # Qdrant needs int IDs
                vector=embedding.tolist(),
                payload={
                    "doc_id": doc["id"],
                    "text": doc["text"],
                    **doc["metadata"],
                },
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Ingested {len(points)} documents")

    def search(
        self,
        query: str,
        k: int = 5,
        filters: dict | None = None,
    ) -> list[FilteredResult]:
        """
        Search with optional metadata filters.

        Filter examples:
          {"year": 2024}                    → exact match
          {"year__gte": 2023}               → range: year >= 2023
          {"year__lte": 2024}               → range: year <= 2024
          {"type": "earnings"}              → exact match
          {"year": 2024, "type": "earnings"} → AND of both
        """
        query_vec = self.model.encode(query, normalize_embeddings=True).tolist()

        # Build Qdrant filter
        qdrant_filter = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vec,
            query_filter=qdrant_filter,
            limit=k,
        )

        return [
            FilteredResult(
                doc_id=r.payload.get("doc_id", ""),
                text=r.payload.get("text", ""),
                score=r.score,
                metadata={
                    k: v for k, v in r.payload.items()
                    if k not in ("doc_id", "text")
                },
            )
            for r in results
        ]

    def _build_filter(self, filters: dict) -> Filter:
        """Convert simple filter dict to Qdrant Filter object."""
        conditions = []
        for key, value in filters.items():
            if key.endswith("__gte"):
                field = key[:-5]
                conditions.append(FieldCondition(
                    key=field,
                    range=Range(gte=value),
                ))
            elif key.endswith("__lte"):
                field = key[:-5]
                conditions.append(FieldCondition(
                    key=field,
                    range=Range(lte=value),
                ))
            elif key.endswith("__gt"):
                field = key[:-4]
                conditions.append(FieldCondition(
                    key=field,
                    range=Range(gt=value),
                ))
            elif key.endswith("__lt"):
                field = key[:-4]
                conditions.append(FieldCondition(
                    key=field,
                    range=Range(lt=value),
                ))
            else:
                conditions.append(FieldCondition(
                    key=key,
                    match=MatchValue(value=value),
                ))

        return Filter(must=conditions)


# ─── Usage ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    search = MetadataFilteredSearch(collection_name="earnings")
    search.create_collection()

    documents = [
        {"id": "e-2024-q3-1", "text": "Revenue grew 15% year-over-year to $2.1B",
         "metadata": {"year": 2024, "quarter": "Q3", "type": "earnings", "department": "finance"}},
        {"id": "e-2024-q2-1", "text": "Revenue was flat at $1.8B compared to prior year",
         "metadata": {"year": 2024, "quarter": "Q2", "type": "earnings", "department": "finance"}},
        {"id": "e-2023-q3-1", "text": "Revenue declined 2% to $1.82B",
         "metadata": {"year": 2023, "quarter": "Q3", "type": "earnings", "department": "finance"}},
        {"id": "p-2024-q3-1", "text": "Launched new AI-powered analytics platform",
         "metadata": {"year": 2024, "quarter": "Q3", "type": "press_release", "department": "product"}},
    ]

    search.ingest(documents)

    # Search with filters
    print("=== Q3 2024 Earnings Only ===")
    results = search.search(
        query="revenue performance",
        k=3,
        filters={"year": 2024, "quarter": "Q3", "type": "earnings"},
    )
    for r in results:
        print(f"  [{r.score:.4f}] {r.text}")

    print("\n=== All 2024 Results ===")
    results = search.search(
        query="revenue performance",
        k=5,
        filters={"year__gte": 2024},
    )
    for r in results:
        print(f"  [{r.score:.4f}] {r.text} (Q{r.metadata.get('quarter', '?')})")
```

---

## Filter Syntax Across Vector Databases

```
┌─────────────────────────────────────────────────────────────────┐
│  DB         │ Filter Example (year=2024 AND type="earnings")    │
├─────────────┼───────────────────────────────────────────────────┤
│ Pinecone    │ {"year": {"$eq": 2024}, "type": {"$eq":"earnings"}} │
│ Qdrant      │ Filter(must=[FieldCondition(key="year",           │
│             │   match=MatchValue(value=2024)), ...])            │
│ Weaviate    │ where: {operator: And, operands: [{path:["year"], │
│             │   operator: Equal, valueInt: 2024}, ...]}         │
│ pgvector    │ WHERE metadata->>'year' = '2024'                  │
│             │   AND metadata->>'type' = 'earnings'              │
│ Milvus      │ 'year == 2024 and type == "earnings"'             │
└─────────────┴───────────────────────────────────────────────────┘
```

---

## Pitfalls & Common Mistakes

| Mistake                          | Impact                              | Fix                                              |
| -------------------------------- | ----------------------------------- | ------------------------------------------------ |
| **Too many filter fields**       | Slow queries, complex maintenance   | Index only fields you actually filter on         |
| **Overly specific filters**      | Zero results                        | Add fallback: try without least important filter |
| **Not indexing filter fields**   | Full scan on every query            | Create indexes on commonly filtered fields       |
| **Inconsistent metadata**        | "Q3" vs "q3" vs "3" — filter misses | Normalize and validate metadata at ingest time   |
| **No filter for access control** | Users see docs they shouldn't       | Always filter by user's access level             |

---

## Key Takeaways

1. **Pre-filtering is essential** — pure vector search returns "similar" but not necessarily "relevant" results.
2. **Store temporal, structural, and access metadata** on every chunk.
3. **Normalize metadata at ingest** — inconsistent values break filters silently.
4. **Use in-search filtering** (not post-filtering) for correctness and performance.
5. **Build fallback logic** — progressively relax filters if results are too few.
