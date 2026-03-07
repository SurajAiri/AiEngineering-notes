# Metadata Design for RAG

## Why This Matters

Metadata is the difference between a RAG system that "kind of works" and one that gives precise, trustworthy answers. Without good metadata, you can only search by semantic similarity. With metadata, you can **filter**, **scope**, **attribute**, and **rank** your results.

---

## What Is Metadata in RAG?

```
┌─────────────────────────────────────────────────┐
│                    CHUNK                         │
│                                                  │
│ content: "Returns accepted within 14 days..."    │
│                                                  │
│ metadata: {                                      │
│   "source":      "return_policy_v2.pdf",        │
│   "doc_id":      "return_policy",               │
│   "doc_type":    "policy",                      │
│   "department":  "customer_service",            │
│   "version":     2,                             │
│   "page":        3,                             │
│   "section":     "Returns > Timeframes",        │
│   "effective":   "2024-06-01",                  │
│   "is_current":  true,                          │
│   "chunk_index": 5,                             │
│   "total_chunks": 12                            │
│ }                                                │
└─────────────────────────────────────────────────┘
```

**Metadata enables:**

- **Filtering**: "Only search in policy documents"
- **Scoping**: "Only search documents from the engineering department"
- **Attribution**: "This answer came from return_policy_v2.pdf, page 3"
- **Ranking**: "Prefer current versions over superseded ones"
- **Debugging**: "Why did this chunk get retrieved?"

---

## Metadata Categories

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│    SOURCE        │  │    STRUCTURAL    │  │    TEMPORAL       │
│                  │  │                  │  │                  │
│ • file_name      │  │ • section_path   │  │ • created_date   │
│ • file_type      │  │ • heading_level  │  │ • modified_date  │
│ • url            │  │ • page_number    │  │ • effective_date │
│ • doc_id         │  │ • chunk_index    │  │ • expiry_date    │
│ • author         │  │ • parent_id      │  │ • ingested_at    │
│ • system         │  │ • total_chunks   │  │ • version        │
└──────────────────┘  └──────────────────┘  └──────────────────┘

┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│    SEMANTIC      │  │    ACCESS        │  │    QUALITY       │
│                  │  │                  │  │                  │
│ • doc_type       │  │ • visibility     │  │ • confidence     │
│ • category       │  │ • department     │  │ • is_verified    │
│ • tags           │  │ • access_level   │  │ • noise_ratio    │
│ • topic          │  │ • tenant_id      │  │ • extraction     │
│ • language       │  │ • team           │  │   _method        │
│ • entities       │  │ • region         │  │ • is_current     │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## Simple Code — Basic Metadata Tagging

```python
"""
Simple metadata assignment during document ingestion.
"""

from pathlib import Path
from datetime import datetime


def create_chunk_metadata(
    content: str,
    source_file: str,
    chunk_index: int,
    total_chunks: int,
    section_heading: str = "",
) -> dict:
    """Create basic metadata for a chunk."""
    path = Path(source_file)

    return {
        # Source metadata
        "source": source_file,
        "file_name": path.name,
        "file_type": path.suffix.lstrip('.'),

        # Structural metadata
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "section": section_heading,
        "char_count": len(content),
        "word_count": len(content.split()),

        # Temporal metadata
        "ingested_at": datetime.utcnow().isoformat(),
    }


# Usage
chunks = [
    "Vector databases store high-dimensional embeddings...",
    "HNSW is the most common indexing algorithm...",
    "Quantization reduces memory usage by compressing vectors...",
]

for i, chunk_text in enumerate(chunks):
    meta = create_chunk_metadata(
        content=chunk_text,
        source_file="docs/vector_databases_guide.pdf",
        chunk_index=i,
        total_chunks=len(chunks),
        section_heading="Chapter 2: Indexing",
    )
    print(f"Chunk {i}: {meta}")
```

---

## Production Code — Metadata Schema & Enrichment

```python
"""
Production metadata system with schema validation, enrichment, and
multi-tenant support.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DocType(Enum):
    POLICY = "policy"
    API_DOCS = "api_docs"
    TUTORIAL = "tutorial"
    FAQ = "faq"
    KNOWLEDGE_BASE = "knowledge_base"
    INTERNAL_WIKI = "internal_wiki"
    LEGAL = "legal"
    SUPPORT_TICKET = "support_ticket"


# ─── Schema Definition ───

REQUIRED_FIELDS = {"source", "doc_id", "chunk_index", "ingested_at"}
FILTERABLE_FIELDS = {"doc_type", "department", "is_current", "language", "visibility"}
INDEXED_FIELDS = REQUIRED_FIELDS | FILTERABLE_FIELDS


@dataclass
class ChunkMetadata:
    """Validated metadata for a single chunk."""
    # Required
    source: str
    doc_id: str
    chunk_index: int
    ingested_at: str

    # Source info
    file_name: str = ""
    file_type: str = ""
    url: str = ""
    author: str = ""

    # Structural
    section_path: str = ""           # e.g., "Chapter 2 > Indexing > HNSW"
    page_number: int | None = None
    heading: str = ""
    total_chunks: int = 0
    parent_chunk_id: str = ""        # for hierarchical chunking

    # Temporal
    created_date: str = ""
    modified_date: str = ""
    version: int = 1
    is_current: bool = True

    # Semantic
    doc_type: str = ""
    category: str = ""
    tags: list[str] = field(default_factory=list)
    language: str = "en"

    # Access
    visibility: str = "internal"     # internal, public, restricted
    department: str = ""
    tenant_id: str = ""              # for multi-tenant systems

    # Quality
    extraction_method: str = ""      # "pdf_parser", "ocr", "html_scraper"
    confidence: float = 1.0          # extraction confidence

    # Content-derived
    char_count: int = 0
    word_count: int = 0
    content_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for vector store insertion."""
        d = {}
        for k, v in self.__dict__.items():
            if v is not None and v != "" and v != [] and v != 0:
                d[k] = v
        return d

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors = []
        if not self.source:
            errors.append("source is required")
        if not self.doc_id:
            errors.append("doc_id is required")
        if self.chunk_index < 0:
            errors.append("chunk_index must be >= 0")
        if not self.ingested_at:
            errors.append("ingested_at is required")
        if self.confidence < 0 or self.confidence > 1:
            errors.append("confidence must be between 0 and 1")
        return errors


class MetadataEnricher:
    """
    Enriches raw metadata with derived fields.

    Use this during ingestion to automatically populate metadata
    from document content and source information.
    """

    # Map file extensions to doc types (basic heuristic)
    TYPE_HINTS = {
        ".md": DocType.KNOWLEDGE_BASE,
        ".pdf": DocType.KNOWLEDGE_BASE,
        ".html": DocType.API_DOCS,
    }

    def __init__(self, default_department: str = "", default_tenant: str = ""):
        self.default_department = default_department
        self.default_tenant = default_tenant

    def enrich(
        self,
        content: str,
        source: str,
        chunk_index: int,
        total_chunks: int,
        overrides: dict[str, Any] | None = None,
    ) -> ChunkMetadata:
        """Create enriched metadata from content and source info."""
        path = Path(source)
        now = datetime.now(timezone.utc).isoformat()

        meta = ChunkMetadata(
            # Required
            source=source,
            doc_id=self._stable_doc_id(source),
            chunk_index=chunk_index,
            ingested_at=now,
            # Source
            file_name=path.name,
            file_type=path.suffix.lstrip('.'),
            # Structural
            total_chunks=total_chunks,
            heading=self._extract_heading(content),
            # Semantic
            doc_type=self._guess_doc_type(path),
            language=self._detect_language_hint(content),
            # Access
            department=self.default_department,
            tenant_id=self.default_tenant,
            # Quality
            extraction_method=self._guess_extraction(path),
            # Content-derived
            char_count=len(content),
            word_count=len(content.split()),
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
        )

        # Apply any manual overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(meta, key):
                    setattr(meta, key, value)

        # Validate
        errors = meta.validate()
        if errors:
            logger.warning(f"Metadata validation errors for {source}: {errors}")

        return meta

    def _stable_doc_id(self, source: str) -> str:
        """Generate a stable document ID from the source path."""
        # Remove version suffixes and normalize
        name = Path(source).stem
        name = re.sub(r'_v\d+$', '', name)
        name = re.sub(r'\s+', '_', name.lower())
        return name

    def _guess_doc_type(self, path: Path) -> str:
        doc_type = self.TYPE_HINTS.get(path.suffix, DocType.KNOWLEDGE_BASE)
        return doc_type.value

    def _extract_heading(self, content: str) -> str:
        """Extract the first heading from content."""
        match = re.match(r'^#+\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        # First non-empty line as fallback
        for line in content.split('\n'):
            line = line.strip()
            if line:
                return line[:100]
        return ""

    def _detect_language_hint(self, content: str) -> str:
        # Simple heuristic — production would use langdetect
        if re.search(r'[àáâãäçèéêëìíîïòóôõöùúûü]', content):
            return "pt"  # or other latin languages
        return "en"

    def _guess_extraction(self, path: Path) -> str:
        mapping = {".pdf": "pdf_parser", ".html": "html_scraper", ".md": "markdown"}
        return mapping.get(path.suffix, "plaintext")


# ─── Usage ───

if __name__ == "__main__":
    enricher = MetadataEnricher(
        default_department="engineering",
        default_tenant="acme_corp",
    )

    chunk_content = """## HNSW Algorithm

    HNSW (Hierarchical Navigable Small World) is a graph-based
    approximate nearest neighbor algorithm. It builds a multi-layer
    graph where each layer contains a subset of the points."""

    meta = enricher.enrich(
        content=chunk_content,
        source="docs/vector_databases_guide.pdf",
        chunk_index=5,
        total_chunks=20,
        overrides={
            "category": "algorithms",
            "tags": ["hnsw", "ann", "indexing"],
            "version": 2,
            "section_path": "Chapter 3 > Indexing > HNSW",
        },
    )

    print("Metadata:")
    for k, v in meta.to_dict().items():
        print(f"  {k}: {v}")
```

---

## Metadata for Vector Store Filtering

Different vector stores support metadata filtering differently:

```python
"""
Examples of metadata filtering across different vector stores.
These show how metadata enables precise retrieval.
"""

# ─── Pinecone ───
pinecone_filter = {
    "doc_type": {"$eq": "policy"},
    "department": {"$eq": "engineering"},
    "is_current": {"$eq": True},
}

# ─── Qdrant ───
from qdrant_client.models import Filter, FieldCondition, MatchValue

qdrant_filter = Filter(
    must=[
        FieldCondition(key="doc_type", match=MatchValue(value="policy")),
        FieldCondition(key="is_current", match=MatchValue(value=True)),
    ]
)

# ─── Weaviate ───
weaviate_where = {
    "operator": "And",
    "operands": [
        {"path": ["doc_type"], "operator": "Equal", "valueText": "policy"},
        {"path": ["is_current"], "operator": "Equal", "valueBoolean": True},
    ]
}

# ─── pgvector (SQL) ───
pgvector_query = """
    SELECT content, embedding <=> %s AS distance
    FROM chunks
    WHERE metadata->>'doc_type' = 'policy'
      AND (metadata->>'is_current')::boolean = true
    ORDER BY distance
    LIMIT 10
"""
```

---

## Metadata Design Patterns

### Pattern 1: Hierarchical Section Paths

```
Instead of:                    Use:
  section: "HNSW"              section_path: "Ch3 > Indexing > HNSW"

Why: Enables filtering at any level of the hierarchy
  "Show me everything in Chapter 3"
  "Show me all Indexing content"
  "Show me specifically HNSW"
```

### Pattern 2: Multi-Tenant Isolation

```python
# ALWAYS include tenant_id in metadata for multi-tenant systems
# Filter on tenant_id FIRST to ensure data isolation

def search(query: str, tenant_id: str, filters: dict = None):
    """Search with mandatory tenant isolation."""
    combined_filter = {"tenant_id": {"$eq": tenant_id}}
    if filters:
        combined_filter.update(filters)
    return vector_store.search(query, filter=combined_filter)
```

### Pattern 3: Chunk Relationship Metadata

```python
# For hierarchical chunking, maintain parent-child relationships
chunk_meta = {
    "chunk_id": "doc_001_chunk_5",
    "parent_chunk_id": "doc_001_chunk_2",  # section-level chunk
    "chunk_level": "paragraph",             # section, paragraph, sentence
    "sibling_chunks": ["doc_001_chunk_4", "doc_001_chunk_6"],
}
```

---

## Pitfalls & Common Mistakes

| Mistake                            | Impact                                               | Fix                                                           |
| ---------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------- |
| **No metadata at all**             | Can't filter, can't attribute, can't debug           | Define a minimum schema: source, doc_id, chunk_index          |
| **Unstable doc_ids**               | Same document gets different IDs across reingestions | Derive doc_id from content path, not random UUIDs             |
| **Too many metadata fields**       | Index bloat, slower queries                          | Only index fields you'll filter on; store the rest as payload |
| **No data type consistency**       | Filters break (comparing string "2024" vs int 2024)  | Enforce types in your metadata schema                         |
| **Missing tenant isolation**       | Data leaks between tenants                           | Always filter on tenant_id; make it mandatory                 |
| **Not indexing filterable fields** | Full metadata scan on every query                    | Configure vector store to index your filter fields            |

---

## What to Index vs Store

```
┌─────────────┬────────────────────────────────────────┐
│   INDEXED   │  Used in WHERE/filter clauses          │
│             │  doc_type, department, is_current,     │
│   (faster   │  tenant_id, language, visibility       │
│    filters)  │                                        │
├─────────────┼────────────────────────────────────────┤
│   STORED    │  Returned with results but not         │
│   ONLY      │  used in filters                       │
│             │  section_path, heading, page_number,   │
│   (payload) │  tags, content_hash, author            │
└─────────────┴────────────────────────────────────────┘
```

**Rule of thumb**: Index fields you filter on frequently. Store everything else as payload.

---

## Key Takeaways

1. **Define a metadata schema** before you start ingesting — retrofitting is painful.
2. **source + doc_id + chunk_index** is the minimum viable metadata.
3. **Use stable doc_ids** derived from content paths, not random UUIDs.
4. **Metadata enables filtering** — without it, you're limited to pure semantic search.
5. **Different metadata for different roles**: source attribution, access control, temporal queries, and debugging all need different fields.
6. **Test your filters** — verify they work with your vector store before ingesting millions of chunks.
