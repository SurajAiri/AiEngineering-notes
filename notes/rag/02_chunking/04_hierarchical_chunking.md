# Hierarchical Chunking

## Why This Matters

Real documents have structure — chapters, sections, subsections, paragraphs. Hierarchical chunking preserves this structure by creating chunks at multiple levels of granularity and maintaining parent-child relationships. This lets you retrieve at the right level of detail: a specific paragraph when precision matters, a full section when context is needed.

---

## How It Works

```
Document
├── Chapter 1: Introduction                    ← Level 1 (broadest)
│   ├── 1.1 Background                         ← Level 2
│   │   ├── Paragraph about history             ← Level 3 (most specific)
│   │   └── Paragraph about motivation
│   └── 1.2 Scope
│       ├── Paragraph about what's covered
│       └── Paragraph about limitations
└── Chapter 2: Architecture
    ├── 2.1 Components
    │   ├── Paragraph about service A
    │   └── Paragraph about service B
    └── 2.2 Data Flow
        ├── Paragraph about ingestion
        └── Paragraph about processing


Level 1 chunks: Full chapters (high context, low precision)
Level 2 chunks: Sections (balanced)
Level 3 chunks: Paragraphs (low context, high precision)
```

### Two Retrieval Strategies

```
Strategy 1: RETRIEVE LEAF, EXPAND TO PARENT
┌─────────────────────────────────┐
│ Query: "What service handles auth?" │
│                                 │
│ 1. Search at paragraph level    │
│ 2. Match: "Service A handles   │
│    authentication using JWT"    │
│ 3. Expand: include parent       │
│    section for full context     │
└─────────────────────────────────┘

Strategy 2: RETRIEVE AT MULTIPLE LEVELS
┌─────────────────────────────────┐
│ Query: "Explain the architecture"│
│                                 │
│ 1. Search at BOTH section and   │
│    paragraph levels             │
│ 2. Section match gives overview │
│ 3. Paragraph match gives detail │
│ 4. Combine for complete answer  │
└─────────────────────────────────┘
```

---

## Simple Code — Understanding the Concept

```python
"""
Minimal hierarchical chunking:
parse markdown structure into a tree of chunks.
"""

import re
from dataclasses import dataclass, field


@dataclass
class HierarchicalChunk:
    text: str
    level: int           # 1=chapter, 2=section, 3=paragraph
    heading: str
    path: str            # "Ch1 > Section 1.1 > ..."
    children: list = field(default_factory=list)
    parent_id: str = ""
    chunk_id: str = ""


def hierarchical_chunk_markdown(text: str) -> list[HierarchicalChunk]:
    """
    Parse markdown into hierarchical chunks.
    Each heading level creates a new level in the hierarchy.
    """
    lines = text.split('\n')
    chunks = []
    heading_stack = []  # [(level, heading_text, chunk)]
    current_content = []

    for line in lines:
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if heading_match:
            # Save accumulated content to current chunk
            if current_content and heading_stack:
                heading_stack[-1][2].text = '\n'.join(current_content).strip()
                current_content = []

            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()

            # Pop stack to find parent
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            # Build path
            path = ' > '.join(h[1] for h in heading_stack)
            if path:
                path += ' > ' + title
            else:
                path = title

            chunk = HierarchicalChunk(
                text="",
                level=level,
                heading=title,
                path=path,
                chunk_id=f"chunk_{len(chunks)}",
                parent_id=heading_stack[-1][2].chunk_id if heading_stack else "",
            )

            # Add as child of parent
            if heading_stack:
                heading_stack[-1][2].children.append(chunk.chunk_id)

            chunks.append(chunk)
            heading_stack.append((level, title, chunk))
        else:
            current_content.append(line)

    # Save final content
    if current_content and heading_stack:
        heading_stack[-1][2].text = '\n'.join(current_content).strip()

    return chunks


# Example
doc = """# Vector Databases

## What Are They

Vector databases store high-dimensional embeddings.
They enable fast similarity search over millions of vectors.

## Indexing Algorithms

### HNSW

HNSW builds a multi-layer graph for approximate nearest neighbor search.
It offers a good balance of speed and recall.

### IVF

IVF partitions the vector space into clusters using k-means.
Queries are compared only against nearby clusters.

## Use Cases

RAG systems use vector databases to store document embeddings.
Recommendation engines use them for user-item matching.
"""

chunks = hierarchical_chunk_markdown(doc)
for chunk in chunks:
    indent = "  " * (chunk.level - 1)
    print(f"{indent}[L{chunk.level}] {chunk.heading}")
    if chunk.text:
        print(f"{indent}     Content: {chunk.text[:80]}...")
    if chunk.parent_id:
        print(f"{indent}     Parent: {chunk.parent_id}")
    print()
```

---

## Production Code — Full Hierarchical Chunking System

```python
"""
Production hierarchical chunking with:
- Multi-level chunk storage
- Parent context expansion
- Token-aware size limits
- Retrieval at any level

Requirements: pip install tiktoken
"""

import re
import uuid
import logging
import tiktoken
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class HChunk:
    """A chunk in the hierarchy."""
    chunk_id: str
    text: str
    level: int
    heading: str
    path: str
    parent_id: str | None
    children_ids: list[str]
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # Full text including all children (for section-level retrieval)
    full_text: str = ""
    full_token_count: int = 0


class HierarchicalChunker:
    """
    Creates chunks at multiple levels of document hierarchy.

    Supports two usage modes:
    1. **Leaf retrieval + parent expansion**: Search at paragraph level,
       expand to section for context
    2. **Multi-level retrieval**: Search at all levels, combine results
    """

    def __init__(
        self,
        max_leaf_tokens: int = 256,
        max_section_tokens: int = 1024,
        encoding: str = "cl100k_base",
    ):
        self.max_leaf_tokens = max_leaf_tokens
        self.max_section_tokens = max_section_tokens
        self.tokenizer = tiktoken.get_encoding(encoding)
        self._chunks: dict[str, HChunk] = {}

    def chunk(self, text: str, source: str = "") -> list[HChunk]:
        """Parse document into hierarchical chunks."""
        self._chunks.clear()

        # Parse structure
        sections = self._parse_sections(text)

        # Build chunk tree
        all_chunks = []
        for section in sections:
            chunk = self._process_section(section, parent_id=None)
            all_chunks.extend(self._flatten(chunk))

        # Calculate full_text for each chunk (includes children)
        for chunk in all_chunks:
            chunk.full_text = self._get_full_text(chunk)
            chunk.full_token_count = len(self.tokenizer.encode(chunk.full_text))
            chunk.metadata["source"] = source

        logger.info(
            f"Hierarchical chunking: {len(all_chunks)} chunks across "
            f"{len(set(c.level for c in all_chunks))} levels"
        )
        return all_chunks

    def get_parent(self, chunk: HChunk) -> HChunk | None:
        """Get the parent chunk."""
        if chunk.parent_id:
            return self._chunks.get(chunk.parent_id)
        return None

    def get_children(self, chunk: HChunk) -> list[HChunk]:
        """Get child chunks."""
        return [self._chunks[cid] for cid in chunk.children_ids if cid in self._chunks]

    def get_with_parent_context(self, chunk: HChunk) -> str:
        """
        Get chunk text with parent context prepended.
        Used in 'leaf retrieval + parent expansion' mode.
        """
        parts = []
        parent = self.get_parent(chunk)
        if parent:
            # Add parent heading for context
            parts.append(f"[Context: {parent.path}]")
            # Add sibling headings for orientation
            siblings = self.get_children(parent)
            sibling_headings = [s.heading for s in siblings]
            parts.append(f"[Subsections: {', '.join(sibling_headings)}]")
            parts.append("")

        parts.append(chunk.text)
        return '\n'.join(parts)

    def prepare_for_indexing(self, chunks: list[HChunk]) -> list[dict]:
        """
        Prepare chunks for vector store ingestion.
        Returns both leaf chunks and section summaries.
        """
        index_items = []

        for chunk in chunks:
            # Leaf chunks (paragraphs) — for precise retrieval
            if not chunk.children_ids:
                index_items.append({
                    "id": chunk.chunk_id,
                    "content": chunk.text,
                    "metadata": {
                        "type": "leaf",
                        "level": chunk.level,
                        "heading": chunk.heading,
                        "path": chunk.path,
                        "parent_id": chunk.parent_id,
                        "token_count": chunk.token_count,
                        **chunk.metadata,
                    }
                })

            # Section chunks (headings with child content) — for broad retrieval
            if chunk.children_ids and chunk.full_token_count <= self.max_section_tokens:
                index_items.append({
                    "id": f"{chunk.chunk_id}_section",
                    "content": chunk.full_text,
                    "metadata": {
                        "type": "section",
                        "level": chunk.level,
                        "heading": chunk.heading,
                        "path": chunk.path,
                        "children_count": len(chunk.children_ids),
                        "token_count": chunk.full_token_count,
                        **chunk.metadata,
                    }
                })

        return index_items

    def _parse_sections(self, text: str) -> list[dict]:
        """Parse markdown into a section tree."""
        lines = text.split('\n')
        sections = []
        current = {"level": 0, "heading": "(root)", "content": [], "children": []}
        stack = [current]

        for line in lines:
            match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if match:
                level = len(match.group(1))
                heading = match.group(2).strip()

                new_section = {"level": level, "heading": heading, "content": [], "children": []}

                # Find parent in stack
                while len(stack) > 1 and stack[-1]["level"] >= level:
                    stack.pop()

                stack[-1]["children"].append(new_section)
                stack.append(new_section)
            else:
                if stack:
                    stack[-1]["content"].append(line)

        return current["children"] if current["children"] else [current]

    def _process_section(self, section: dict, parent_id: str | None) -> HChunk:
        """Recursively process a section into chunks."""
        chunk_id = str(uuid.uuid4())[:8]
        content = '\n'.join(section["content"]).strip()
        token_count = len(self.tokenizer.encode(content)) if content else 0

        # Build path from parent
        parent = self._chunks.get(parent_id) if parent_id else None
        if parent:
            path = f"{parent.path} > {section['heading']}"
        else:
            path = section["heading"]

        chunk = HChunk(
            chunk_id=chunk_id,
            text=content,
            level=section["level"],
            heading=section["heading"],
            path=path,
            parent_id=parent_id,
            children_ids=[],
            token_count=token_count,
        )
        self._chunks[chunk_id] = chunk

        # Process children
        for child_section in section.get("children", []):
            child_chunk = self._process_section(child_section, parent_id=chunk_id)
            chunk.children_ids.append(child_chunk.chunk_id)

        # If content is too large and no children, split into sub-chunks
        if token_count > self.max_leaf_tokens and not chunk.children_ids:
            sub_chunks = self._split_large_content(content, chunk_id, path, section["level"] + 1)
            for sc in sub_chunks:
                chunk.children_ids.append(sc.chunk_id)
            chunk.text = ""  # Content moved to children

        return chunk

    def _split_large_content(
        self, content: str, parent_id: str, parent_path: str, level: int
    ) -> list[HChunk]:
        """Split oversized content into paragraph-level chunks."""
        paragraphs = re.split(r'\n\s*\n', content)
        sub_chunks = []

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            chunk_id = str(uuid.uuid4())[:8]
            chunk = HChunk(
                chunk_id=chunk_id,
                text=para,
                level=level,
                heading=f"Paragraph {i + 1}",
                path=f"{parent_path} > p{i + 1}",
                parent_id=parent_id,
                children_ids=[],
                token_count=len(self.tokenizer.encode(para)),
            )
            self._chunks[chunk_id] = chunk
            sub_chunks.append(chunk)

        return sub_chunks

    def _flatten(self, chunk: HChunk) -> list[HChunk]:
        """Flatten tree into list."""
        result = [chunk]
        for child_id in chunk.children_ids:
            child = self._chunks[child_id]
            result.extend(self._flatten(child))
        return result

    def _get_full_text(self, chunk: HChunk) -> str:
        """Get text including all descendants."""
        parts = []
        if chunk.heading != "(root)":
            parts.append(f"{'#' * chunk.level} {chunk.heading}")
        if chunk.text:
            parts.append(chunk.text)
        for child_id in chunk.children_ids:
            child = self._chunks[child_id]
            parts.append(self._get_full_text(child))
        return '\n\n'.join(parts)


# ─── Usage ───
if __name__ == "__main__":
    chunker = HierarchicalChunker(
        max_leaf_tokens=100,
        max_section_tokens=500,
    )

    document = """# System Architecture

## Authentication Service

The auth service handles user authentication using JWT tokens.
It validates credentials against the user database and issues
short-lived access tokens with refresh token rotation.

The service supports multiple authentication methods including
OAuth2, SAML, and API key authentication for service-to-service
communication.

## Data Pipeline

### Ingestion Layer

Raw data arrives from multiple sources: REST APIs, message
queues (Kafka), and file uploads (S3). Each source has a
dedicated adapter that normalizes the data format.

### Processing Layer

Data goes through validation, enrichment, and transformation
stages. Each stage is implemented as an independent worker
that can be scaled horizontally.

### Storage Layer

Processed data is stored in PostgreSQL for structured queries
and Elasticsearch for full-text search. A Redis cache sits in
front for hot data access.
"""

    chunks = chunker.chunk(document, source="architecture.md")

    print("=== Chunk Tree ===")
    for chunk in chunks:
        indent = "  " * chunk.level
        print(f"{indent}[L{chunk.level}] {chunk.heading} ({chunk.token_count} tokens)")
        if chunk.children_ids:
            print(f"{indent}  children: {len(chunk.children_ids)}")

    print("\n=== Index Items ===")
    items = chunker.prepare_for_indexing(chunks)
    for item in items:
        print(f"  [{item['metadata']['type']}] {item['metadata']['path']}")
        print(f"    Content: {item['content'][:80]}...")
        print()
```

---

## Pitfalls & Common Mistakes

| Mistake                              | Impact                                          | Fix                                           |
| ------------------------------------ | ----------------------------------------------- | --------------------------------------------- |
| **Flat chunking of structured docs** | Loses heading context                           | Use hierarchical chunking for structured docs |
| **Only indexing leaf chunks**        | Missing section-level queries ("overview of X") | Index at multiple levels                      |
| **No parent context**                | Leaf chunks lack context                        | Prepend parent heading/path in retrieval      |
| **Too many levels**                  | Excessive index size                            | Limit to 2-3 levels for practical systems     |
| **Ignoring unstructured sections**   | Large unheaded content blocks dropped           | Split oversized leaf content into paragraphs  |

---

## Trade-offs

```
More levels:                          Fewer levels:
├── More retrieval granularity       ├── Simpler implementation
├── Larger index (2-3x)             ├── Smaller index
├── More flexible queries           ├── Faster queries
├── Complex to implement            ├── Less precise for detail
└── Best for: structured docs       └── Best for: flat docs
```

---

## Key Takeaways

1. **Hierarchical chunking preserves document structure** — headings, sections, and their relationships.
2. **Index at multiple levels** — leaf for precision, section for context.
3. **Parent context expansion** is key — when you retrieve a paragraph, include its section heading.
4. **Only use for structured documents** — unstructured prose doesn't benefit.
5. **Limit to 2-3 practical levels** — chapter/section/paragraph is usually enough.
