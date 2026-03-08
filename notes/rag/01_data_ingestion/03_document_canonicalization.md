# Document Canonicalization

## Why This Matters

When the same information exists in different formats, structures, or naming conventions, retrieval becomes unpredictable. Canonicalization ensures that documents have a **consistent, normalized form** so that identical content always matches, sections can be reliably referenced, and chunks can be traced back to their origin.

---

## What Canonicalization Covers

```
┌─────────────────────────────────────────────────────────┐
│              DOCUMENT CANONICALIZATION                   │
│                                                         │
│  1. Normalized Formatting                               │
│     → Consistent headings, lists, whitespace            │
│                                                         │
│  2. Stable IDs                                          │
│     → Deterministic document IDs from content/path      │
│                                                         │
│  3. Section Path Normalization                          │
│     → "Chapter 3 > Indexing > HNSW" consistently        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 1. Normalized Formatting

### The Problem

```
Same content, three different formats:

PDF extracted:      "3.1   HNSW Algorithm\n\nHNSW (Hierarchical..."
HTML scraped:       "<h2>3.1 HNSW Algorithm</h2><p>HNSW (Hierarchical..."
Markdown source:    "## 3.1 HNSW Algorithm\n\nHNSW (Hierarchical..."

These will produce DIFFERENT embeddings even though the content is identical.
```

### Simple Code — Format Normalizer

```python
import re


def normalize_to_plaintext(text: str) -> str:
    """
    Normalize any text format to clean plaintext.
    Preserves semantic structure (paragraphs, lists) but
    removes format-specific artifacts.
    """
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Convert markdown headings to plain text with newlines
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Normalize bullet points
    text = re.sub(r'^[\s]*[-*•]\s+', '- ', text, flags=re.MULTILINE)

    # Normalize numbered lists
    text = re.sub(r'^[\s]*\d+[.)]\s+', '', text, flags=re.MULTILINE)

    # Collapse whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip each line
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text.strip()


# All three produce the same normalized output
pdf_text = "3.1   HNSW Algorithm\n\n\nHNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm."
html_text = "<h2>3.1 HNSW Algorithm</h2><p>HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm.</p>"
md_text = "## 3.1 HNSW Algorithm\n\nHNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm."

print(normalize_to_plaintext(pdf_text))
print("---")
print(normalize_to_plaintext(html_text))
print("---")
print(normalize_to_plaintext(md_text))
# All produce:
# 3.1 HNSW Algorithm
#
# HNSW (Hierarchical Navigable Small World) is a graph-based ANN algorithm.
```

---

## 2. Stable IDs

### The Problem

```
Random UUIDs:
  First ingestion:  doc_id = "a3f2b1c4-..."
  Re-ingestion:     doc_id = "8d7e6f5a-..."  ← DIFFERENT ID for same document!

Content-derived IDs:
  First ingestion:  doc_id = "return_policy__v3"
  Re-ingestion:     doc_id = "return_policy__v3"  ← SAME ID, deterministic
```

### Simple Code — Stable ID Generation

```python
import hashlib
import re
from pathlib import Path


def stable_doc_id(source_path: str) -> str:
    """
    Generate a deterministic document ID from the source path.
    Strips version suffixes so different versions share a base ID.
    """
    stem = Path(source_path).stem
    # Remove version suffixes like _v2, _v3, -draft, etc.
    stem = re.sub(r'[_-]v\d+$', '', stem)
    stem = re.sub(r'[_-]draft$', '', stem, flags=re.IGNORECASE)
    # Normalize
    return stem.lower().replace(' ', '_')


def stable_chunk_id(doc_id: str, chunk_index: int, content: str) -> str:
    """
    Generate a deterministic chunk ID.
    Uses content hash so that identical content always gets the same ID,
    even if chunk_index changes due to re-chunking.
    """
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"{doc_id}__chunk_{chunk_index}__{content_hash}"


# Examples
print(stable_doc_id("docs/return_policy_v3.pdf"))   # return_policy
print(stable_doc_id("docs/Return Policy.pdf"))       # return_policy
print(stable_doc_id("docs/return_policy_v2.pdf"))    # return_policy (same!)

chunk_id = stable_chunk_id(
    "return_policy", 0, "Returns accepted within 7 days."
)
print(chunk_id)  # return_policy__chunk_0__a1b2c3d4
```

---

## 3. Section Path Normalization

### The Problem

```
Documents have hierarchical structure:

  Chapter 3: Indexing Algorithms
    3.1 HNSW
      3.1.1 Multi-layer Structure
    3.2 IVF
      3.2.1 Inverted File Index

When you chunk "3.1.1 Multi-layer Structure", you need to preserve
the path: "Indexing Algorithms > HNSW > Multi-layer Structure"

Without this, a chunk about "Multi-layer Structure" has no context
about WHERE in the document it came from.
```

### Production Code — Section Path Extractor

```python
"""
Extract and normalize section paths from documents.
Preserves hierarchical context for each chunk.
"""

import re
from dataclasses import dataclass


@dataclass
class Section:
    level: int          # heading depth (1 = h1, 2 = h2, etc.)
    title: str
    content: str
    path: str           # full path: "Ch3 > HNSW > Structure"
    start_pos: int      # character position in original doc
    end_pos: int


class SectionPathExtractor:
    """
    Parse a document into sections with hierarchical paths.
    Works with markdown-style headings.
    """

    # Match markdown headings: # Title, ## Title, ### Title
    HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    def __init__(self, separator: str = " > "):
        self.separator = separator

    def extract_sections(self, text: str) -> list[Section]:
        """Parse document into sections with full path context."""
        headings = list(self.HEADING_RE.finditer(text))

        if not headings:
            # No headings — treat entire doc as one section
            return [Section(
                level=0, title="(root)", content=text.strip(),
                path="(root)", start_pos=0, end_pos=len(text),
            )]

        sections = []
        # Stack tracks the current path: [(level, title), ...]
        path_stack: list[tuple[int, str]] = []

        for i, match in enumerate(headings):
            level = len(match.group(1))   # number of # characters
            title = match.group(2).strip()

            # Pop stack until we find a parent level
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, title))

            # Build full path
            path = self.separator.join(t for _, t in path_stack)

            # Content is everything between this heading and the next
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            content = text[start:end].strip()

            sections.append(Section(
                level=level,
                title=title,
                content=content,
                path=path,
                start_pos=match.start(),
                end_pos=end,
            ))

        return sections

    def enrich_chunks_with_paths(
        self, chunks: list[str], sections: list[Section]
    ) -> list[dict]:
        """
        Given chunks and sections, find which section each chunk belongs to
        and attach the section path as metadata.
        """
        enriched = []
        for chunk in chunks:
            # Find the section that contains this chunk
            best_section = None
            for section in sections:
                if chunk[:100] in section.content:
                    best_section = section
                    break

            enriched.append({
                "content": chunk,
                "section_path": best_section.path if best_section else "(unknown)",
                "section_title": best_section.title if best_section else "(unknown)",
                "section_level": best_section.level if best_section else 0,
            })

        return enriched


# ─── Usage ───
if __name__ == "__main__":
    document = """# Vector Database Guide

## Chapter 1: Introduction

Vector databases store high-dimensional embeddings for similarity search.

## Chapter 2: Indexing Algorithms

### HNSW

HNSW builds a multi-layer graph structure for approximate nearest neighbors.

#### Multi-layer Architecture

The top layer has few nodes and long-range connections.
Lower layers have more nodes and shorter connections.

### IVF

IVF uses inverted file indexes to partition the vector space.

## Chapter 3: Querying

Queries are encoded into vectors and compared against the index.
"""

    extractor = SectionPathExtractor()
    sections = extractor.extract_sections(document)

    for s in sections:
        print(f"[L{s.level}] {s.path}")
        print(f"    Content: {s.content[:80]}...")
        print()

    # Output:
    # [L1] Vector Database Guide
    #     Content: ...
    # [L2] Vector Database Guide > Chapter 1: Introduction
    #     Content: Vector databases store high-dimensional embeddings...
    # [L2] Vector Database Guide > Chapter 2: Indexing Algorithms
    #     Content: ...
    # [L3] Vector Database Guide > Chapter 2: Indexing Algorithms > HNSW
    #     Content: HNSW builds a multi-layer graph structure...
    # [L4] Vector Database Guide > Chapter 2: Indexing Algorithms > HNSW > Multi-layer Architecture
    #     Content: The top layer has few nodes...
    # [L3] Vector Database Guide > Chapter 2: Indexing Algorithms > IVF
    #     Content: IVF uses inverted file indexes...
    # [L2] Vector Database Guide > Chapter 3: Querying
    #     Content: Queries are encoded into vectors...
```

---

## Full Canonicalization Pipeline

```python
"""
Complete document canonicalization: normalize format, assign stable IDs,
extract section paths — ready for chunking.
"""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CanonicalDocument:
    doc_id: str
    content: str
    sections: list[dict] = field(default_factory=list)
    source: str = ""
    content_hash: str = ""


class DocumentCanonicalizer:
    """Full canonicalization pipeline."""

    def __init__(self):
        self.section_extractor = SectionPathExtractor()

    def canonicalize(self, raw_text: str, source_path: str) -> CanonicalDocument:
        # Step 1: Normalize formatting
        normalized = self._normalize_format(raw_text)

        # Step 2: Generate stable ID
        doc_id = self._stable_id(source_path)

        # Step 3: Content hash
        content_hash = hashlib.sha256(normalized.encode()).hexdigest()[:16]

        # Step 4: Extract sections with paths
        sections = self.section_extractor.extract_sections(normalized)
        section_dicts = [
            {
                "title": s.title,
                "path": s.path,
                "level": s.level,
                "content": s.content,
            }
            for s in sections
        ]

        return CanonicalDocument(
            doc_id=doc_id,
            content=normalized,
            sections=section_dicts,
            source=source_path,
            content_hash=content_hash,
        )

    def _normalize_format(self, text: str) -> str:
        # Remove HTML
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = '\n'.join(line.strip() for line in text.split('\n'))
        return text.strip()

    def _stable_id(self, source_path: str) -> str:
        stem = Path(source_path).stem
        stem = re.sub(r'[_-]v\d+$', '', stem)
        return stem.lower().replace(' ', '_')


# Usage
canonicalizer = DocumentCanonicalizer()
result = canonicalizer.canonicalize(
    raw_text="## API Reference\n\nThe `/users` endpoint returns...",
    source_path="docs/api_reference_v2.md",
)
print(f"Doc ID: {result.doc_id}")  # api_reference
print(f"Hash: {result.content_hash}")
print(f"Sections: {[s['path'] for s in result.sections]}")
```

---

## Pitfalls & Common Mistakes

| Mistake                      | Impact                                               | Fix                                    |
| ---------------------------- | ---------------------------------------------------- | -------------------------------------- |
| **Random UUIDs as doc IDs**  | Re-ingestion creates duplicate entries               | Use deterministic IDs from source path |
| **No format normalization**  | Same content → different embeddings                  | Normalize before embedding             |
| **Losing heading hierarchy** | Chunks lack context about position in document       | Extract and attach section paths       |
| **Over-normalizing**         | Removing meaningful formatting (code blocks, tables) | Protect structured content blocks      |
| **Inconsistent ID schemes**  | Can't match chunks across reingestions               | Define one ID scheme and stick with it |

---

## Key Takeaways

1. **Normalize format before embedding** — same content should produce same vectors regardless of source format.
2. **Use deterministic document IDs** — derived from source path, not random UUIDs.
3. **Preserve hierarchical section paths** — chunks need to know where they came from in the document structure.
4. **Content hashing enables change detection** — skip re-embedding when content hasn't changed.
5. **Protect structured content** — code blocks, tables, and lists carry meaning that shouldn't be stripped.
