# Clean vs Noisy Data Handling

## Why This Matters

RAG quality is bounded by data quality. If your documents are noisy — full of boilerplate, OCR artifacts, duplicated headers, or broken formatting — your embeddings will encode noise, your retrieval will return irrelevant chunks, and your LLM will hallucinate or produce unreliable answers.

**The rule:** Garbage in, garbage out — but in RAG, it's worse: garbage in, _confidently wrong_ answers out.

---

## Clean vs Noisy Data — What's the Difference?

```
┌─────────────────────────────────────────────────────────┐
│                     CLEAN DATA                          │
│                                                         │
│  • Well-structured text with clear sections              │
│  • Consistent formatting                                │
│  • No artifacts from conversion                         │
│  • Metadata is accurate and complete                    │
│  • Content is current and authoritative                 │
│  • Tables are properly formatted                        │
│  • Code blocks are intact                               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     NOISY DATA                          │
│                                                         │
│  • OCR artifacts: "teh" instead of "the"                │
│  • Broken tables: rows merged into single lines         │
│  • Headers/footers repeated in every chunk              │
│  • HTML artifacts: &amp;nbsp;, <div>, leftover tags     │
│  • Mixed languages unexpectedly                         │
│  • Watermarks embedded in text                          │
│  • Navigation menus scraped with content                │
│  • Duplicate paragraphs from copy-paste errors          │
└─────────────────────────────────────────────────────────┘
```

---

## The Data Cleaning Pipeline

```
Raw Documents
     │
     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Extract    │───▶│   Clean &    │───▶│   Validate   │
│   (Parse)    │    │   Normalize  │    │   Quality    │
└──────────────┘    └──────────────┘    └──────────────┘
     │                    │                    │
     │                    │                    │
  • PDF → text         • Remove noise       • Check coverage
  • HTML → text        • Fix encoding       • Spot-check samples
  • DOCX → text        • Normalize format   • Measure signal/noise
  • Images → OCR       • Strip boilerplate  • Flag low quality
```

---

## Simple Code — Understanding the Concept

This minimal example shows common cleaning operations:

```python
import re

def clean_text(raw_text: str) -> str:
    """Basic text cleaning for RAG ingestion."""

    # 1. Fix common encoding issues
    text = raw_text.replace('\x00', '')           # null bytes
    text = text.replace('\xa0', ' ')              # non-breaking spaces
    text = text.replace('\u200b', '')             # zero-width spaces

    # 2. Remove excessive whitespace (but preserve paragraph breaks)
    text = re.sub(r'[ \t]+', ' ', text)           # collapse horizontal spaces
    text = re.sub(r'\n{3,}', '\n\n', text)        # max 2 newlines

    # 3. Remove common boilerplate patterns
    text = re.sub(r'Page \d+ of \d+', '', text)   # page numbers
    text = re.sub(r'©.*?\d{4}', '', text)         # copyright lines

    # 4. Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


# Example usage
raw = """
    Page 1 of 5


    Introduction to\xa0Machine Learning

    Machine   learning is a    subset of artificial intelligence.
    It allows     systems to learn from data.

    © Acme Corp 2024
    Page 2 of 5
"""

cleaned = clean_text(raw)
print(cleaned)
# Output:
# Introduction to Machine Learning
#
# Machine learning is a subset of artificial intelligence.
# It allows systems to learn from data.
```

---

## Production Code — Full Cleaning Pipeline

```python
"""
Production data cleaning pipeline for RAG ingestion.
Handles multiple document types, logs quality metrics, and flags issues.
"""

import re
import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class QualityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECTED = "rejected"


@dataclass
class CleaningResult:
    original_length: int
    cleaned_length: int
    cleaned_text: str
    quality: QualityLevel
    issues: list[str] = field(default_factory=list)
    noise_ratio: float = 0.0  # fraction of text that was noise


class TextCleaner:
    """Configurable text cleaner for RAG document ingestion."""

    # Patterns that are almost always noise
    BOILERPLATE_PATTERNS = [
        r'Page\s+\d+\s*(of\s+\d+)?',           # page numbers
        r'©.*?\d{4}',                            # copyright
        r'All rights reserved\.?',               # legal
        r'Table of Contents',                    # TOC headers (content is separate)
        r'^\s*\d+\s*$',                          # standalone page numbers
        r'CONFIDENTIAL',                         # watermarks
        r'DRAFT',                                # watermarks
        r'(?:https?://)?(?:www\.)?[\w.-]+\.(?:com|org|net|io)(?:/\S*)?',  # URLs (optional)
    ]

    # Characters that suggest OCR/encoding problems
    GARBAGE_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]')

    def __init__(
        self,
        remove_urls: bool = False,
        min_line_length: int = 3,
        max_noise_ratio: float = 0.5,
        custom_boilerplate: list[str] | None = None,
    ):
        self.remove_urls = remove_urls
        self.min_line_length = min_line_length
        self.max_noise_ratio = max_noise_ratio

        patterns = self.BOILERPLATE_PATTERNS.copy()
        if not remove_urls:
            patterns = [p for p in patterns if 'https?' not in p]
        if custom_boilerplate:
            patterns.extend(custom_boilerplate)

        self.boilerplate_re = re.compile(
            '|'.join(f'({p})' for p in patterns),
            re.IGNORECASE | re.MULTILINE,
        )

    def clean(self, text: str, source: str = "unknown") -> CleaningResult:
        original_length = len(text)
        issues = []

        # Step 1: Fix encoding
        text = self._fix_encoding(text, issues)

        # Step 2: Remove boilerplate
        text = self._remove_boilerplate(text, issues)

        # Step 3: Normalize whitespace
        text = self._normalize_whitespace(text)

        # Step 4: Remove too-short lines (likely noise)
        text = self._remove_short_lines(text, issues)

        # Step 5: Final cleanup
        text = text.strip()
        cleaned_length = len(text)

        # Quality assessment
        noise_ratio = 1 - (cleaned_length / original_length) if original_length > 0 else 0
        quality = self._assess_quality(text, noise_ratio, issues)

        if quality == QualityLevel.REJECTED:
            logger.warning(f"Document rejected: source={source}, issues={issues}")
        elif quality == QualityLevel.LOW:
            logger.warning(f"Low quality document: source={source}, issues={issues}")

        return CleaningResult(
            original_length=original_length,
            cleaned_length=cleaned_length,
            cleaned_text=text,
            quality=quality,
            issues=issues,
            noise_ratio=noise_ratio,
        )

    def _fix_encoding(self, text: str, issues: list[str]) -> str:
        garbage_count = len(self.GARBAGE_CHARS.findall(text))
        if garbage_count > 0:
            issues.append(f"Found {garbage_count} garbage characters")
            text = self.GARBAGE_CHARS.sub('', text)

        # Common encoding replacements
        replacements = {
            '\xa0': ' ',        # non-breaking space
            '\u200b': '',       # zero-width space
            '\u200c': '',       # zero-width non-joiner
            '\u200d': '',       # zero-width joiner
            '\ufeff': '',       # BOM
            '\u2018': "'",      # left single quote
            '\u2019': "'",      # right single quote
            '\u201c': '"',      # left double quote
            '\u201d': '"',      # right double quote
            '\u2013': '-',      # en dash
            '\u2014': '--',     # em dash
            '\u2026': '...',    # ellipsis
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def _remove_boilerplate(self, text: str, issues: list[str]) -> str:
        matches = self.boilerplate_re.findall(text)
        if matches:
            issues.append(f"Removed {len(matches)} boilerplate matches")
        return self.boilerplate_re.sub('', text)

    def _normalize_whitespace(self, text: str) -> str:
        # Collapse horizontal whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        # Max 2 consecutive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _remove_short_lines(self, text: str, issues: list[str]) -> str:
        lines = text.split('\n')
        kept = []
        removed = 0
        for line in lines:
            stripped = line.strip()
            if len(stripped) >= self.min_line_length or stripped == '':
                kept.append(line)
            else:
                removed += 1
        if removed > 0:
            issues.append(f"Removed {removed} short lines")
        return '\n'.join(kept)

    def _assess_quality(
        self, text: str, noise_ratio: float, issues: list[str]
    ) -> QualityLevel:
        if len(text) < 50:
            issues.append("Text too short after cleaning")
            return QualityLevel.REJECTED
        if noise_ratio > self.max_noise_ratio:
            issues.append(f"Noise ratio too high: {noise_ratio:.2%}")
            return QualityLevel.REJECTED
        if noise_ratio > 0.3:
            return QualityLevel.LOW
        if noise_ratio > 0.1:
            return QualityLevel.MEDIUM
        return QualityLevel.HIGH


# ─── Usage ───

if __name__ == "__main__":
    cleaner = TextCleaner(
        remove_urls=True,
        custom_boilerplate=[r'INTERNAL USE ONLY'],
    )

    raw_document = """
    Page 1 of 12
    CONFIDENTIAL
    INTERNAL USE ONLY

    Chapter 1: Introduction to Vector Databases

    A vector database is a specialized database designed to store, index,
    and query high-dimensional vector embeddings. These embeddings are
    numerical representations of data (text, images, audio) that capture
    semantic meaning.

    Key concepts:
    - Embeddings: Dense vector representations
    - Similarity search: Finding nearest neighbors
    - Indexing: HNSW, IVF, and other algorithms

    For more info visit https://example.com/docs

    © VectorDB Corp 2024
    All rights reserved.
    Page 2 of 12
    """

    result = cleaner.clean(raw_document, source="vectordb_intro.pdf")

    print(f"Quality: {result.quality.value}")
    print(f"Noise ratio: {result.noise_ratio:.1%}")
    print(f"Issues: {result.issues}")
    print(f"--- Cleaned text ---")
    print(result.cleaned_text)
```

---

## Domain-Specific Cleaning

Different document types need different cleaning strategies:

````python
"""
Domain-specific cleaners that extend the base cleaner.
"""

import re
from abc import ABC, abstractmethod


class DomainCleaner(ABC):
    """Base class for domain-specific cleaning."""

    @abstractmethod
    def clean(self, text: str) -> str:
        pass


class LegalDocumentCleaner(DomainCleaner):
    """Cleans legal documents — preserves section numbers, removes exhibit markers."""

    def clean(self, text: str) -> str:
        # Remove exhibit markers but keep section references
        text = re.sub(r'\[Exhibit \w+\]', '', text)
        # Remove repeated party names in full caps (common in legal docs)
        text = re.sub(r'^[A-Z\s,]+(?:INC|LLC|CORP|LTD)\.?\s*$', '', text, flags=re.MULTILINE)
        # Normalize section references
        text = re.sub(r'Section\s+(\d+)', r'§\1', text)
        return text


class MedicalDocumentCleaner(DomainCleaner):
    """Cleans medical/clinical documents."""

    def clean(self, text: str) -> str:
        # Remove patient identifiers (basic PHI removal — use proper tools in production!)
        text = re.sub(r'MRN:\s*\d+', 'MRN: [REDACTED]', text)
        text = re.sub(r'DOB:\s*\d{1,2}/\d{1,2}/\d{2,4}', 'DOB: [REDACTED]', text)
        # Keep medical abbreviations intact
        # Remove discharge summary boilerplate
        text = re.sub(r'={3,}', '', text)
        return text


class CodeDocumentCleaner(DomainCleaner):
    """Cleans technical documentation — preserves code blocks."""

    def clean(self, text: str) -> str:
        # Protect code blocks from other cleaning
        code_blocks = []
        def save_code(match):
            code_blocks.append(match.group(0))
            return f'__CODE_BLOCK_{len(code_blocks) - 1}__'

        text = re.sub(r'```[\s\S]*?```', save_code, text)

        # Clean non-code portions
        text = re.sub(r'^\s*#\s*Table of Contents\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\[↑ Back to top\].*$', '', text, flags=re.MULTILINE)

        # Restore code blocks
        for i, block in enumerate(code_blocks):
            text = text.replace(f'__CODE_BLOCK_{i}__', block)

        return text
````

---

## Pitfalls & Common Mistakes

| Mistake                     | Why It's Bad                                                                         | Fix                                                    |
| --------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| **Over-cleaning**           | Removing meaningful content (e.g., stripping all numbers from financial docs)        | Create domain-specific rules, test with sample queries |
| **Ignoring encoding**       | `\xa0` and zero-width chars create invisible differences that break dedup and search | Always normalize Unicode first                         |
| **No quality metrics**      | You never know if ingestion is degrading                                             | Log noise ratio, character counts, and quality scores  |
| **One pipeline for all**    | Legal docs ≠ code docs ≠ medical records                                             | Build domain-specific cleaners                         |
| **Cleaning too late**       | Cleaning after chunking means noise is already embedded in vector space              | Always clean before chunking                           |
| **Removing all formatting** | Headings, bullet points, and structure carry semantic meaning                        | Normalize formatting, don't destroy it                 |

---

## Trade-offs

```
Aggressive Cleaning ◄──────────────────────────────► Minimal Cleaning

Pros:                                               Pros:
• Less noise in embeddings                          • No information loss
• More focused retrieval                            • Preserves original context
• Smaller index size                                • Faster pipeline

Cons:                                               Cons:
• Risk of removing useful info                      • Noise in embeddings
• More complex pipeline                             • Worse retrieval precision
• Domain knowledge needed                           • Larger index
```

---

## Key Takeaways

1. **Clean before you chunk.** Noise in your text becomes noise in your embeddings.
2. **Measure cleaning quality** — log noise ratios, track rejected documents.
3. **Domain-specific cleaning** beats one-size-fits-all every time.
4. **Preserve structure** — headings, lists, and formatting carry meaning.
5. **Test cleaning with retrieval** — verify that cleaned text still retrieves correctly for your queries.
