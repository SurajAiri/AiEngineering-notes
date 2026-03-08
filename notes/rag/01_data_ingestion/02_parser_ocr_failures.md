# Parser & OCR Failure Awareness

## Why This Matters

Most RAG systems don't fail at retrieval or generation — they fail at **extraction**. If your parser can't properly read a PDF table, if OCR garbles a scanned document, if reading order is wrong in a multi-column layout — no amount of sophisticated retrieval will fix the broken input.

**You don't need to build parsers from scratch, but you MUST know when they break.**

---

## Common Parser Failures

```
┌─────────────────────────────────────────────────────────────┐
│                PARSER FAILURE MODES                         │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ READING      │  │ TABLE        │  │ LAYOUT       │     │
│  │ ORDER        │  │ EXTRACTION   │  │ CONFUSION    │     │
│  │              │  │              │  │              │     │
│  │ Multi-column │  │ Merged cells │  │ Headers /    │     │
│  │ text read    │  │ become one   │  │ footers      │     │
│  │ left-to-right│  │ long line    │  │ mixed with   │     │
│  │ across both  │  │              │  │ content      │     │
│  │ columns      │  │ Numbers lose │  │              │     │
│  │              │  │ alignment    │  │ Sidebars     │     │
│  │              │  │              │  │ interleaved  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ OCR          │  │ IMAGE /      │  │ ENCODING     │     │
│  │ ERRORS       │  │ CAPTION      │  │ ISSUES       │     │
│  │              │  │ LOSS         │  │              │     │
│  │ "teh" for    │  │ Charts not   │  │ Unicode      │     │
│  │ "the"        │  │ described    │  │ mojibake     │     │
│  │              │  │              │  │              │     │
│  │ "l" for "1"  │  │ Captions     │  │ Ligatures    │     │
│  │ "rn" for "m" │  │ disconnected │  │ broken       │     │
│  │              │  │ from figures │  │ (fi → f i)   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## Real-World Examples of Each Failure

### 1. Reading Order Breaks

```
ORIGINAL PDF (two columns):         PARSED TEXT (wrong order):

┌──────────┬──────────┐             "Vector databases store
│ Vector   │ They use │             They use approximate
│ databases│ approx-  │             embeddings for
│ store    │ imate    │             nearest neighbor
│ embeddings│ nearest │             similarity search.
│ for      │ neighbor │             algorithms like HNSW."
│ similarity│ algo-   │
│ search.  │ rithms   │             ← Reads across columns
│          │ like HNSW│                instead of down each
└──────────┴──────────┘
```

### 2. Table Extraction Failure

```
ORIGINAL TABLE:

| Model      | Params  | MMLU  |
|------------|---------|-------|
| GPT-4      | ~1.7T   | 86.4  |
| Claude 3   | Unknown | 86.8  |

PARSED AS:
"Model Params MMLU GPT-4 ~1.7T 86.4 Claude 3 Unknown 86.8"

← All structure lost! Numbers no longer aligned with models.
```

### 3. OCR Errors in Scanned Documents

```
ORIGINAL:     "The algorithm converges in O(n log n) time."
OCR RESULT:   "Tbe algorithrn converges in 0(n Iog n) tirne."

Errors: "The" → "Tbe", "algorithm" → "algorithrn",
        "O" → "0", "log" → "Iog", "time" → "tirne"
```

---

## Detecting Parser Failures

### Simple Code — Quality Checks

```python
"""
Basic quality checks to detect parser/OCR failures.
Run these on extracted text BEFORE chunking and embedding.
"""

import re
from dataclasses import dataclass


@dataclass
class ExtractionQuality:
    score: float           # 0.0 to 1.0
    issues: list[str]
    is_acceptable: bool


def check_extraction_quality(text: str) -> ExtractionQuality:
    """Run heuristic quality checks on extracted text."""
    issues = []
    penalties = 0.0

    # 1. Check for OCR garbage — high ratio of non-alpha characters
    if text:
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
        if alpha_ratio < 0.6:
            issues.append(f"Low alpha ratio: {alpha_ratio:.2f} (possible OCR garbage)")
            penalties += 0.3

    # 2. Check for encoding issues — mojibake patterns
    mojibake_patterns = [r'Ã¤', r'Ã¶', r'Ã¼', r'â€™', r'â€œ', r'Â©']
    for pattern in mojibake_patterns:
        if re.search(pattern, text):
            issues.append(f"Mojibake detected: {pattern}")
            penalties += 0.2
            break

    # 3. Check for reading order issues — sentences that don't make sense
    # Heuristic: very short "sentences" suggest reading order breaks
    sentences = re.split(r'[.!?]+', text)
    short_sentences = sum(1 for s in sentences if 0 < len(s.strip()) < 10)
    if sentences and short_sentences / len(sentences) > 0.5:
        issues.append("Many very short sentences (possible reading order break)")
        penalties += 0.2

    # 4. Check for table destruction — repeated patterns of mixed text/numbers
    # with no structure
    lone_numbers = re.findall(r'\b\d+\.?\d*\b', text)
    if len(lone_numbers) > 20 and len(text) < 500:
        issues.append("Dense numbers without structure (possible table extraction failure)")
        penalties += 0.2

    # 5. Check for missing content — very little text from a large document
    if len(text.strip()) < 100:
        issues.append("Very little text extracted")
        penalties += 0.3

    # 6. Check for excessive whitespace (common in bad PDF extraction)
    if text:
        whitespace_ratio = text.count(' ') / len(text)
        if whitespace_ratio > 0.4:
            issues.append(f"High whitespace ratio: {whitespace_ratio:.2f}")
            penalties += 0.1

    score = max(0.0, 1.0 - penalties)
    return ExtractionQuality(
        score=score,
        issues=issues,
        is_acceptable=score >= 0.5,
    )


# Test with good text
good_text = """
Vector databases are specialized systems designed to store, index, and
query high-dimensional vector embeddings. They enable similarity search
across large collections of unstructured data.
"""

# Test with OCR-damaged text
ocr_text = "Tbe algorithrn converges in 0(n Iog n) tirne. lt uses a dMde and conquer apprcach."

# Test with table destruction
table_text = "Model Params MMLU GPT-4 1.7T 86.4 Claude 3 Unknown 86.8 Llama 70B 82.0 Mixtral 46.7B 81.2"

for label, text in [("Good", good_text), ("OCR", ocr_text), ("Table", table_text)]:
    result = check_extraction_quality(text)
    print(f"{label}: score={result.score:.2f}, acceptable={result.is_acceptable}")
    for issue in result.issues:
        print(f"  ⚠ {issue}")
    print()
```

---

## Production Code — Parser Failure Detection & Fallback

```python
"""
Production parser failure detection with fallback strategies.
Tries multiple extraction methods and picks the best result.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    text: str
    method: str
    quality_score: float
    page_count: int = 0
    has_tables: bool = False
    has_images: bool = False
    issues: list[str] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class BaseExtractor(ABC):
    """Base class for document extractors."""

    @abstractmethod
    def extract(self, file_path: str) -> ExtractionResult:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class PyMuPDFExtractor(BaseExtractor):
    """Extract using PyMuPDF (fitz). Good for text-native PDFs."""

    def name(self) -> str:
        return "pymupdf"

    def extract(self, file_path: str) -> ExtractionResult:
        try:
            import fitz
            doc = fitz.open(file_path)
            pages = []
            for page in doc:
                pages.append(page.get_text())
            text = '\n\n'.join(pages)
            doc.close()

            return ExtractionResult(
                text=text,
                method=self.name(),
                quality_score=self._assess_quality(text),
                page_count=len(pages),
            )
        except Exception as e:
            return ExtractionResult(
                text="", method=self.name(), quality_score=0.0,
                issues=[f"Extraction failed: {str(e)}"],
            )

    def _assess_quality(self, text: str) -> float:
        if not text.strip():
            return 0.0
        alpha_ratio = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
        return min(alpha_ratio * 1.2, 1.0)


class OCRExtractor(BaseExtractor):
    """Extract using OCR. Fallback for scanned documents."""

    def name(self) -> str:
        return "ocr_tesseract"

    def extract(self, file_path: str) -> ExtractionResult:
        # In production, you'd use pytesseract + pdf2image
        # This is a placeholder showing the interface
        return ExtractionResult(
            text="[OCR extraction would happen here]",
            method=self.name(),
            quality_score=0.6,  # OCR typically lower quality
            issues=["OCR quality varies — manual review recommended"],
        )


class LLMExtractor(BaseExtractor):
    """Use an LLM (vision model) to extract text. Highest quality, highest cost."""

    def name(self) -> str:
        return "llm_vision"

    def extract(self, file_path: str) -> ExtractionResult:
        # Would use GPT-4V, Claude vision, etc.
        return ExtractionResult(
            text="[LLM vision extraction would happen here]",
            method=self.name(),
            quality_score=0.9,
            issues=["High cost per page"],
        )


class RobustExtractor:
    """
    Tries multiple extraction methods and picks the best result.
    Falls back to more expensive methods when cheap ones fail.
    """

    def __init__(
        self,
        extractors: list[BaseExtractor] | None = None,
        min_quality: float = 0.5,
    ):
        self.extractors = extractors or [
            PyMuPDFExtractor(),
            OCRExtractor(),
            LLMExtractor(),
        ]
        self.min_quality = min_quality

    def extract(self, file_path: str) -> ExtractionResult:
        """Try extractors in order, return the first acceptable result."""
        results = []

        for extractor in self.extractors:
            logger.info(f"Trying {extractor.name()} for {file_path}")
            result = extractor.extract(file_path)
            results.append(result)

            if result.quality_score >= self.min_quality:
                logger.info(
                    f"Accepted {extractor.name()} result "
                    f"(quality={result.quality_score:.2f})"
                )
                return result
            else:
                logger.warning(
                    f"{extractor.name()} quality too low "
                    f"({result.quality_score:.2f}), trying next"
                )

        # All extractors tried — return the best one
        best = max(results, key=lambda r: r.quality_score)
        logger.warning(
            f"No extractor met quality threshold. "
            f"Best: {best.method} ({best.quality_score:.2f})"
        )
        best.issues.append("Below quality threshold — manual review recommended")
        return best


class TableDetector:
    """Detect and validate table extraction quality."""

    @staticmethod
    def detect_tables(text: str) -> list[dict]:
        """Find potential tables in extracted text and assess quality."""
        tables = []

        # Look for markdown-style tables
        table_pattern = re.compile(
            r'(\|.+\|)\n(\|[-:| ]+\|)\n((?:\|.+\|\n?)+)',
            re.MULTILINE,
        )
        for match in table_pattern.finditer(text):
            header = match.group(1)
            rows = match.group(3).strip().split('\n')
            cols = len(header.split('|')) - 2  # exclude outer pipes

            tables.append({
                "type": "markdown_table",
                "rows": len(rows),
                "cols": cols,
                "quality": "good",
                "position": match.start(),
            })

        # Look for destroyed tables (numbers jumbled together)
        number_dense = re.findall(r'(?:\d+\.?\d*\s+){5,}', text)
        for match_text in number_dense:
            tables.append({
                "type": "destroyed_table",
                "raw": match_text[:100],
                "quality": "bad",
            })

        return tables


# ─── Usage ───

if __name__ == "__main__":
    # Extraction with fallback
    extractor = RobustExtractor(min_quality=0.6)
    # result = extractor.extract("path/to/document.pdf")

    # Table detection
    detector = TableDetector()

    good_table = """
Some text before the table.

| Model | Params | MMLU |
|-------|--------|------|
| GPT-4 | 1.7T  | 86.4 |
| Claude | ???   | 86.8 |

Some text after.
"""

    bad_table = """
Model Params MMLU GPT-4 1.7T 86.4 Claude ??? 86.8 Llama 70B 82.0
"""

    print("Good table:")
    for t in detector.detect_tables(good_table):
        print(f"  {t}")

    print("\nBad table:")
    for t in detector.detect_tables(bad_table):
        print(f"  {t}")
```

---

## Common Parser Tools & Their Limitations

| Tool                            | Strengths                       | Known Failures                                       |
| ------------------------------- | ------------------------------- | ---------------------------------------------------- |
| **PyMuPDF**                     | Fast, good for text-native PDFs | Multi-column, complex tables                         |
| **pdfplumber**                  | Better table extraction         | Slow on large docs, header/footer confusion          |
| **Unstructured**                | Multi-format, layout analysis   | Can miss formatting, expensive for large batches     |
| **LlamaParse**                  | LLM-powered, understands layout | Cost per page, latency, API dependency               |
| **Tesseract OCR**               | Free, works offline             | Low accuracy on noisy scans, no layout understanding |
| **Amazon Textract**             | Good OCR + table extraction     | Cost, latency, AWS dependency                        |
| **Azure Document Intelligence** | Forms + tables                  | Cost, Azure dependency                               |

---

## Defensive Patterns

```python
"""
Defensive patterns for handling parser failures gracefully.
"""

import re


def add_extraction_warning(text: str, method: str, quality: float) -> str:
    """
    Prepend a warning to low-quality extractions so the LLM knows
    the source may be unreliable.
    """
    if quality < 0.7:
        warning = (
            f"[NOTE: This text was extracted using {method} with "
            f"quality score {quality:.2f}. Content may contain errors.]\n\n"
        )
        return warning + text
    return text


def preserve_table_structure(text: str) -> str:
    """
    Attempt to reconstruct table structure from flat text.
    If we detect a destroyed table, wrap it in a note.
    """
    # Detect sequences of numbers/words that look like table rows
    # This is a heuristic — not perfect
    lines = text.split('\n')
    reconstructed = []

    for line in lines:
        # If a line has multiple tab/multi-space separated values, it might be a table row
        parts = re.split(r'\s{2,}|\t', line.strip())
        if len(parts) >= 3 and any(re.match(r'\d', p) for p in parts):
            reconstructed.append(' | '.join(parts))
        else:
            reconstructed.append(line)

    return '\n'.join(reconstructed)


def validate_extraction_completeness(
    extracted: str, expected_pages: int, avg_chars_per_page: int = 1500
) -> dict:
    """
    Check if extraction seems complete.
    Flags suspiciously short extractions.
    """
    expected_chars = expected_pages * avg_chars_per_page
    actual_chars = len(extracted)
    ratio = actual_chars / expected_chars if expected_chars > 0 else 0

    return {
        "expected_chars": expected_chars,
        "actual_chars": actual_chars,
        "completeness_ratio": ratio,
        "likely_complete": ratio > 0.5,
        "warning": (
            f"Only {ratio:.0%} of expected content extracted"
            if ratio < 0.5 else None
        ),
    }
```

---

## Pitfalls & Common Mistakes

| Mistake                             | Impact                                                                 | Fix                                                                         |
| ----------------------------------- | ---------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Trusting parser output blindly**  | Garbage text gets embedded and retrieved                               | Always run quality checks on extracted text                                 |
| **One parser for all formats**      | PDF parser fails on scanned docs, HTML parser fails on complex layouts | Use format-specific parsers with fallbacks                                  |
| **Not detecting table destruction** | Numbers misattributed to wrong entities                                | Detect tables separately, use specialized table extraction                  |
| **Ignoring reading order**          | Multi-column docs create nonsense chunks                               | Use layout-aware parsers or validate sentence coherence                     |
| **No extraction quality logging**   | Can't identify which docs need reprocessing                            | Log quality scores and flag low-quality extractions                         |
| **Embedding OCR artifacts**         | Misspelled text creates bad vectors                                    | Clean OCR output before embedding; consider re-extracting with better tools |

---

## Key Takeaways

1. **Most RAG failures start at extraction** — not at retrieval or generation.
2. **Always check extraction quality** — use heuristics to flag bad extractions.
3. **Have fallback extractors** — try fast/cheap first, fall back to expensive when quality is low.
4. **Tables need special handling** — generic parsers usually destroy table structure.
5. **Log extraction quality** — you need to know which documents need reprocessing.
6. **You don't need to build parsers** — but you must know when they fail and have a recovery plan.

---

## Popular Libraries

| Library      | Best For                          | Install                    |
| ------------ | --------------------------------- | -------------------------- |
| PyMuPDF      | Fast PDF text + layout extraction | `pip install pymupdf`      |
| LlamaParse   | LLM-powered complex doc parsing   | Via LlamaIndex Cloud API   |
| Unstructured | Multi-format auto-detection       | `pip install unstructured` |
| pdfplumber   | PDF tables extraction             | `pip install pdfplumber`   |
| Tesseract    | OCR for scanned documents         | `brew install tesseract`   |

### Quick Example — PyMuPDF (fitz)

```python
import fitz  # pip install pymupdf

def extract_pdf_with_quality_check(path: str) -> list[dict]:
    doc = fitz.open(path)
    pages = []
    for page in doc:
        text = page.get_text()
        # Basic quality check
        if len(text.strip()) < 20:
            print(f"Warning: Page {page.number} has very little text (possible scan)")
        pages.append({"page": page.number, "text": text, "char_count": len(text)})
    return pages

pages = extract_pdf_with_quality_check("report.pdf")
```

### Quick Example — LlamaParse (for complex documents)

```python
from llama_parse import LlamaParse

parser = LlamaParse(
    result_type="markdown",  # Get clean markdown output
    num_workers=4,
    verbose=True,
)

# Handles tables, images, complex layouts
documents = parser.load_data("complex_report.pdf")
for doc in documents:
    print(doc.text[:200])
```

---

## Common Questions

### Q: Which parser should I start with?

**A:** Start with **PyMuPDF** — it's fast, free, and handles most PDFs well. If you encounter scanned docs or complex tables, add **LlamaParse** as a fallback. Build a quality-score router that automatically escalates to the better (more expensive) parser when the cheap one fails.

### Q: How do I handle scanned PDFs vs native PDFs?

**A:** Check if the PDF has selectable text (native) or is an image scan. PyMuPDF's `page.get_text()` returns empty/garbage for scans. If text is < 50 chars per page, route to OCR (Tesseract) or LlamaParse.

### Q: Should I extract tables as text or structured data?

**A:** Extract tables as **structured data** (CSV/markdown format) whenever possible. Flattening a table to plain text destroys the row/column relationships. Use `pdfplumber` for simple tables or LlamaParse for complex ones.
