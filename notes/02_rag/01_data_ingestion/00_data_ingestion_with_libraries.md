# Data Ingestion with Libraries — Practical Cookbook

## 🟢 How to Approach This Topic

> **Why this matters for your job:** In production, you won't write PDF parsers from scratch. You'll pick from battle-tested libraries like Docling, Unstructured, LlamaParse, or LangChain document loaders. Knowing which tool to use for which data — and understanding when they fail — is what separates a working system from a broken one.

**Prerequisites:** Read [01_clean_vs_noisy_data.md](./01_clean_vs_noisy_data.md) first to understand WHY data quality matters.

**Reading order:**

1. Skim the library comparison table below (5 min)
2. Try the "Quick Start" for your data type (15 min)
3. Run the "Same PDF, 4 Parsers" comparison on your own data (30 min)
4. Read the decision framework and pitfalls (10 min)
5. Explore advanced patterns as needed

**⏱️ Core concept: 1 hour | Full exploration: 3 hours**

---

## Library Landscape (2025–2026)

```
                        ┌─────────────────────────┐
                        │     Your Raw Documents   │
                        │  (PDF, HTML, DOCX, etc.) │
                        └────────────┬────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
         ┌────▼────┐          ┌──────▼──────┐         ┌─────▼─────┐
         │ Simple  │          │  Complex    │         │  Scanned  │
         │ Text    │          │  Layout     │         │  / Image  │
         │ PDFs    │          │  (tables,   │         │  PDFs     │
         │         │          │  columns)   │         │           │
         └────┬────┘          └──────┬──────┘         └─────┬─────┘
              │                      │                      │
         PyMuPDF               Docling /               LlamaParse /
         pdfplumber            Unstructured             Unstructured
         LangChain             LlamaParse               (with OCR)
         loaders               Docling
```

| Library                | Best For                             | Speed         | Quality                         | Cost                      | Notes                                          |
| ---------------------- | ------------------------------------ | ------------- | ------------------------------- | ------------------------- | ---------------------------------------------- |
| **PyMuPDF (fitz)**     | Text-native PDFs                     | ⚡ Very fast  | ⭐⭐⭐ Good for simple docs     | Free                      | No OCR, no table detection                     |
| **pdfplumber**         | PDFs with tables                     | ⚡ Fast       | ⭐⭐⭐⭐ Great table extraction | Free                      | Better table handling than PyMuPDF             |
| **Docling**            | Scientific/structured docs           | 🔄 Medium     | ⭐⭐⭐⭐⭐ Excellent structure  | Free                      | IBM's parser. Tables, figures, structure-aware |
| **Unstructured**       | Multi-format (PDF, HTML, DOCX, PPTX) | 🔄 Medium     | ⭐⭐⭐⭐ Good general quality   | Free (local) / Paid (API) | Most formats supported                         |
| **LlamaParse**         | Complex layouts, scanned docs        | 🐌 Slow (API) | ⭐⭐⭐⭐⭐ Best for hard docs   | Paid API                  | LLM-powered parsing                            |
| **LangChain Loaders**  | Quick prototyping                    | ⚡ Fast       | ⭐⭐⭐ Varies by loader         | Free                      | Wraps other parsers, many integrations         |
| **LlamaIndex Readers** | LlamaIndex pipelines                 | ⚡ Fast       | ⭐⭐⭐ Varies by reader         | Free                      | SimpleDirectoryReader, many formats            |
| **BeautifulSoup**      | HTML/web content                     | ⚡ Very fast  | ⭐⭐⭐⭐ Good for HTML          | Free                      | Standard for web scraping                      |
| **markitdown**         | Office docs to markdown              | ⚡ Fast       | ⭐⭐⭐⭐ Good for Office files  | Free                      | Microsoft's tool, DOCX/PPTX/XLSX               |

---

## Quick Start — By Data Type

### PDF (Simple, Text-Native)

```python
# Option 1: PyMuPDF (fastest for simple PDFs)
import fitz  # pip install pymupdf

def parse_pdf_pymupdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    doc.close()
    return text

text = parse_pdf_pymupdf("report.pdf")
print(text[:500])
```

```python
# Option 2: LangChain loader (convenient, wraps PyMuPDF)
# pip install langchain-community pymupdf
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("report.pdf")
docs = loader.load()

for doc in docs:
    print(f"Page {doc.metadata['page']}: {doc.page_content[:200]}")
    # Each doc has .page_content (str) and .metadata (dict)
```

```python
# Option 3: LlamaIndex reader
# pip install llama-index-readers-file
from llama_index.readers.file import PDFReader

reader = PDFReader()
documents = reader.load_data("report.pdf")

for doc in documents:
    print(doc.text[:200])
    print(doc.metadata)  # includes page numbers
```

### PDF (Complex Layout — Tables, Columns, Figures)

```python
# Option 1: Docling (best for structured/scientific documents)
# pip install docling
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("complex_report.pdf")

# Get full markdown (preserves structure!)
markdown = result.document.export_to_markdown()
print(markdown[:1000])

# Access tables specifically
for table in result.document.tables:
    print(table.export_to_dataframe())  # pandas DataFrame!
```

```python
# Option 2: pdfplumber (great for tables without external APIs)
# pip install pdfplumber
import pdfplumber

with pdfplumber.open("financial_report.pdf") as pdf:
    for page in pdf.pages:
        # Extract text
        text = page.extract_text()
        print(text[:200])

        # Extract tables as list of lists
        tables = page.extract_tables()
        for table in tables:
            for row in table:
                print(row)
```

```python
# Option 3: LlamaParse (LLM-powered, best quality for hard docs)
# pip install llama-parse
from llama_parse import LlamaParse

# Requires LLAMA_CLOUD_API_KEY env var
parser = LlamaParse(
    result_type="markdown",  # or "text"
    num_workers=4,
    verbose=True,
)

documents = parser.load_data("complex_report.pdf")
for doc in documents:
    print(doc.text[:500])
```

### PDF (Scanned / Image-Based)

```python
# Unstructured with OCR (handles scanned documents)
# pip install unstructured[pdf]  # includes pytesseract, pdf2image
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    "scanned_doc.pdf",
    strategy="ocr_only",  # or "hi_res" for layout-aware OCR
    ocr_languages="eng",
)

for element in elements:
    print(f"[{element.category}] {element.text[:200]}")
    # Categories: Title, NarrativeText, ListItem, Table, Image, etc.
```

### HTML / Web Pages

```python
# BeautifulSoup (manual control)
# pip install beautifulsoup4 requests
from bs4 import BeautifulSoup
import requests

response = requests.get("https://example.com/docs/api-guide", timeout=10)
soup = BeautifulSoup(response.content, "html.parser")

# Remove scripts, styles, nav
for tag in soup(["script", "style", "nav", "footer", "header"]):
    tag.decompose()

text = soup.get_text(separator="\n", strip=True)
print(text[:500])
```

```python
# LangChain web loader
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com/docs/api-guide")
docs = loader.load()
print(docs[0].page_content[:500])
```

### DOCX / PPTX / Office Documents

```python
# markitdown (Microsoft's tool — DOCX, PPTX, XLSX to markdown)
# pip install markitdown
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert("presentation.pptx")
print(result.text_content[:500])
```

```python
# Docling (also handles Office formats)
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("report.docx")
print(result.document.export_to_markdown()[:500])
```

```python
# LangChain DOCX loader
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("contract.docx")
docs = loader.load()
print(docs[0].page_content[:500])
```

---

## Same PDF, Multiple Parsers — Quality Comparison

This is the most important exercise you can do. Run the same document through multiple parsers and compare output quality.

```python
"""
Compare parser output quality on the same PDF.
Run this on YOUR actual documents to pick the right parser.
"""
from pathlib import Path


def compare_parsers(pdf_path: str):
    results = {}

    # 1. PyMuPDF
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        results["pymupdf"] = text
    except Exception as e:
        results["pymupdf"] = f"ERROR: {e}"

    # 2. pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        results["pdfplumber"] = text
    except Exception as e:
        results["pdfplumber"] = f"ERROR: {e}"

    # 3. Unstructured
    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(pdf_path, strategy="auto")
        text = "\n".join(el.text for el in elements if el.text)
        results["unstructured"] = text
    except Exception as e:
        results["unstructured"] = f"ERROR: {e}"

    # 4. Docling
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        text = result.document.export_to_markdown()
        results["docling"] = text
    except Exception as e:
        results["docling"] = f"ERROR: {e}"

    # Compare
    print(f"\n{'='*60}")
    print(f"PARSING COMPARISON: {Path(pdf_path).name}")
    print(f"{'='*60}")

    for parser_name, text in results.items():
        if text.startswith("ERROR"):
            print(f"\n--- {parser_name} ---")
            print(text)
            continue

        print(f"\n--- {parser_name} ---")
        print(f"  Length: {len(text)} chars")
        print(f"  Lines: {len(text.splitlines())}")
        print(f"  First 300 chars:")
        print(f"  {text[:300]}")
        print()


# Usage:
# compare_parsers("your_document.pdf")
```

### What to Look For in the Comparison

| Signal                   | Good Parser               | Bad Parser                   |
| ------------------------ | ------------------------- | ---------------------------- |
| **Reading order**        | Text flows logically      | Columns mixed together       |
| **Tables**               | Rows/columns preserved    | Cells jumbled or missing     |
| **Headings**             | Hierarchy preserved       | Flat text, no structure      |
| **Lists**                | Items separated correctly | Items merged into paragraphs |
| **Special chars**        | Unicode handled properly  | Garbled characters           |
| **Page headers/footers** | Removed or tagged         | Mixed into body text         |

---

## Production Pipeline — Multi-Format Ingestion

```python
"""
Production-grade document ingestion pipeline.
Routes documents to the best parser based on file type and complexity.
"""
from pathlib import Path
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    content: str
    metadata: dict = field(default_factory=dict)
    source: str = ""
    parser_used: str = ""


class DocumentIngestionPipeline:
    """Route documents to the best parser based on type and quality needs."""

    def __init__(self, use_llm_parser: bool = False):
        self.use_llm_parser = use_llm_parser

    def ingest(self, file_path: str) -> ParsedDocument:
        path = Path(file_path)
        ext = path.suffix.lower()

        router = {
            ".pdf": self._parse_pdf,
            ".html": self._parse_html,
            ".htm": self._parse_html,
            ".docx": self._parse_docx,
            ".pptx": self._parse_pptx,
            ".md": self._parse_text,
            ".txt": self._parse_text,
            ".csv": self._parse_csv,
        }

        parser_fn = router.get(ext)
        if not parser_fn:
            raise ValueError(f"Unsupported file type: {ext}")

        logger.info(f"Parsing {path.name} with {parser_fn.__name__}")
        doc = parser_fn(str(path))
        doc.source = str(path)
        return doc

    def _parse_pdf(self, path: str) -> ParsedDocument:
        """PDF parsing with fallback chain:
        1. Try Docling (best structure preservation)
        2. Fall back to PyMuPDF (fastest, good for simple PDFs)
        3. Fall back to Unstructured (broadest compatibility)
        """
        # Try Docling first (best quality)
        try:
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            result = converter.convert(path)
            content = result.document.export_to_markdown()
            if len(content.strip()) > 50:
                return ParsedDocument(
                    content=content,
                    parser_used="docling",
                    metadata={"format": "markdown"},
                )
        except Exception as e:
            logger.warning(f"Docling failed: {e}")

        # Fallback: PyMuPDF
        try:
            import fitz
            doc = fitz.open(path)
            content = "\n".join(page.get_text() for page in doc)
            doc.close()
            if len(content.strip()) > 50:
                return ParsedDocument(
                    content=content,
                    parser_used="pymupdf",
                    metadata={"pages": len(doc)},
                )
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")

        # Final fallback: Unstructured
        from unstructured.partition.auto import partition
        elements = partition(path)
        content = "\n".join(el.text for el in elements if el.text)
        return ParsedDocument(
            content=content,
            parser_used="unstructured",
            metadata={"elements": len(elements)},
        )

    def _parse_html(self, path: str) -> ParsedDocument:
        from bs4 import BeautifulSoup
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return ParsedDocument(
            content=soup.get_text(separator="\n", strip=True),
            parser_used="beautifulsoup",
        )

    def _parse_docx(self, path: str) -> ParsedDocument:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(path)
        return ParsedDocument(
            content=result.text_content,
            parser_used="markitdown",
        )

    def _parse_pptx(self, path: str) -> ParsedDocument:
        from markitdown import MarkItDown
        md = MarkItDown()
        result = md.convert(path)
        return ParsedDocument(
            content=result.text_content,
            parser_used="markitdown",
        )

    def _parse_text(self, path: str) -> ParsedDocument:
        with open(path, "r", encoding="utf-8") as f:
            return ParsedDocument(content=f.read(), parser_used="plaintext")

    def _parse_csv(self, path: str) -> ParsedDocument:
        import csv
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        content = "\n".join(" | ".join(row) for row in rows)
        return ParsedDocument(content=content, parser_used="csv")


# Usage:
# pipeline = DocumentIngestionPipeline()
# doc = pipeline.ingest("quarterly_report.pdf")
# print(f"Parser: {doc.parser_used}, Length: {len(doc.content)}")
```

---

## LangChain Full Ingestion Pipeline

```python
"""
LangChain-based ingestion: load → split → embed → store.
This is the most common pattern you'll see in production.
"""
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    DirectoryLoader,
    UnstructuredFileLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load documents (pick one or combine)
# --- Single PDF ---
loader = PyMuPDFLoader("report.pdf")
docs = loader.load()

# --- Entire directory ---
loader = DirectoryLoader(
    "documents/",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True,
)
docs = loader.load()

# --- Multiple formats with Unstructured ---
loader = DirectoryLoader(
    "documents/",
    glob="**/*.*",
    loader_cls=UnstructuredFileLoader,
)
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,  # use tiktoken for token-based: see chunking cookbook
)
chunks = splitter.split_documents(docs)
print(f"Split {len(docs)} docs into {len(chunks)} chunks")

# 3. Embed and store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Save for later
vectorstore.save_local("faiss_index")

# 5. Search
results = vectorstore.similarity_search("What is the revenue?", k=5)
for r in results:
    print(f"[{r.metadata.get('source', '?')}] {r.page_content[:200]}")
```

---

## LlamaIndex Full Ingestion Pipeline

```python
"""
LlamaIndex-based ingestion: SimpleDirectoryReader → IngestionPipeline → VectorStoreIndex.
"""
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

# 1. Load documents
documents = SimpleDirectoryReader(
    input_dir="documents/",
    recursive=True,  # include subdirectories
    required_exts=[".pdf", ".docx", ".txt", ".md"],
).load_data()
print(f"Loaded {len(documents)} documents")

# 2. Ingestion pipeline (split + embed in one step)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=50),
        Settings.embed_model,
    ],
)
nodes = pipeline.run(documents=documents)
print(f"Created {len(nodes)} nodes")

# 3. Build index
index = VectorStoreIndex(nodes)

# 4. Query
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("What is the revenue?")
print(response)

# Show source documents
for node in response.source_nodes:
    print(f"  Score: {node.score:.3f} | {node.text[:200]}")
```

---

## Docling Full Pipeline (Structure-Aware)

```python
"""
Docling pipeline: preserves document structure (headings, tables, lists).
Best for scientific papers, technical documentation, reports with tables.
"""
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# 1. Convert document (preserves structure)
converter = DocumentConverter()
result = converter.convert("research_paper.pdf")

# 2. Access structured content
doc = result.document

# Full markdown export
markdown = doc.export_to_markdown()

# Access specific elements
for item in doc.iterate_items():
    print(f"[{item[0].label}] {item[0].text[:100]}")
    # Labels: title, section_header, paragraph, table, list_item, etc.

# 3. Structure-aware chunking (Docling's built-in chunker)
chunker = HybridChunker(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=512,
)
chunks = list(chunker.chunk(doc))

for chunk in chunks[:3]:
    print(f"Chunk: {chunk.text[:200]}")
    print(f"  Meta: {chunk.meta}")
    print()

# 4. Extract tables as DataFrames
for table in doc.tables:
    df = table.export_to_dataframe()
    print(f"Table ({df.shape[0]} rows x {df.shape[1]} cols):")
    print(df.head())
```

---

## Decision Framework — Which Parser to Use

```
START HERE
    │
    ▼
What is your document format?
    │
    ├── PDF ─────────────────────────────────────────────────────────┐
    │   │                                                           │
    │   Is it scanned / image-based?                                │
    │   │── YES → LlamaParse (best) or Unstructured (ocr_only)      │
    │   │── NO ↓                                                    │
    │   │                                                           │
    │   Does it have complex tables/figures/columns?                │
    │   │── YES → Docling (free) or LlamaParse (paid, best quality) │
    │   │── NO  → PyMuPDF (fastest) or pdfplumber (if some tables)  │
    │                                                               │
    ├── HTML → BeautifulSoup (manual) or LangChain WebBaseLoader    │
    │                                                               │
    ├── DOCX/PPTX → markitdown (Microsoft) or Docling              │
    │                                                               │
    ├── Multiple formats → Unstructured (broadest support)          │
    │                      or LangChain DirectoryLoader             │
    │                                                               │
    └── Already text/markdown → Just read it directly               │
```

---

## Common Pitfalls

| Pitfall                            | Impact                             | Fix                                          |
| ---------------------------------- | ---------------------------------- | -------------------------------------------- |
| Using PyMuPDF for complex layouts  | Tables and columns get garbled     | Switch to Docling or LlamaParse              |
| Not checking parser output quality | Silently bad extractions break RAG | Run the comparison script above on YOUR data |
| Ignoring page headers/footers      | Repetitive text pollutes chunks    | Post-process to remove repeated lines        |
| Parsing everything with LlamaParse | Expensive and slow for simple docs | Route simple docs to fast parsers (PyMuPDF)  |
| Not handling encoding issues       | Garbled characters in output       | Detect and fix encoding before parsing       |
| Skipping metadata extraction       | Can't filter or attribute sources  | Capture filename, page, section in metadata  |

---

## 📚 Additional Reading

- [Docling documentation](https://ds4sd.github.io/docling/) — IBM's open-source document parser
- [LlamaParse docs](https://docs.cloud.llamaindex.ai/) — LLM-powered parsing API
- [Unstructured docs](https://docs.unstructured.io/) — Multi-format document processing
- [markitdown](https://github.com/microsoft/markitdown) — Microsoft's Office-to-Markdown converter
- [LangChain Document Loaders](https://python.langchain.com/docs/integrations/document_loaders/) — 100+ loader integrations

---

## Syllabus Mapping

Maps to **§2.1** in `p2_rag_depth.md`. This cookbook complements the concept files (01–07) by showing how to use production libraries instead of writing parsers from scratch.
