# Chunking with Libraries — Practical Cookbook

## 🟢 How to Approach This Topic

> **Why this matters for your job:** You'll rarely write chunking logic from scratch in production. Libraries like LangChain, LlamaIndex, Docling, and Chonkie provide battle-tested splitters. But you need to understand what they do internally (see files 01–06) so you can pick the right one and tune it properly.

**Prerequisites:** Read [01_fixed_size_chunking.md](./01_fixed_size_chunking.md) to understand the baseline concept.

**Reading order:**

1. Library comparison table below (5 min)
2. Try the LangChain examples — most common in industry (15 min)
3. Try LlamaIndex equivalents (15 min)
4. Run the evaluation script on your data (30 min)
5. Read decision framework (5 min)

**⏱️ Core concept: 45 min | Full exploration: 2 hours**

---

## Library Landscape

```
                    ┌──────────────────────────────┐
                    │     Chunking Strategies       │
                    └──────────────┬───────────────┘
                                   │
        ┌──────────┬───────────────┼───────────────┬──────────┐
        │          │               │               │          │
   ┌────▼───┐ ┌───▼────┐   ┌──────▼──────┐  ┌─────▼────┐ ┌──▼───────┐
   │Fixed   │ │Sliding │   │ Semantic    │  │Hierarchi-│ │Structure │
   │Size    │ │Window  │   │ (meaning-   │  │cal       │ │Aware     │
   │        │ │        │   │  based)     │  │(parent/  │ │(tables,  │
   │        │ │        │   │            │  │ child)   │ │ code)    │
   └────────┘ └────────┘   └─────────────┘  └──────────┘ └──────────┘
       │          │               │               │           │
   LangChain  LangChain     LangChain       LlamaIndex    Docling
   LlamaIndex LlamaIndex  LlamaIndex        (auto-merge)  Unstructured
   Chonkie    Chonkie     Chonkie           LangChain     Chonkie
```

| Library                     | Strategy Support                                 | Speed          | Token-Aware            | Notes                                         |
| --------------------------- | ------------------------------------------------ | -------------- | ---------------------- | --------------------------------------------- |
| **LangChain TextSplitters** | Fixed, recursive, semantic, HTML, Markdown, code | ⚡ Fast        | ✅ With tiktoken       | Most popular, widest adoption                 |
| **LlamaIndex NodeParsers**  | Sentence, token, semantic, hierarchical          | ⚡ Fast        | ✅ Built-in            | Tight integration with LlamaIndex index/query |
| **Chonkie**                 | Token, word, sentence, semantic, SDPM            | ⚡⚡ Very fast | ✅ Multiple tokenizers | Modern, fast alternative. Great API           |
| **Docling Chunker**         | Hybrid (structure + token-aware)                 | 🔄 Medium      | ✅ Configurable        | Best for preserving document structure        |
| **Unstructured**            | By element type (title, paragraph, table)        | 🔄 Medium      | ❌ Element-based       | Structure-aware by default                    |

---

## LangChain — Chunking Examples

### RecursiveCharacterTextSplitter (The Default Starting Point)

```python
"""
The most commonly used splitter in production.
Tries to split by: \n\n → \n → " " → "" (paragraphs → sentences → words → chars).
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
# Introduction

Machine learning is a subset of artificial intelligence that focuses on
building systems that learn from data. Unlike traditional programming where
rules are explicitly coded, ML systems discover patterns automatically.

## Types of ML

There are three main types:
1. Supervised learning - learns from labeled examples
2. Unsupervised learning - finds patterns without labels
3. Reinforcement learning - learns through trial and error

## Deep Learning

Deep learning uses neural networks with many layers. These networks can
learn hierarchical representations of data, making them powerful for tasks
like image recognition and natural language processing.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", " ", ""],  # default hierarchy
)
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i} ({len(chunk)} chars): {chunk[:80]}...")
```

### Token-Based Splitting (Production Standard)

```python
"""
Use token count instead of character count.
Critical for staying within LLM context limits.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Token-aware splitting with tiktoken
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",       # or "gpt-3.5-turbo"
    chunk_size=512,            # in tokens, not characters
    chunk_overlap=50,          # overlap in tokens
)
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks")
```

### Markdown-Aware Splitting

```python
"""
Respects markdown structure (headings, code blocks, lists).
Great for documentation and technical content.
"""
from langchain_text_splitters import MarkdownHeaderTextSplitter

markdown_text = """
# Chapter 1: Getting Started

This chapter covers the basics.

## Installation

Run pip install mypackage.

## Configuration

Set the API_KEY environment variable.

# Chapter 2: Advanced Usage

This chapter covers advanced features.

## Custom Models

You can bring your own models.
"""

# Split by headers — each chunk gets header metadata
headers = [
    ("#", "h1"),
    ("##", "h2"),
]
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown_text)

for chunk in chunks:
    print(f"Headers: {chunk.metadata}")
    print(f"Content: {chunk.page_content[:100]}")
    print()

# Output:
# Headers: {'h1': 'Chapter 1: Getting Started', 'h2': 'Installation'}
# Content: Run pip install mypackage.
```

### Semantic Chunking (LangChain Experimental)

```python
"""
Splits text at semantic boundary changes using embeddings.
Groups semantically similar sentences together.
"""
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Split when semantic similarity drops significantly
chunker = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=70,          # 70th percentile = split
)
chunks = chunker.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Semantic chunk {i}: {chunk[:100]}...")
```

### Code-Aware Splitting

```python
"""
Splits code files while respecting function/class boundaries.
"""
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
)

python_code = '''
def calculate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts).tolist()


class VectorStore:
    """Simple vector store implementation."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.vectors = []
        self.documents = []

    def add(self, vector, document):
        self.vectors.append(vector)
        self.documents.append(document)

    def search(self, query_vector, k=5):
        # Cosine similarity search
        similarities = [cosine_sim(query_vector, v) for v in self.vectors]
        top_k = sorted(range(len(similarities)),
                       key=lambda i: similarities[i], reverse=True)[:k]
        return [(self.documents[i], similarities[i]) for i in top_k]
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=30,
)
chunks = splitter.split_text(python_code)

for i, chunk in enumerate(chunks):
    print(f"Code chunk {i}:\n{chunk}\n{'---'*10}")
```

---

## LlamaIndex — Chunking Examples

### SentenceSplitter (Default)

```python
"""
LlamaIndex's default splitter. Splits on sentence boundaries.
Equivalent to LangChain's RecursiveCharacterTextSplitter but sentence-aware.
"""
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(
    chunk_size=512,           # in tokens
    chunk_overlap=50,
    paragraph_separator="\n\n",
)

# From raw text
from llama_index.core import Document
doc = Document(text=text)
nodes = splitter.get_nodes_from_documents([doc])

for node in nodes:
    print(f"Node ({len(node.text)} chars): {node.text[:100]}...")
    print(f"  Metadata: {node.metadata}")
```

### SemanticSplitterNodeParser

```python
"""
LlamaIndex's semantic chunking — groups sentences by embedding similarity.
"""
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(model="text-embedding-3-small")

splitter = SemanticSplitterNodeParser(
    embed_model=embed_model,
    breakpoint_percentile_threshold=70,
    buffer_size=1,  # sentences to group for comparison
)

nodes = splitter.get_nodes_from_documents([Document(text=text)])
for node in nodes:
    print(f"Semantic node: {node.text[:100]}...")
```

### Hierarchical Chunking (Auto-Merging Retriever)

```python
"""
LlamaIndex's parent-child chunking pattern.
Retrieve specific chunks, but expand to parent if needed.
"""
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.core import Document

# Create multi-level hierarchy
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],  # large → medium → small
)

doc = Document(text=text)
nodes = node_parser.get_nodes_from_documents([doc])
leaf_nodes = get_leaf_nodes(nodes)

print(f"Total nodes: {len(nodes)}")
print(f"Leaf nodes: {len(leaf_nodes)}")

# In retrieval, use AutoMergingRetriever to expand
# small matches to their parent context:
#
# from llama_index.core.retrievers import AutoMergingRetriever
# retriever = AutoMergingRetriever(
#     vector_retriever,
#     storage_context,
#     simple_ratio_thresh=0.4,  # merge if 40%+ children match
# )
```

---

## Chonkie — Modern Alternative

```python
"""
Chonkie: fast, modern chunking library. Clean API, multiple strategies.
pip install chonkie[all]
"""
from chonkie import TokenChunker, SemanticChunker, SDPMChunker

# Token-based chunking
chunker = TokenChunker(
    tokenizer="gpt2",     # or "cl100k_base" for GPT-4
    chunk_size=512,
    chunk_overlap=50,
)
chunks = chunker.chunk(text)
for chunk in chunks:
    print(f"Tokens: {chunk.token_count}, Text: {chunk.text[:80]}...")

# Semantic chunking
chunker = SemanticChunker(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=512,
    similarity_threshold=0.5,
)
chunks = chunker.chunk(text)
for chunk in chunks:
    print(f"Semantic chunk: {chunk.text[:80]}...")

# SDPM (Semantic Double-Pass Merge) — best quality, slower
chunker = SDPMChunker(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=512,
    similarity_threshold=0.6,
)
chunks = chunker.chunk(text)
```

---

## Docling — Structure-Aware Chunking

```python
"""
Docling preserves document structure through chunking.
Tables, headings, lists stay intact as chunk boundaries.
"""
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

# 1. Parse the document (structure-aware)
converter = DocumentConverter()
result = converter.convert("technical_doc.pdf")

# 2. Chunk with structure awareness
chunker = HybridChunker(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=512,
    merge_peers=True,  # merge small adjacent chunks
)
chunks = list(chunker.chunk(result.document))

for chunk in chunks[:5]:
    print(f"Chunk ({len(chunk.text)} chars):")
    print(f"  Text: {chunk.text[:150]}...")
    print(f"  Meta: {chunk.meta}")
    # Meta includes: doc_items (which structural elements are in this chunk),
    # headings (section hierarchy), etc.
    print()
```

---

## Evaluating Chunking Quality

The most important practice: **measure which chunking strategy works best for YOUR data and queries.**

```python
"""
Compare chunking strategies by measuring retrieval quality.
Run this before committing to a chunking approach.
"""
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_chunking(chunks: list[str], queries: list[str],
                      expected_chunks: list[int]) -> dict:
    """
    Measure how well chunking supports retrieval.

    Args:
        chunks: list of chunk texts
        queries: test queries
        expected_chunks: index of the chunk that should be retrieved for each query
    """
    chunk_embeddings = model.encode(chunks)
    query_embeddings = model.encode(queries)

    hits_at_1 = 0
    hits_at_3 = 0
    mrr = 0.0

    for i, (q_emb, expected) in enumerate(zip(query_embeddings, expected_chunks)):
        # Compute similarities
        sims = np.dot(chunk_embeddings, q_emb)
        ranked = np.argsort(sims)[::-1]

        # Check metrics
        rank = np.where(ranked == expected)[0][0] + 1
        mrr += 1.0 / rank
        if rank == 1:
            hits_at_1 += 1
        if rank <= 3:
            hits_at_3 += 1

    n = len(queries)
    return {
        "hits@1": hits_at_1 / n,
        "hits@3": hits_at_3 / n,
        "mrr": mrr / n,
        "num_chunks": len(chunks),
        "avg_chunk_size": np.mean([len(c) for c in chunks]),
    }


# Example usage:
# results_fixed = evaluate_chunking(fixed_chunks, queries, expected)
# results_semantic = evaluate_chunking(semantic_chunks, queries, expected)
# print(f"Fixed:    {results_fixed}")
# print(f"Semantic: {results_semantic}")
```

---

## Decision Framework

```
Which chunking library should I use?

    Already using LangChain?
    │── YES → RecursiveCharacterTextSplitter (with tiktoken)
    │         Need semantic? → SemanticChunker (experimental)
    │         Have markdown? → MarkdownHeaderTextSplitter
    │── NO ↓
    │
    Already using LlamaIndex?
    │── YES → SentenceSplitter (default)
    │         Need hierarchical? → HierarchicalNodeParser + AutoMergingRetriever
    │         Need semantic? → SemanticSplitterNodeParser
    │── NO ↓
    │
    Have structured documents (scientific papers, reports)?
    │── YES → Docling HybridChunker (best structure preservation)
    │── NO ↓
    │
    Need maximum speed?
    │── YES → Chonkie TokenChunker
    │── NO ↓
    │
    Default → LangChain RecursiveCharacterTextSplitter (most examples/tutorials)
```

---

## Common Pitfalls

| Pitfall                                      | Impact                                            | Fix                                                     |
| -------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------- |
| Using character count instead of token count | Tokens ≠ characters; you'll exceed context limits | Use `.from_tiktoken_encoder()` or token-based splitters |
| Not measuring chunking quality               | You don't know if your strategy works             | Run the evaluation script above                         |
| Same chunk size for all document types       | FAQs need small chunks, manuals need large        | Route by doc_type, use different sizes                  |
| Ignoring overlap                             | Losing context at boundaries                      | Start with 10-15% overlap (50-75 tokens for 512 chunks) |
| Chunking before cleaning                     | Noise propagates into every chunk                 | Always: clean → chunk → embed                           |
| Semantic chunking on short docs              | Overhead isn't worth it for <2 pages              | Use fixed-size for short documents                      |

---

## Syllabus Mapping

Maps to **§2.2** in `p2_rag_depth.md`. This cookbook complements the concept files (01–06) by showing production library usage for each strategy.
