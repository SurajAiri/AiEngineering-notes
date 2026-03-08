# Phase 2 — RAG (Retrieval-Augmented Generation) In Depth

> Given documents and a query, reliably retrieve the right information and assemble it into a usable context.

RAG is split into two halves:

- **2(A) Building Real RAG Systems** — Can I build it?
- **2(B) Validating & Trusting RAG Systems** — Can I trust it?

---

## 📖 Table of Contents

### Prerequisites

| #   | Section                    | Files   | Focus                                                   |
| --- | -------------------------- | ------- | ------------------------------------------------------- |
| 00  | [Overview](./00_overview/) | 3 files | What is RAG, embedding basics, end-to-end pipeline demo |

> **New to RAG?** Start with Section 00. If you already know what RAG is and how embeddings work, skip to Section 01.

### Phase 2(A) — Building RAG

| #   | Section                                                           | Files    | Focus                                                      |
| --- | ----------------------------------------------------------------- | -------- | ---------------------------------------------------------- |
| 01  | [Data Ingestion](./01_data_ingestion/)                            | 7 files  | Cleaning, parsing, canonicalization, dedup, metadata       |
| 02  | [Chunking](./02_chunking/)                                        | 6 files  | Fixed, sliding, semantic, hierarchical, structure-aware    |
| 03  | [Retrieval](./03_retrieval/)                                      | 11 files | Vector, BM25, hybrid, query rewriting, HyDE, indexes       |
| 04  | [Re-ranking & Context Assembly](./04_reranking_context_assembly/) | 4 files  | Cross-encoders, packing, diversity, prompt design for RAG  |
| 05  | [Advanced RAG Architectures](./05_advanced_rag_architectures/)    | 4 files  | ⚠️ GraphRAG, Agentic RAG, Multimodal RAG, CRAG             |
| 06  | [Failure Modes](./06_failure_modes/)                              | 1 file   | All 6 build-time failure modes with detection & prevention |

### Phase 2(B) — Validating RAG

| #   | Section                                          | Files   | Focus                                                      |
| --- | ------------------------------------------------ | ------- | ---------------------------------------------------------- |
| 07  | [Evaluation](./07_evaluation/)                   | 3 files | Golden sets, retrieval metrics, faithfulness & correctness |
| 08  | [Production](./08_production/)                   | 3 files | Drift, cost/latency, observability & debugging             |
| 09  | [Advanced Embeddings](./09_advanced_embeddings/) | 1 file  | ⚠️ ColBERT, Matryoshka, fine-tuning, multi-vector          |

> ⚠️ = Advanced/Optional — good to read for depth, but not required for building your first production RAG system.

---

## 🗺️ Learning Path

```
START HERE
    │
    ▼
00 Overview ──────────── What is RAG? Embeddings? End-to-end demo
    │
    ▼
01 Data Ingestion ──── How does raw data become clean, indexed text?
    │                   (Parse → Clean → Normalize → Deduplicate → Tag)
    ▼
02 Chunking ─────────── How do you split documents into retrievable units?
    │
    ▼
03 Retrieval ─────────── How do you find the right chunks for a query?
    │                   (Vector → BM25 → Hybrid → Query Rewriting)
    ▼
04 Re-ranking ────────── How do you pick the BEST chunks from candidates?
    │                   (Cross-encoder → Pack context → Prompt design)
    ▼
06 Failure Modes ─────── What can go wrong and how to detect it?
    │
    ▼
07 Evaluation ──────────── How do you MEASURE if your RAG system works?
    │                      (RAGAS, retrieval metrics, faithfulness)
    ▼
08 Production ──────────── How do you keep it working in production?
    │                      (Drift, cost, observability)
    ▼
┌──────────────────────────── OPTIONAL / ADVANCED ────────────────────────────┐
│ 05 Advanced RAG ──────── GraphRAG, Agentic RAG, Multimodal, CRAG           │
│ 09 Advanced Embeddings ── ColBERT, Matryoshka, fine-tuning                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Key Libraries Used Throughout

| Library               | Category     | Purpose                             |
| --------------------- | ------------ | ----------------------------------- |
| LangChain             | Framework    | Most popular RAG framework          |
| LlamaIndex            | Framework    | Data framework for LLM applications |
| sentence-transformers | Embeddings   | Open-source embedding models        |
| FAISS                 | Vector Store | Fast local similarity search        |
| ChromaDB              | Vector Store | Simple getting-started vector DB    |
| Pinecone / Qdrant     | Vector Store | Managed production vector databases |
| RAGAS                 | Evaluation   | RAG evaluation metrics + test gen   |
| Cohere Rerank         | Reranking    | High-quality reranking API          |
| Unstructured          | Parsing      | Multi-format document parsing       |

---

## 📌 Key Insight

Most engineers stop at chunking + vector search. The difference between a demo and a production RAG system is everything from Section 04 onward: re-ranking, failure mode awareness, evaluation, and observability.

Every section includes **popular library references** and **common Q&A** to help you build practical intuition alongside the concepts.

## Syllabus Reference

See [p2_rag_depth.md](../../syllabus/p2_rag_depth.md) for the detailed syllabus this maps to.
