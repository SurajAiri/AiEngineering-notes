# Phase 2 — RAG (Retrieval-Augmented Generation) In Depth

> Given documents and a query, reliably retrieve the right information and assemble it into a usable context.

RAG is split into two halves:

- **2(A) Building Real RAG Systems** — Can I build it?
- **2(B) Validating & Trusting RAG Systems** — Can I trust it?

---

## 📖 Table of Contents

### Phase 2(A) — Building RAG

| #   | Section                                                           | Files    | Focus                                                       |
| --- | ----------------------------------------------------------------- | -------- | ----------------------------------------------------------- |
| 01  | [Data Ingestion](./01_data_ingestion/)                            | 7 files  | Cleaning, deduplication, metadata, parsing                  |
| 02  | [Chunking](./02_chunking/)                                        | 6 files  | Fixed, sliding, semantic, hierarchical, structure-aware     |
| 03  | [Retrieval](./03_retrieval/)                                      | 11 files | Vector, BM25, hybrid, HyDE, adaptive, indexes, quantization |
| 04  | [Re-ranking & Context Assembly](./04_reranking_context_assembly/) | 3 files  | Cross-encoders, packing strategies, diversity, citations    |
| 05  | [Advanced RAG Architectures](./05_advanced_rag_architectures/)    | 4 files  | GraphRAG, Agentic RAG, Multimodal RAG, CRAG                 |
| 06  | [Failure Modes](./06_failure_modes/)                              | 1 file   | All 6 build-time failure modes with detection & prevention  |

### Phase 2(B) — Validating RAG

| #   | Section                                          | Files   | Focus                                                      |
| --- | ------------------------------------------------ | ------- | ---------------------------------------------------------- |
| 07  | [Evaluation](./07_evaluation/)                   | 3 files | Golden sets, retrieval metrics, faithfulness & correctness |
| 08  | [Production](./08_production/)                   | 3 files | Drift, cost/latency, observability & debugging             |
| 09  | [Advanced Embeddings](./09_advanced_embeddings/) | 1 file  | ColBERT, Matryoshka, fine-tuning, multi-vector             |

---

## 🗺️ Learning Path

```
START HERE
    │
    ▼
01 Data Ingestion ──── How does raw data become clean, indexed text?
    │
    ▼
02 Chunking ─────────── How do you split documents into retrievable units?
    │
    ▼
03 Retrieval ─────────── How do you find the right chunks for a query?
    │
    ▼
04 Re-ranking ────────── How do you pick the BEST chunks from candidates?
    │
    ▼
05 Advanced RAG ──────── GraphRAG, Agentic RAG, Multimodal, CRAG
    │
    ▼
06 Failure Modes ─────── What can go wrong and how to detect it?
    │
    ▼
07 Evaluation ──────────── How do you MEASURE if your RAG system works?
    │
    ▼
08 Production ──────────── How do you keep it working in production?
    │
    ▼
09 Advanced Embeddings ── ColBERT, Matryoshka, fine-tuning, multi-vector
```

---

## 📌 Key Insight

Most engineers stop at chunking + vector search. The difference between a demo and a production RAG system is everything from Section 04 onward: re-ranking, failure mode awareness, evaluation, and observability.

## Syllabus Reference

See [p2_rag_depth.md](../../syllabus/p2_rag_depth.md) for the detailed syllabus this maps to.
