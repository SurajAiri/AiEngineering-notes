# 10 — Capstone: End-to-End Production RAG

## Purpose

This section ties everything together. You've learned individual components — now build **complete, production-grade RAG systems** using popular frameworks.

## What's Here

| File                                   | What You'll Build                | Framework        | Time      |
| -------------------------------------- | -------------------------------- | ---------------- | --------- |
| `01_production_rag_with_llamaindex.md` | Complete RAG system from scratch | LlamaIndex       | 2–3 hours |
| `02_production_rag_with_langchain.md`  | Complete RAG system from scratch | LangChain (LCEL) | 2–3 hours |

## Learning Order

```
┌──────────────────────────────────────────────────┐
│         Pick ONE framework first                  │
│                                                    │
│    ┌──────────────┐     ┌──────────────────┐      │
│    │  LlamaIndex  │ OR  │   LangChain      │      │
│    │  (simpler    │     │   (more flexible, │      │
│    │   for RAG)   │     │    bigger         │      │
│    │              │     │    ecosystem)     │      │
│    └──────────────┘     └──────────────────┘      │
│                                                    │
│    Then build the other for comparison.            │
│    In job interviews, knowing BOTH is ideal.       │
└──────────────────────────────────────────────────┘
```

## Prerequisites

Before starting these capstones, you should be comfortable with:

- ✅ Chunking strategies (section 02)
- ✅ Retrieval and hybrid search (section 03)
- ✅ Reranking (section 04)
- ✅ Evaluation with RAGAS (section 07)
- ✅ At least read through the library cookbooks (00\_ files in each section)

## What You'll Learn

Each capstone covers the **full pipeline** — not just retrieval, but:

1. Document loading and parsing
2. Chunking
3. Embedding and indexing
4. Retrieval (hybrid)
5. Reranking
6. Generation with proper prompt design
7. Evaluation with RAGAS
8. Observability with tracing
9. API deployment

This mirrors what you'll build at work.
