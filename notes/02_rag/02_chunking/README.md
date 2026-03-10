# 2.2 Chunking (Critical Layer)

> Chunking decides the retrieval unit. Good chunking preserves meaning; bad chunking destroys it before retrieval even starts.

## 📌 Key Lesson

Chunking is not just about splitting text — it's about creating units that **align with how users ask questions**. A chunk that splits a concept across two pieces is worse than no chunking at all.

## Learning Order

Start simple (fixed-size), understand why it fails, then learn smarter strategies:

```
Fixed-size (baseline) → Sliding window (fixes boundary loss)
→ Semantic (splits by meaning) → Hierarchical (multi-level)
→ Tradeoffs (how to tune) → Alignment & structure-aware (advanced)
```

## Files

| #   | File                                                                                                 | Topic                             | Key Concepts                                                                     |
| --- | ---------------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------- |
| 01  | [01_fixed_size_chunking.md](./01_fixed_size_chunking.md)                                             | Fixed-Size Chunking               | Character/token splitting, baseline chunking, when it works and fails            |
| 02  | [02_sliding_window_chunking.md](./02_sliding_window_chunking.md)                                     | Sliding Window                    | Overlap strategies, stride vs window size, deduplication with overlap            |
| 03  | [03_semantic_chunking.md](./03_semantic_chunking.md)                                                 | Semantic Chunking                 | Embedding-based boundary detection, similarity thresholds, breakpoint algorithms |
| 04  | [04_hierarchical_chunking.md](./04_hierarchical_chunking.md)                                         | Hierarchical Chunking             | Parent-child chunks, multi-level retrieval, parent document retriever pattern    |
| 05  | [05_chunk_size_overlap_tradeoffs.md](./05_chunk_size_overlap_tradeoffs.md)                           | Size & Overlap Tradeoffs          | Chunk size vs recall vs cost, overlap percentage impact, empirical tuning        |
| 06  | [06_chunk_query_alignment_and_structure_aware.md](./06_chunk_query_alignment_and_structure_aware.md) | Query Alignment & Structure-Aware | Headings, tables, code blocks, chunk-query mismatch failure modes                |
| 00  | [00_chunking_with_libraries.md](./00_chunking_with_libraries.md)                                     | 📚 Library Cookbook               | LangChain splitters, LlamaIndex parsers, Chonkie, Docling chunker                |

## Popular Libraries

| Library                 | Purpose                                  | Notes                                                          |
| ----------------------- | ---------------------------------------- | -------------------------------------------------------------- |
| LangChain TextSplitters | Multiple chunking strategies             | `RecursiveCharacterTextSplitter` is the default starting point |
| LlamaIndex NodeParser   | Document → Node chunking                 | Built-in sentence, token, semantic splitters                   |
| tiktoken                | Token counting for chunk sizing          | OpenAI tokenizer, use for accurate token-based chunks          |
| spaCy                   | Sentence detection for semantic chunking | More accurate sentence boundaries than regex                   |
| semchunk                | Semantic chunking library                | Standalone semantic chunking tool                              |

## Choosing a Strategy — Quick Guide

```
START HERE
    │
    ▼
Is your data well-structured (markdown, HTML, docs with clear headings)?
    │── YES → Use Structure-Aware chunking (respect headings/tables)
    │── NO  ↓
    │
Are your queries mostly fact-lookup ("What is X?")?
    │── YES → Fixed-size (256-512 tokens) with sentence boundaries
    │── NO  ↓
    │
Do queries need multi-paragraph context ("Explain how X works")?
    │── YES → Hierarchical chunking (retrieve leaf, expand to parent)
    │── NO  ↓
    │
Do your documents have clear topic shifts?
    │── YES → Semantic chunking (split at meaning boundaries)
    │── NO  → Fixed-size with sliding window overlap (safe default)
```

## Common Questions

### Q: What chunk size should I start with?

**A:** **256-512 tokens** with **50-100 token overlap** is the safe starting point for most use cases. Measure retrieval quality and adjust from there.

### Q: Why not just make chunks really small for precision?

**A:** Tiny chunks (< 100 tokens) lose context. "The algorithm is O(n log n)" means nothing without knowing WHICH algorithm. The LLM needs enough context in each chunk to understand the information.

### Q: How does chunking relate to retrieval?

**A:** Chunks are the **retrieval unit** — when you search, you get back chunks, not whole documents. Your chunking strategy directly determines what the retriever can and cannot find. Bad chunking → bad retrieval → bad answers, no matter how good your embedding model is.

### Q: Should I chunk before or after cleaning the data?

**A:** **After cleaning**, always. Clean first (data ingestion), then chunk. If you chunk noisy data, each chunk carries noise, and cleaning individual chunks is harder than cleaning whole documents.

### Q: Can I use different chunk sizes for different document types?

**A:** Yes, and you should if your data is diverse. FAQ docs might be best at 128 tokens, while technical documentation might need 512 tokens. Route by `doc_type` metadata.

## Syllabus Mapping

Maps to **§2.2** in `p2_rag_depth.md` — covers fixed-size, sliding window, semantic, hierarchical chunking, overlap tradeoffs, chunk size vs recall vs cost, chunk-query alignment failure modes, and structure-aware chunking (headings, tables, code blocks).
