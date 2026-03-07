# 2.2 Chunking (Critical Layer)

> Chunking decides the retrieval unit. Good chunking preserves meaning; bad chunking destroys it before retrieval even starts.

## 📌 Key Lesson

Chunking is not just about splitting text — it's about creating units that **align with how users ask questions**. A chunk that splits a concept across two pieces is worse than no chunking at all.

## Files

| File                                                                                                 | Topic                             | Key Concepts                                                                     |
| ---------------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------------------------------------- |
| [01_fixed_size_chunking.md](./01_fixed_size_chunking.md)                                             | Fixed-Size Chunking               | Character/token splitting, baseline chunking, when it works and fails            |
| [02_sliding_window_chunking.md](./02_sliding_window_chunking.md)                                     | Sliding Window                    | Overlap strategies, stride vs window size, deduplication with overlap            |
| [03_semantic_chunking.md](./03_semantic_chunking.md)                                                 | Semantic Chunking                 | Embedding-based boundary detection, similarity thresholds, breakpoint algorithms |
| [04_hierarchical_chunking.md](./04_hierarchical_chunking.md)                                         | Hierarchical Chunking             | Parent-child chunks, multi-level retrieval, parent document retriever pattern    |
| [05_chunk_size_overlap_tradeoffs.md](./05_chunk_size_overlap_tradeoffs.md)                           | Size & Overlap Tradeoffs          | Chunk size vs recall vs cost, overlap percentage impact, empirical tuning        |
| [06_chunk_query_alignment_and_structure_aware.md](./06_chunk_query_alignment_and_structure_aware.md) | Query Alignment & Structure-Aware | Headings, tables, code blocks, chunk-query mismatch failure modes                |

## Syllabus Mapping

Maps to **§2.2** in `p2_rag_depth.md` — covers fixed-size, sliding window, semantic, hierarchical chunking, overlap tradeoffs, chunk size vs recall vs cost, chunk-query alignment failure modes, and structure-aware chunking (headings, tables, code blocks).
