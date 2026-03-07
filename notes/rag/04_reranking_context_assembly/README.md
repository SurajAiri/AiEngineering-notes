# 2.4 Re-ranking & Context Assembly

> This section is about what text enters the model context and in what order — not about prompting theory.

## 📌 Key Lesson

The gap between retrieval results and LLM context is where most production quality is won or lost. Cross-encoder re-ranking, context packing strategy, diversity management, and citation alignment turn noisy retrieval into reliable answers.

## Files

| File                                                                                 | Topic                                    | Key Concepts                                                                                                                            |
| ------------------------------------------------------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| [01_cross_encoder_reranking.md](./01_cross_encoder_reranking.md)                     | Cross-Encoder Re-ranking                 | Bi-encoder vs cross-encoder, production pipeline, model comparison, batching                                                            |
| [02_score_normalization_dedup_packing.md](./02_score_normalization_dedup_packing.md) | Score Normalization & Dedup              | Min-max/sigmoid normalization, context deduplication (MMR), token budget allocation                                                     |
| [03_diversity_ordering_citations.md](./03_diversity_ordering_citations.md)           | Diversity, Packing, Ordering & Citations | MMR, breadth-first vs depth-first packing, diversity vs relevance tradeoff, λ tuning, citation alignment, "lost in the middle" ordering |

## Syllabus Mapping

Maps to **§2.4** in `p2_rag_depth.md` — covers cross-encoder re-ranking, score normalization, context deduplication, token budget allocation, citation alignment, context packing strategies (breadth-first vs depth-first), diversity vs relevance tradeoffs, and ordering context for answerability.
