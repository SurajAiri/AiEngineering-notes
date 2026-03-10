# 2.4 Re-ranking & Context Assembly

> This section is about what text enters the model context and in what order — not about prompting theory.

## 📌 Key Lesson

The gap between retrieval results and LLM context is where most production quality is won or lost. Cross-encoder re-ranking, context packing strategy, diversity management, and citation alignment turn noisy retrieval into reliable answers.

## Learning Order

```
Cross-encoder re-ranking (score chunks better)
→ Score normalization & dedup (clean up scores, remove dupes)
→ Diversity, ordering & citations (assemble final context for LLM)
```

## Files

| #   | File                                                                                 | Topic                                    | Key Concepts                                                                                                                            |
| --- | ------------------------------------------------------------------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | [01_cross_encoder_reranking.md](./01_cross_encoder_reranking.md)                     | Cross-Encoder Re-ranking                 | Bi-encoder vs cross-encoder, production pipeline, model comparison, batching                                                            |
| 02  | [02_score_normalization_dedup_packing.md](./02_score_normalization_dedup_packing.md) | Score Normalization & Dedup              | Min-max/sigmoid normalization, context deduplication (MMR), token budget allocation                                                     |
| 03  | [03_diversity_ordering_citations.md](./03_diversity_ordering_citations.md)           | Diversity, Packing, Ordering & Citations | MMR, breadth-first vs depth-first packing, diversity vs relevance tradeoff, λ tuning, citation alignment, "lost in the middle" ordering |
| 04  | [04_prompt_design_for_rag.md](./04_prompt_design_for_rag.md)                         | Prompt Design for RAG                    | Context formatting, citation instructions, refusal behavior, system prompts for RAG                                                     |
| 00  | [00_reranking_with_libraries.md](./00_reranking_with_libraries.md)                   | 📚 Library Cookbook                      | FlashRank, Cohere Rerank, LangChain compression, LlamaIndex rerankers                                                                   |

## Popular Libraries

| Library               | Purpose                        | Notes                                                           |
| --------------------- | ------------------------------ | --------------------------------------------------------------- |
| sentence-transformers | Cross-encoder models           | `cross-encoder/ms-marco-MiniLM-L-6-v2` is a good starting model |
| Cohere Rerank         | Managed re-ranking API         | `rerank-v3` — highest quality, paid                             |
| FlashRank             | Fast lightweight re-ranking    | Good for latency-sensitive applications                         |
| LangChain             | ContextualCompressionRetriever | Wraps retrievers with re-ranking built in                       |
| LlamaIndex            | SentenceTransformerRerank      | Node postprocessor for easy integration                         |

## Common Questions

### Q: Is re-ranking necessary? Can't I just use retrieval scores?

**A:** Retrieval scores (from vector search) are from a **bi-encoder** — fast but imprecise. A cross-encoder re-ranker reads the query AND each chunk together, producing much better relevance scores. For production RAG, re-ranking typically improves answer quality by 10-25%.

### Q: Doesn't re-ranking add latency?

**A:** Yes, ~50-200ms depending on the model and number of chunks. That's why you retrieve many (k=20-50 with fast vector search) and re-rank to pick the top 5-10. The latency is worth the quality improvement in most cases.

### Q: What is "lost in the middle"?

**A:** Research shows LLMs pay more attention to context at the **beginning and end** of the prompt, and less to text in the middle. When ordering your chunks in the prompt, put the most relevant ones first and last, not buried in the middle.

### Q: How many chunks should I send to the LLM?

**A:** Depends on your token budget, but **3-7 chunks** is typical. More chunks = more context = higher cost + potential confusion. Use a token budget (e.g., 2000-4000 tokens for context) and pack the most relevant chunks that fit.

## Syllabus Mapping

Maps to **§2.4** in `p2_rag_depth.md` — covers cross-encoder re-ranking, score normalization, context deduplication, token budget allocation, citation alignment, context packing strategies (breadth-first vs depth-first), diversity vs relevance tradeoffs, and ordering context for answerability.
