# 2.12 Advanced Embedding Strategies

> Advanced embeddings unlock better retrieval quality, flexible storage, and domain-specific precision — but start simple.

## 📌 Key Lesson

Start with a standard bi-encoder (`all-MiniLM-L6-v2` or `text-embedding-3-small`). Only move to advanced strategies when your evaluation shows embeddings are the bottleneck. Fine-tuning on domain data gives the biggest return. ColBERT gives the best quality ceiling.

## Files

| File                                                                         | Topic                   | Key Concepts                                                                                                                                                                                                            |
| ---------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [01_advanced_embedding_strategies.md](./01_advanced_embedding_strategies.md) | All Advanced Strategies | Late-interaction models (ColBERT/ColPali), Matryoshka embeddings (variable dimensions), domain-specific fine-tuning (contrastive learning, synthetic data), multi-vector representations (sentence-level, aspect-level) |

## Strategy Decision Guide

| Strategy              | Best For                    | Complexity |
| --------------------- | --------------------------- | ---------- |
| Bi-encoder (baseline) | Most use cases              | Low        |
| Fine-tuning           | Domain-specific vocabulary  | Medium     |
| Matryoshka            | Storage/speed flexibility   | Low        |
| ColBERT               | Maximum retrieval quality   | High       |
| Multi-vector          | Long, multi-topic documents | Medium     |

## Syllabus Mapping

Maps to **§2.12** in `p2_rag_depth.md` — covers late-interaction models (ColBERT, ColPali), Matryoshka embeddings, domain-specific embedding fine-tuning, and multi-vector representations.
