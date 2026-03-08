# 2.12 Advanced Embedding Strategies

> ⚠️ **Advanced / Optional.** Start with standard embeddings. Only move to these when your evaluation shows embeddings are the bottleneck.

## 📌 Key Lesson

Start with a standard bi-encoder (`all-MiniLM-L6-v2` or `text-embedding-3-small`). Only move to advanced strategies when your evaluation shows embeddings are the bottleneck. Fine-tuning on domain data gives the biggest return. ColBERT gives the best quality ceiling.

## Files

| #   | File                                                                         | Topic                   | Key Concepts                                                                                                                                                                                                            |
| --- | ---------------------------------------------------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | [01_advanced_embedding_strategies.md](./01_advanced_embedding_strategies.md) | All Advanced Strategies | Late-interaction models (ColBERT/ColPali), Matryoshka embeddings (variable dimensions), domain-specific fine-tuning (contrastive learning, synthetic data), multi-vector representations (sentence-level, aspect-level) |

## Strategy Decision Guide

| Strategy              | Best For                    | Complexity | When to Use                          |
| --------------------- | --------------------------- | ---------- | ------------------------------------ |
| Bi-encoder (baseline) | Most use cases              | Low        | Always start here                    |
| Fine-tuning           | Domain-specific vocabulary  | Medium     | When domain jargon causes mismatches |
| Matryoshka            | Storage/speed flexibility   | Low        | When you need multi-tier search      |
| ColBERT               | Maximum retrieval quality   | High       | When bi-encoder quality isn't enough |
| Multi-vector          | Long, multi-topic documents | Medium     | When documents cover multiple topics |

## Popular Libraries

| Library               | Purpose                             | Notes                                   |
| --------------------- | ----------------------------------- | --------------------------------------- |
| sentence-transformers | Fine-tuning, pre-trained models     | Easiest way to fine-tune embeddings     |
| RAGatouille           | ColBERT for RAG                     | Simplified ColBERT usage for retrieval  |
| nomic-embed           | Open-source long-context embeddings | Good Matryoshka support                 |
| FlagEmbedding         | BGE family of models                | State-of-the-art open-source embeddings |

## Common Questions

### Q: When should I fine-tune embeddings?

**A:** When your domain has specialized vocabulary that off-the-shelf models don't understand well (medical, legal, internal product names). Run an evaluation first — if retrieval recall is low and the cause is vocabulary mismatch, fine-tuning helps. If recall is low because of bad chunking, fine-tuning won't help.

### Q: Is ColBERT worth the complexity?

**A:** For most production systems, no — a good bi-encoder with hybrid search and re-ranking gets you 90% of the quality. ColBERT shines when you need the absolute best retrieval quality and can afford the storage/compute overhead (token-level embeddings = ~100x more storage).

## Syllabus Mapping

Maps to **§2.12** in `p2_rag_depth.md` — covers late-interaction models (ColBERT, ColPali), Matryoshka embeddings, domain-specific embedding fine-tuning, and multi-vector representations.
