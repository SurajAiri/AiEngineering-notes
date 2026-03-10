# 📚 Phase 0.5: ML Foundations

## Overview

Essential ML concepts for AI engineering. These foundations help you understand how LLMs and embeddings work under the hood.

---

## 📖 Notes Index

| #   | Topic                                                              | Description                                                     |
| --- | ------------------------------------------------------------------ | --------------------------------------------------------------- |
| 00  | [Transformer Architecture](./00_transformer_architecture.md)       | Self-attention, positional encodings, efficiency                |
| 01  | [Math Foundations](./01_math_foundations.md)                       | Linear algebra, probability, loss functions, similarity metrics |
| 02  | [Optimization & Gradient Flow](./02_optimization_gradient_flow.md) | SGD→AdamW, LR schedules, LoRA/QLoRA                             |

---

## 🔗 Key Concepts

### What You Need to Know

| Concept                 | Why It Matters for AI Engineering             |
| ----------------------- | --------------------------------------------- |
| **Attention**           | Understand context limits, prompt engineering |
| **Tokenization**        | Token counting, cost estimation, chunking     |
| **Embeddings**          | Semantic search, RAG retrieval                |
| **Positional Encoding** | Context window limitations                    |
| **Quantization**        | Model deployment, inference optimization      |

### What You Can Skip (for now)

| Concept                 | When to Learn               |
| ----------------------- | --------------------------- |
| Backpropagation details | If fine-tuning from scratch |
| Loss function math      | If training custom models   |
| Optimizer internals     | Advanced fine-tuning        |

---

## 📊 Relevance to Other Phases

```
┌─────────────────────────────────────────────────────────────────┐
│                 PHASE 0.5 CONNECTIONS                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Transformers → Phase 1 (LLM Fundamentals)                     │
│     - Context windows                                           │
│     - Token limits                                              │
│     - Model selection                                           │
│                                                                  │
│   Embeddings → Phase 2 (RAG)                                    │
│     - Vector similarity                                         │
│     - Semantic search                                           │
│     - Index types                                               │
│                                                                  │
│   Efficiency → Production Deployment                            │
│     - Quantization                                              │
│     - KV-cache                                                  │
│     - Batch inference                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ➡️ Next Steps

Continue to **Phase 1** for LLM Fundamentals.
