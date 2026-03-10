# 📐 Math Foundations for AI Engineering

## 📚 Overview

You don't need a PhD in math to be an AI engineer, but you DO need to understand the math that shows up daily: why cosine similarity works for embeddings, what cross-entropy loss actually measures, how matrix multiplication enables attention, and why KL divergence matters for model distillation. This note covers the math you'll actually use — with intuition first, formulas second, code always.

> 📌 _Know the math well enough to debug — not well enough to derive from scratch. If cosine similarity returns 0.99 for unrelated texts, you need to know why._

---

## 🎯 Learning Objectives

- **Compute** and interpret similarity metrics (cosine, dot product, L2) with intuition
- **Understand** matrix operations that power attention (QKV projections, softmax)
- **Apply** probability concepts (Bayes, conditional probability) to LLM outputs
- **Interpret** loss functions (cross-entropy, KL divergence) for fine-tuning decisions
- **Connect** SVD/PCA to dimensionality reduction in embeddings (Matryoshka, compression)

---

## 🧠 Sections (To Be Written)

### 1. Linear Algebra for Transformers

- Vectors and embeddings (geometric intuition)
- Matrix multiplication as learned projections (Q, K, V)
- Dot product as similarity (why attention uses it)
- Softmax as probability distribution
- Eigenvalues/SVD — intuition for PCA, LoRA rank

### 2. Similarity Metrics (Deep Dive)

- Cosine similarity: formula, geometric meaning, failure cases
- L2 (Euclidean) distance: when to use, normalization effects
- Dot product: relationship to cosine, scale sensitivity
- Implementation in NumPy + comparison table
- When each metric fails (and why normalized embeddings help)

### 3. Probability for LLMs

- Probability distributions over vocabulary (softmax output)
- Conditional probability (next token prediction)
- Bayes' theorem (relevance to retrieval scoring)
- Log probabilities and perplexity (model confidence)
- Calibration: when model confidence ≠ correctness

### 4. Loss Functions & Information Theory

- Cross-entropy loss: what it measures, intuition
- KL divergence: measuring distribution difference (distillation)
- Information theory basics: entropy, mutual information
- Connecting loss to model behavior (why lower loss ≠ better generation)

### 5. Dimensionality Reduction

- PCA: intuition for embedding compression
- SVD: connection to LoRA (low-rank approximation)
- t-SNE/UMAP: visualization of embedding spaces
- Matryoshka embeddings: mathematical basis

### 6. Common Pitfalls

| Symptom                                    | Cause                      | Fix                               |
| ------------------------------------------ | -------------------------- | --------------------------------- |
| High cosine similarity for unrelated texts | Un-normalized embeddings   | Normalize before comparison       |
| Perplexity looks good but outputs bad      | Perplexity ≠ quality       | Use task-specific evaluation      |
| LoRA rank too low                          | Underfitting               | Increase rank, check SVD spectrum |
| PCA loses semantic meaning                 | Linear assumption violated | Use learned compression instead   |

---

## 📖 Resources

- 3Blue1Brown — Essence of Linear Algebra (video series)
- Lilian Weng — "From Autoencoder to Beta-VAE" (for KL divergence)
- Jay Alammar — "The Illustrated Transformer"

---

## ➡️ Next Steps

Continue to [Optimization & Gradient Flow](./02_optimization_gradient_flow.md) →
