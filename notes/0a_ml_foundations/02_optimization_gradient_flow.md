# 📉 Optimization & Gradient Flow

## 📚 Overview

Understanding optimization is the bridge between "I use LLMs via API" and "I can fine-tune, debug training, and understand why LoRA works." This note covers the optimization concepts an AI engineer encounters: from AdamW (the optimizer behind every LLM) to learning rate schedules, gradient clipping, and the mathematical intuition behind LoRA/QLoRA — why low-rank decomposition works for efficient fine-tuning.

> 📌 _You don't train foundation models. But you fine-tune, and when fine-tuning fails, understanding optimization is the difference between debugging and guessing._

---

## 🎯 Learning Objectives

- **Trace** the evolution from SGD → Adam → AdamW and why each step matters
- **Design** learning rate schedules (warmup, cosine decay) for fine-tuning
- **Understand** gradient clipping and why it prevents training instability
- **Explain** LoRA's math — why low-rank weight updates work (SVD connection)
- **Configure** QLoRA: quantization + LoRA for memory-efficient fine-tuning
- **Debug** common training failures (loss plateau, divergence, catastrophic forgetting)

---

## 🧠 Sections (To Be Written)

### 1. Gradient Descent Family

- SGD: the foundation (intuition + code)
- Momentum: accelerating convergence
- Adam: adaptive learning rates per parameter
- AdamW: weight decay done right (why it matters for LLMs)
- Implementation comparison in PyTorch

### 2. Learning Rate Schedules

- Constant vs decay vs warmup
- Linear warmup + cosine decay (the standard for fine-tuning)
- Warmup rationale (why cold starts cause instability)
- OneCycleLR and other schedules
- Choosing the right LR for fine-tuning (rule of thumb: 1e-5 to 5e-5)

### 3. Gradient Management

- Gradient clipping (max_norm): when and why
- Gradient accumulation: simulating larger batch sizes
- Mixed precision training (FP16/BF16): speed vs stability
- Gradient checkpointing: trading compute for memory

### 4. LoRA: Low-Rank Adaptation

- The insight: weight changes during fine-tuning are low-rank
- SVD decomposition of weight matrices
- LoRA implementation: $W' = W + BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$
- Rank selection (r=4, 8, 16, 32 — tradeoffs)
- Which layers to adapt (attention vs MLP)
- Parameter efficiency calculation

### 5. QLoRA: Quantized LoRA

- 4-bit quantization (NF4)
- Double quantization
- Paged optimizers
- Memory savings calculation
- When QLoRA > LoRA > full fine-tuning

### 6. Common Pitfalls

| Symptom                 | Cause                         | Fix                           |
| ----------------------- | ----------------------------- | ----------------------------- |
| Loss explodes           | LR too high                   | Reduce LR, add warmup         |
| Loss plateaus           | LR too low or wrong schedule  | Increase LR, try cosine decay |
| Catastrophic forgetting | Overfitting to fine-tune data | Lower LR, fewer epochs, LoRA  |
| OOM during training     | Full precision + large batch  | QLoRA + gradient accumulation |

---

## 📖 Resources

- Sebastian Ruder — "An Overview of Gradient Descent Optimization Algorithms"
- Hu et al. — "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- Dettmers et al. — "QLoRA" paper (2023)

---

## ➡️ Next Steps

Continue to **Phase 1: LLM Fundamentals** → [LLM Fundamentals](../phase_1_llm_fundamentals/00_llm_fundamentals.md)
