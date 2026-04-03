# 📚 Phase 1: LLM Fundamentals

## Overview

Essential skills for working with Large Language Models. Covers how LLMs work, prompt engineering, API integration, and structured outputs.

---

## 📖 Notes Index

| #   | Topic                                                              | Description                                         |
| --- | ------------------------------------------------------------------ | --------------------------------------------------- |
| 00  | [LLM Fundamentals](./00_llm_fundamentals.md)                       | How LLMs work, sampling, prompting, model selection |
| 01  | [Working with APIs](./01_working_with_apis.md)                     | OpenAI, Anthropic, streaming, error handling, costs |
| 02  | [Structured Outputs](./02_structured_outputs.md)                   | JSON mode, Instructor, function calling, MCP        |
| 03  | [Model Selection](./03_model_selection.md)                         | Benchmarks, model routing, open vs closed tradeoffs |
| 04  | [Prompt Engineering Patterns](./04_prompt_engineering_patterns.md) | DSPy, chaining, versioning, few-shot selection      |

---

## 🔗 Key Concepts

### What You Need to Know

| Concept                    | Why It Matters             |
| -------------------------- | -------------------------- |
| **Sampling (temp, top_p)** | Control output randomness  |
| **Prompt Engineering**     | Get better outputs         |
| **Streaming**              | Real-time user experience  |
| **Function Calling**       | Connect LLMs to tools/APIs |
| **Context Management**     | Handle long conversations  |
| **Cost Tracking**          | Stay within budget         |

---

## 📊 Quick Reference

### Sampling Parameters

| Parameter   | Low               | High            | Use Case           |
| ----------- | ----------------- | --------------- | ------------------ |
| Temperature | 0 (deterministic) | 2 (random)      | 0.7 for most tasks |
| Top-P       | 0.1 (focused)     | 1.0 (full dist) | 0.9 default        |
| Top-K       | 1 (greedy)        | 100 (diverse)   | 40-50 typical      |

### Prompt Patterns

| Pattern       | Example                       | When to Use   |
| ------------- | ----------------------------- | ------------- |
| Zero-shot     | "Classify: ..."               | Simple tasks  |
| Few-shot      | "Example: ...\nClassify: ..." | Need examples |
| CoT           | "Let's think step by step"    | Reasoning     |
| System prompt | "You are a..."                | Set behavior  |

---

## ➡️ Next Steps

After Phase 1, continue to:

- **Phase 2: RAG** for retrieval-augmented generation
- **Phase 3: Memory Systems** for persistent, adaptive agent memory
