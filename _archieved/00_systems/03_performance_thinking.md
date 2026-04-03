# ⚡ Performance Thinking for AI Systems

## 📚 Overview

Performance engineering for AI systems requires different thinking than traditional web apps. LLM calls are **10–100x slower** than database queries, embedding computations are CPU/GPU-bound, and vector searches have unique latency profiles. This note covers profiling, caching, tail latency management, and memory optimization — all through the lens of AI workloads.

> 📌 _The biggest performance win in AI engineering is usually avoiding an LLM call entirely — not making it faster._

---

## 🎯 Learning Objectives

- **Profile** AI workloads to identify bottlenecks (CPU, GPU, I/O, network)
- **Design** caching strategies for embeddings, LLM responses, and retrieval results
- **Manage** tail latency in multi-step pipelines (RAG, agents)
- **Detect and fix** memory leaks in long-running LLM/embedding services
- **Apply** batching strategies for embedding and inference workloads

---

## 🧠 Sections (To Be Written)

### 1. Profiling AI Workloads

- CPU vs GPU vs I/O bound identification
- cProfile, py-spy, line_profiler for Python
- Profiling async pipelines (where time actually goes in RAG)
- Flame graphs for LLM-serving systems

### 2. Caching Strategies

- Embedding cache (semantic dedup before compute)
- LLM response cache (exact match, semantic similarity)
- Retrieval result cache (query → chunks)
- Cache invalidation strategies for dynamic knowledge bases
- Redis vs in-memory vs disk cache tradeoffs

### 3. Tail Latency Management

- P50 vs P95 vs P99 in multi-step pipelines
- Hedged requests for LLM API calls
- Timeout budgets across pipeline stages
- Adaptive timeouts based on model/provider

### 4. Memory Optimization

- Memory leak detection in long-running services (tracemalloc, objgraph)
- Embedding model memory footprint management
- Batch size tuning for GPU memory
- Memory-mapped files for large document collections

### 5. Batching & Throughput

- Embedding batching (optimal batch sizes per model)
- Dynamic batching for inference servers
- Async batching with queue-based architectures
- Throughput vs latency tradeoffs

### 6. Common Pitfalls

| Symptom                 | Cause                 | Fix                        |
| ----------------------- | --------------------- | -------------------------- |
| Slow first request      | Cold model loading    | Preload models at startup  |
| Memory growth over time | Unbounded caches      | LRU with TTL               |
| P99 spikes              | LLM provider variance | Hedged requests + timeouts |
| GPU OOM                 | Large batch sizes     | Dynamic batch sizing       |

---

## 📖 Resources

- Brendan Gregg — Systems Performance (book)
- py-spy: Sampling profiler for Python
- tracemalloc: Python memory tracking

---

## ➡️ Next Steps

Continue to [Testing & Quality](./04_testing_quality.md) →
