# 📚 Phase 0: Systems & Engineering Core

## Overview

This module covers the engineering fundamentals required for building production AI systems. These skills are the foundation for everything else.

---

## 📖 Notes Index

| #   | Topic                                                        | Description                                           |
| --- | ------------------------------------------------------------ | ----------------------------------------------------- |
| 00  | [Python & Concurrency](./00_python_concurrency.md)           | Asyncio, threading, producer-consumer                 |
| 01  | [API Design](./01_api_design.md)                             | FastAPI, streaming, WebSockets                        |
| 02  | [Observability](./02_observability.md)                       | Logging, metrics, tracing                             |
| 03  | [Performance Thinking](./03_performance_thinking.md)         | Profiling, caching, tail latency, memory optimization |
| 04  | [Testing & Quality](./04_testing_quality.md)                 | Mocking LLMs, property testing, snapshot tests, CI/CD |
| 05  | [Containerization & DevOps](./05_containerization_devops.md) | Docker, GPU containers, K8s, CI/CD                    |

---

## 🔗 From the Checklist

### 0.1 Python & Concurrency

- [x] Write async pipelines (asyncio)
- [x] Design producer–consumer systems
- [x] Correctly use threads vs processes
- [x] Backpressure handling
- [x] Frame-based streaming pipelines
- [ ] Queue overflow & recovery strategies

### 0.2 API Design & Async

- [x] Design RESTful APIs
- [x] Implement async endpoints
- [x] Streaming responses (SSE)
- [x] WebSocket connections
- [x] Error handling patterns
- [x] Rate limiting

### 0.3 Observability

- [x] Structured logging (JSON)
- [x] Prometheus metrics
- [x] Distributed tracing
- [x] Alerting rules
- [ ] Synthetic monitoring

---

## 📊 Skills Coverage

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 0 COVERAGE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Covered:                                                      │
│   ✓ Python async/concurrency patterns                          │
│   ✓ API design (REST, streaming, WebSocket)                    │
│   ✓ Observability (logging, metrics, tracing)                  │
│                                                                  │
│   Stub Notes (section headers, to be filled):                   │
│   ○ Performance thinking (profiling, caching)                   │
│   ○ Testing & quality (mocking LLMs, CI/CD)                     │
│   ○ Containerization & DevOps (Docker, K8s)                     │
│                                                                  │
│   Not Yet Covered:                                               │
│   ○ Database patterns                                           │
│   ○ Security fundamentals                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ➡️ Next Steps

After Phase 0, continue to:

- **Phase 0.5**: ML Foundations (transformers, embeddings)
- **Phase 1**: LLM Fundamentals (prompting, tokens, context)
