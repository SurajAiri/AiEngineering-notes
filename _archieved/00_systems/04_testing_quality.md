# 🧪 Testing & Quality for AI Systems

## 📚 Overview

Testing AI systems is fundamentally different from testing deterministic software. LLM outputs are non-deterministic, RAG pipelines have probabilistic components, and "correct" is often subjective. This note covers testing strategies that actually work for AI engineering — from unit testing embeddings to snapshot testing prompts to integration testing full RAG pipelines.

> 📌 _If you can't test it, you can't ship it. AI systems need MORE testing discipline, not less — because failures are subtle._

---

## 🎯 Learning Objectives

- **Design** test suites for non-deterministic AI components
- **Mock** LLM responses for fast, reliable unit tests
- **Implement** property-based testing for embedding and retrieval pipelines
- **Build** snapshot tests for prompt templates (catch regressions)
- **Structure** CI/CD pipelines with AI-specific test stages

---

## 🧠 Sections (To Be Written)

### 1. Unit Testing AI Components

- Testing embedding functions (determinism, dimensionality, normalization)
- Testing chunking logic (boundary conditions, overlap correctness)
- Testing prompt templates (variable injection, token counting)
- Testing structured output parsers (schema validation)

### 2. Mocking LLM Responses

- Mock patterns for OpenAI/Anthropic clients
- Fixture-based testing (recorded responses)
- Deterministic seeding for reproducible tests
- Mock streaming responses

### 3. Property-Based Testing

- Hypothesis library for AI pipelines
- Properties: embedding similarity is symmetric, chunking preserves content
- Fuzzing prompt templates with adversarial inputs
- Testing retrieval ranking invariants

### 4. Snapshot Testing for Prompts

- Prompt regression detection
- Template versioning and diff tracking
- Token count snapshots (catch prompt bloat)
- System prompt stability testing

### 5. Integration Testing RAG Pipelines

- End-to-end retrieval accuracy tests
- Golden dataset testing (known query → expected chunks)
- Latency regression tests
- Cost regression tests (token usage tracking)

### 6. CI/CD for AI Systems

- Fast vs slow test separation (mock vs real LLM)
- Nightly evaluation runs
- Model version pinning in tests
- Environment parity (embedding model versioning)

### 7. Common Pitfalls

| Symptom            | Cause                        | Fix                                |
| ------------------ | ---------------------------- | ---------------------------------- |
| Flaky tests        | Non-deterministic LLM output | Mock or set temperature=0          |
| Slow CI            | Real API calls in tests      | Mock layer + nightly real tests    |
| Silent regressions | No prompt snapshot tests     | Add snapshot + diff checks         |
| False confidence   | Testing happy path only      | Property-based + adversarial tests |

---

## 📖 Resources

- Hypothesis: Property-based testing for Python
- pytest-recording: Record/replay HTTP responses
- promptfoo: Prompt testing framework

---

## ➡️ Next Steps

Continue to [Containerization & DevOps](./05_containerization_devops.md) →
