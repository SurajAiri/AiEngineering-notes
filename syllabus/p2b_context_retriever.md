I’ll keep this **engineering-grade**, layered, and failure-oriented.

I am treating this as **PHASE 2.6 / PHASE 2.7** so it nests naturally between RAG and Memory, but you can also promote it to its own phase if you want.

---

# 🔬 PHASE 2.6 — CONTEXT RETRIEVAL SYSTEMS (THE CORE BOTTLENECK)

> _This phase defines whether your LLM is intelligent or confidently wrong._

This phase is about **deciding what enters the context window** — not just retrieving chunks.

---

## 2.6.1 Problem Framing: Context Is a Scarce Resource

You must explicitly reason about:

- [ ] Context window = **fixed budget**
- [ ] Retrieval = **lossy compression**
- [ ] Every retrieved token **displaces another**
- [ ] Wrong context is worse than no context

📌 _If you don’t model context as a constrained optimization problem, you will over-retrieve._

---

## 2.6.2 Query Understanding & Decomposition

### A. Query Intent Classification

Before retrieval, classify:

- [ ] Factual lookup
- [ ] Procedural / how-to
- [ ] Multi-hop reasoning
- [ ] Personal / memory-based
- [ ] Ambiguous / underspecified

**Implementation patterns**

- Small classifier model
- Prompt-based intent router
- Heuristic + fallback

**Failure modes**

- Treating multi-hop as single-hop
- Using vector search for procedural questions

---

### B. Query Decomposition (Critical)

- [ ] Atomic sub-questions
- [ ] Temporal constraints
- [ ] Entity extraction
- [ ] Implicit assumptions

Example:

> “How did we handle auth in the last rollout?”

Becomes:

- Which project?
- Which time range?
- Which auth layer (API / OAuth / IAM)?

**Techniques**

- LLM-based query planner
- Deterministic rule expansion for known domains

---

## 2.6.3 Retrieval Strategy Selection (Routing Layer)

You should **not** always use vector search.

### Retrieval Routers Decide Between:

- [ ] Vector similarity
- [ ] Keyword / BM25
- [ ] Hybrid
- [ ] Structured DB query
- [ ] Memory store
- [ ] No retrieval (answer from model)

**Signals**

- Query length
- Presence of entities
- Temporal words (“latest”, “last time”)
- User identity / session state

📌 _This router is often more important than embedding quality._

---

## 2.6.4 Embedding Strategy (Often Done Wrong)

### A. Embedding Granularity

- [ ] Raw text embeddings
- [ ] Summary embeddings
- [ ] Hierarchical embeddings (doc → section → chunk)
- [ ] Metadata-aware embeddings

### B. Multiple Embedding Spaces

- [ ] One space for semantic similarity
- [ ] One space for intent/task similarity
- [ ] One space for memory recall

**Failure modes**

- Mixing semantic + procedural data in one space
- Re-embedding everything on schema change

---

## 2.6.5 Retrieval Control & Budgeting

### A. Explicit Token Budgeting

- [ ] Per-source budget caps
- [ ] Per-retriever quotas
- [ ] Hard truncation rules
- [ ] Soft relevance decay

Example:

```text
Context budget: 6k tokens
- Memory: 1k
- Docs: 3k
- Tools/logs: 1k
- Safety margin: 1k
```

### B. Score-Aware Truncation

- Sort by (relevance ÷ token_cost)
- Drop long low-density chunks
- Prefer summaries under pressure

📌 _Token efficiency beats raw relevance._

---

## 2.6.6 Re-ranking Beyond “Top-K”

- [ ] Cross-encoder re-ranking
- [ ] Query–chunk entailment scoring
- [ ] Redundancy penalties
- [ ] Diversity constraints (MMR-style)

**Anti-pattern**

> Top-5 vector results → dump into prompt

---

## 2.6.7 Context Assembly (Where Most Bugs Live)

### Assembly Is a Compiler Problem

- [ ] Ordering matters
- [ ] Group by source
- [ ] Explicit separators
- [ ] Stable formatting
- [ ] Citations aligned to spans

**Bad**

```
Chunk A
Chunk C
Chunk B
```

**Good**

```
[Doc: Auth Spec v3]
- Summary
- Relevant Section
```

---

## 2.6.8 Negative Retrieval & Abstention

Teach the system to say:

- [ ] “No relevant context found”
- [ ] “Data may be missing or outdated”
- [ ] “This conflicts with earlier sources”

This requires:

- Minimum relevance thresholds
- Conflict detection
- Confidence estimation

🚨 _Most hallucinations come from forced answering._

---

# 🧠 PHASE 2.7 — CONTEXTUAL MEMORY & LONGITUDINAL RETRIEVAL

This bridges RAG ↔ Memory.

---

## 2.7.1 Memory as a First-Class Retriever

Memory is **not just stored chat history**.

### Memory Types by Use

- [ ] Episodic (what happened)
- [ ] Semantic (what is true)
- [ ] Preference (how to respond)
- [ ] Procedural (how things are done)

Each has:

- Different decay
- Different retrieval logic
- Different trust levels

---

## 2.7.2 Memory Write Policy (Hard Problem)

You must define:

- [ ] What events are memorable
- [ ] Who decides (LLM vs rules)
- [ ] Confidence thresholds
- [ ] User consent / scope

**Failure modes**

- Storing hallucinations
- Reinforcing incorrect beliefs
- Memory bloat

---

## 2.7.3 Memory Retrieval Gating

Before injecting memory:

- [ ] Relevance to current goal
- [ ] Temporal validity
- [ ] Conflict with fresh context
- [ ] Trust score

📌 _Memory should compete with documents for tokens._

---

## 2.7.4 Memory ↔ RAG Arbitration

When memory and documents disagree:

- Prefer:
  - [ ] Newer data
  - [ ] Higher-confidence sources
  - [ ] Explicit user corrections

Surface conflicts to the model explicitly.

---

## 2.7.5 Memory Compression Pipelines

- [ ] Session summarization
- [ ] Event abstraction
- [ ] Hierarchical rollups
- [ ] Loss evaluation

Compression is **destructive** — measure what you lose.

---

# 🧪 PHASE 2.8 — EVALUATING CONTEXT QUALITY (NON-NEGOTIABLE)

You cannot tune what you don’t measure.

---

## Metrics You Must Track

### Retrieval-Level

- Recall@K
- MRR
- Source diversity
- Token waste ratio

### Context-Level

- % of context cited in answer
- Entailment score (answer ↔ context)
- Conflict rate
- Abstention correctness

### System-Level

- Latency per retriever
- Cost per answer
- Silent failure rate

### LLM-as-Judge Evaluation (NEW)

- Pairwise comparison methods
- Reference-based vs reference-free scoring
- Multi-criteria evaluation rubrics
- Judge model calibration
- Human-LLM agreement metrics

---

## Golden Debug Question

> “Was the right information available, retrieved, selected, and used?”

Break failures along those exact stages.

---

# 🧠 Mental Model to Keep

**LLMs don’t reason over knowledge — they reason over _whatever you put in front of them_.**
Your job is not to “improve the model”.
Your job is to **build a context compiler**.

---
