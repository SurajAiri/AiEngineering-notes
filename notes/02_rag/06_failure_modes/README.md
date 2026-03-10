# 2.5 Failure Modes (Build-Time Awareness)

> This phase teaches what can go wrong while building, before formal evaluation starts.

## 📌 Key Lesson

Most RAG failures are **silent** — the system returns an answer that looks plausible but is wrong. Understanding all 6 failure modes before they happen lets you design defenses proactively.

## Files

| #   | File                                                                     | Topic                         | Key Concepts                                                                                                                                                                                                                    |
| --- | ------------------------------------------------------------------------ | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | [01_rag_failure_modes.md](./01_rag_failure_modes.md)                     | All 6 RAG Failure Modes       | Over-retrieval hallucination, missing context hallucination, retrieval noise amplification, stale data errors, chunk boundary hallucinations, authority inversion — with detection signals, prevention code, and summary matrix |
| 02  | [02_rag_debugging_checklist.md](./02_rag_debugging_checklist.md)         | RAG Debugging Checklist       | 6-step diagnostic pipeline: Data → Chunking → Retrieval → Re-ranking → Context → Generation, automated diagnosis script                                                                                                         |
| 03  | [03_guardrails_prompt_injection.md](./03_guardrails_prompt_injection.md) | Guardrails & Prompt Injection | Input/output/context sanitization, NeMo Guardrails, LlamaGuard, Presidio PII detection, production guardrails pipeline                                                                                                          |

## The 6 Failure Modes at a Glance

| #   | Failure Mode                  | What Happens                                  | Which Stage Causes It       |
| --- | ----------------------------- | --------------------------------------------- | --------------------------- |
| 1   | Over-Retrieval Hallucination  | Too many chunks, LLM invents connections      | Retrieval (k too high)      |
| 2   | Missing Context Hallucination | Right answer exists but retrieval missed it   | Retrieval / Chunking        |
| 3   | Retrieval Noise Amplification | Irrelevant chunks dilute good ones            | Retrieval / Re-ranking      |
| 4   | Stale Data Errors             | Index has outdated info                       | Data Ingestion / Production |
| 5   | Chunk Boundary Hallucinations | Answer split across chunks, LLM fills the gap | Chunking                    |
| 6   | Authority Inversion           | Low-quality sources outrank high-quality ones | Metadata / Re-ranking       |

## Common Questions

### Q: How do I know which failure mode is happening in my system?

**A:** You can't always tell from the output alone. That's why you need:

1. **Retrieval tracing** — log what chunks were retrieved and their scores
2. **Evaluation data** — compare against known-correct answers
3. **The detection heuristics** in the detailed file for each mode

### Q: Which failure mode is the most common?

**A:** **Missing context hallucination** (#2) — the system didn't find the right chunk, so the LLM makes up an answer. This is usually a chunking or retrieval problem, and it's often fixable by improving your chunking strategy or using hybrid search.

### Q: Can I prevent all failure modes?

**A:** You can't eliminate them entirely, but you can make them rare and detectable. The key defenses are: adaptive k (prevents #1 and #3), better chunking (#5), metadata scoring (#6), and index freshness checks (#4).

## Syllabus Mapping

Maps to **§2.5** in `p2_rag_depth.md` — covers all 6 failure modes with detection strategies, prevention techniques, and production code examples.
