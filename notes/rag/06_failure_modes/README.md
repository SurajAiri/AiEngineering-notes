# 2.5 Failure Modes (Build-Time Awareness)

> This phase teaches what can go wrong while building, before formal evaluation starts.

## 📌 Key Lesson

Most RAG failures are **silent** — the system returns an answer that looks plausible but is wrong. Understanding all 6 failure modes before they happen lets you design defenses proactively.

## Files

| File                                                 | Topic                   | Key Concepts                                                                                                                                                                                                                    |
| ---------------------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [01_rag_failure_modes.md](./01_rag_failure_modes.md) | All 6 RAG Failure Modes | Over-retrieval hallucination, missing context hallucination, retrieval noise amplification, stale data errors, chunk boundary hallucinations, authority inversion — with detection signals, prevention code, and summary matrix |

## The 6 Failure Modes at a Glance

1. **Over-Retrieval Hallucination** — Too many chunks, LLM invents connections
2. **Missing Context Hallucination** — Right answer exists but retrieval missed it
3. **Retrieval Noise Amplification** — Irrelevant chunks dilute good ones
4. **Stale Data Errors** — Index has outdated info
5. **Chunk Boundary Hallucinations** — Answer split across chunks, LLM fills the gap
6. **Authority Inversion** — Low-quality sources outrank high-quality ones

## Syllabus Mapping

Maps to **§2.5** in `p2_rag_depth.md` — covers all 6 failure modes with detection strategies, prevention techniques, and production code examples.
