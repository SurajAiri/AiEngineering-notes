# 2.6–2.8 Evaluation (Ground Truth, Retrieval Metrics, Faithfulness)

> Prove that the RAG system is correct, reliable, and improving.

## 📌 Key Lesson

Retrieval metrics (Recall, MRR, nDCG) tell you if the right chunks were found. Faithfulness tells you if the LLM used them correctly. Correctness tells you if the final answer matches reality. You need **all three** to trust a RAG system.

## Files

| File                                                               | Topic                          | Key Concepts                                                                                                                             |
| ------------------------------------------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [01_ground_truth_eval_data.md](./01_ground_truth_eval_data.md)     | Ground Truth & Evaluation Data | Golden query sets, document-answer alignment, query difficulty stratification, temporal evaluation sets, synthetic eval data generation  |
| [02_retrieval_evaluation.md](./02_retrieval_evaluation.md)         | Retrieval Evaluation           | Recall@k, MRR, nDCG, coverage vs precision tradeoffs, hybrid retrieval attribution, evaluation reporting                                 |
| [03_faithfulness_correctness.md](./03_faithfulness_correctness.md) | Faithfulness & Correctness     | Claim decomposition, context-only answering, answer-context attribution, citation accuracy, refusal evaluation, partial answer detection |

## Syllabus Mapping

Maps to **§2.6–2.8** in `p2_rag_depth.md`:

- **§2.6** — Golden query sets, document-answer alignment, query difficulty stratification, temporal evaluation sets
- **§2.7** — Recall@k, MRR, nDCG, coverage vs precision, hybrid retrieval attribution
- **§2.8** — Context-only answering, answer-context attribution, citation accuracy, refusal conditions, partial answer detection
