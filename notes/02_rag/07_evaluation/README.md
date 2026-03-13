# 2.6–2.8 Evaluation (Ground Truth, Retrieval Metrics, Faithfulness)

> Prove that the RAG system is correct, reliable, and improving.

## 📌 Key Lesson

Retrieval metrics (Recall, MRR, nDCG) tell you if the right chunks were found. Faithfulness tells you if the LLM used them correctly. Correctness tells you if the final answer matches reality. You need **all three** to trust a RAG system.

## Learning Order

```
Ground truth data (what correct looks like)
→ Retrieval metrics (are we finding the right chunks?)
→ Faithfulness & correctness (is the LLM using them properly?)
```

## Files

| #   | File                                                               | Topic                          | Key Concepts                                                                                                                             |
| --- | ------------------------------------------------------------------ | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 01  | [01_ground_truth_eval_data.md](./01_ground_truth_eval_data.md)     | Ground Truth & Evaluation Data | Golden query sets, document-answer alignment, query difficulty stratification, temporal evaluation sets, synthetic eval data generation  |
| 02  | [02_retrieval_evaluation.md](./02_retrieval_evaluation.md)         | Retrieval Evaluation           | Recall@k, MRR, nDCG, coverage vs precision tradeoffs, hybrid retrieval attribution, evaluation reporting                                 |
| 03  | [03_faithfulness_correctness.md](./03_faithfulness_correctness.md) | Faithfulness & Correctness     | Claim decomposition, context-only answering, answer-context attribution, citation accuracy, refusal evaluation, partial answer detection |
| 04  | [04_evaluation_frameworks.md](./04_evaluation_frameworks.md)       | Evaluation Frameworks          | RAGAS, DeepEval, LangSmith, LangFuse, Arize Phoenix — setup, metrics, combining frameworks                                               |
| 05  | [05_automated_eval_pipeline.md](./05_automated_eval_pipeline.md)   | Automated Eval Pipeline        | CI/CD integration, golden test sets, regression detection, GitHub Actions, DeepEval pytest                                               |

## Popular Libraries

| Library       | Purpose                    | Notes                                                             |
| ------------- | -------------------------- | ----------------------------------------------------------------- |
| RAGAS         | RAG-specific evaluation    | Metrics: faithfulness, answer relevancy, context precision/recall |
| DeepEval      | LLM evaluation framework   | Many metrics, synthetic test generation                           |
| LangSmith     | Tracing + evaluation       | By LangChain, integrates with LangChain apps                      |
| Arize Phoenix | Observability + evaluation | Open-source LLM observability                                     |
| TruLens       | Evaluation + feedback      | Tracks faithfulness, groundedness                                 |

### Quick RAGAS Example

```python
"""
Evaluate your RAG system with RAGAS — the most popular RAG evaluation framework.
Requirements: pip install ragas datasets
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

# Your RAG results
eval_data = {
    "question": ["What is the return policy?"],
    "answer": ["Returns are accepted within 30 days with a receipt."],
    "contexts": [["Our return policy allows full refunds within 30 days of purchase with a receipt."]],
    "ground_truth": ["Full refunds within 30 days with receipt."],
}

dataset = Dataset.from_dict(eval_data)
results = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(results)
# {'faithfulness': 1.0, 'answer_relevancy': 0.95, 'context_precision': 1.0, 'context_recall': 1.0}
```

## Common Questions

### Q: How much ground truth data do I need?

**A:** Start with **50-100 golden query-answer pairs** covering your most important use cases. For production evaluation, aim for 200-500 covering different difficulty levels. Quality matters more than quantity — 50 well-curated pairs beats 1000 sloppy ones.

### Q: Can I generate evaluation data automatically?

**A:** Yes. Use an LLM to generate question-answer pairs from your documents, then **manually verify** a sample. Tools like RAGAS and DeepEval have built-in synthetic test generation. But always manually check — auto-generated eval data can have systematic blind spots.

### Q: What's the most important metric to track?

**A:** **Recall@k** for retrieval (are we finding the right chunks?) and **faithfulness** for generation (is the LLM sticking to the context?). If recall is low, your retrieval is broken. If faithfulness is low, your LLM is hallucinating despite having good context.

### Q: How often should I run evaluations?

**A:** Run retrieval metrics on every deployment. Run faithfulness/correctness evaluations weekly or when you change the pipeline. Set up automated regression tests that alert you when metrics drop.

## Syllabus Mapping

Maps to **2.6–2.8** in `p2_rag_depth.md`:

- **2.6** — Golden query sets, document-answer alignment, query difficulty stratification, temporal evaluation sets
- **2.7** — Recall@k, MRR, nDCG, coverage vs precision, hybrid retrieval attribution
- **2.8** — Context-only answering, answer-context attribution, citation accuracy, refusal conditions, partial answer detection
