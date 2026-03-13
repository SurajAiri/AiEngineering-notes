# 2.4.5 Advanced RAG Architectures

> ⚠️ **Advanced / Optional.** Learn these only after you can build and debug a strong standard RAG pipeline.

## 📌 Key Lesson

These architectures address specific limitations of basic RAG. GraphRAG adds reasoning over entity relationships. Agentic RAG adds self-reflection. Multimodal RAG handles images and tables. CRAG adds quality assessment with fallback. Each adds complexity — use them only when standard RAG demonstrably fails.

## When Do You Need These?

```
Standard RAG works fine for:
  ✅ Single-document factual Q&A
  ✅ FAQ-style queries
  ✅ Technical documentation lookup

You might need advanced architectures when:
  ⚠️ Queries require combining info across many documents → GraphRAG
  ⚠️ Users ask multi-step questions requiring iterative search → Agentic RAG
  ⚠️ Documents contain important images/tables → Multimodal RAG
  ⚠️ Retrieval quality is unreliable and needs fallbacks → CRAG
```

## Files

| #   | File                                           | Topic                 | Key Concepts                                                                                          | Complexity |
| --- | ---------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------- | ---------- |
| 01  | [01_graph_rag.md](./01_graph_rag.md)           | GraphRAG              | Entity extraction, relationship mapping, graph traversal, combining vector + graph retrieval          | High       |
| 02  | [02_agentic_rag.md](./02_agentic_rag.md)       | Agentic RAG           | Self-reflective retrieval, iterative refinement, query-time decision making, LangGraph implementation | High       |
| 03  | [03_multimodal_rag.md](./03_multimodal_rag.md) | Multimodal RAG        | Image/table retrieval, cross-modal retrieval, vision-language model integration                       | Medium     |
| 04  | [04_corrective_rag.md](./04_corrective_rag.md) | Corrective RAG (CRAG) | Retrieval quality assessment, fallback strategies, web search augmentation                            | Medium     |

## Popular Libraries

| Library            | Purpose                             | Notes                                  |
| ------------------ | ----------------------------------- | -------------------------------------- |
| LangGraph          | Agentic RAG workflows               | State machines for iterative retrieval |
| Neo4j              | Graph database for GraphRAG         | Store and query entity relationships   |
| microsoft/graphrag | Microsoft's GraphRAG implementation | End-to-end GraphRAG pipeline           |
| LlamaIndex         | Multi-modal indexes                 | Built-in support for image/table RAG   |

## Common Questions

### Q: Should I learn these before getting a job?

**A:** Know what they are and when they're useful. You don't need to build them from scratch. In interviews, explaining WHEN you'd use GraphRAG vs standard RAG shows more maturity than having built a GraphRAG demo.

### Q: Which one should I learn first?

**A:** **Corrective RAG (CRAG)** — it's the simplest and most practical. It adds quality checks and fallbacks to standard RAG, which is directly useful in production. GraphRAG and Agentic RAG are more specialized.

## Syllabus Mapping

Maps to **2.4.5** in `p2_rag_depth.md` — covers GraphRAG, Agentic RAG, Multimodal RAG, and Corrective RAG (CRAG) with both conceptual explanations and working implementations.
