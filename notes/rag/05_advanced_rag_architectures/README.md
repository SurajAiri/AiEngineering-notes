# 2.4.5 Advanced RAG Architectures

> Learn these only after you can build and debug a strong standard RAG pipeline.

## 📌 Key Lesson

These architectures address specific limitations of basic RAG. GraphRAG adds reasoning over entity relationships. Agentic RAG adds self-reflection. Multimodal RAG handles images and tables. CRAG adds quality assessment with fallback. Each adds complexity — use them only when standard RAG demonstrably fails.

## Files

| File                                           | Topic                 | Key Concepts                                                                                          |
| ---------------------------------------------- | --------------------- | ----------------------------------------------------------------------------------------------------- |
| [01_graph_rag.md](./01_graph_rag.md)           | GraphRAG              | Entity extraction, relationship mapping, graph traversal, combining vector + graph retrieval          |
| [02_agentic_rag.md](./02_agentic_rag.md)       | Agentic RAG           | Self-reflective retrieval, iterative refinement, query-time decision making, LangGraph implementation |
| [03_multimodal_rag.md](./03_multimodal_rag.md) | Multimodal RAG        | Image/table retrieval, cross-modal retrieval, vision-language model integration                       |
| [04_corrective_rag.md](./04_corrective_rag.md) | Corrective RAG (CRAG) | Retrieval quality assessment, fallback strategies, web search augmentation                            |

## Syllabus Mapping

Maps to **§2.4.5** in `p2_rag_depth.md` — covers GraphRAG, Agentic RAG, Multimodal RAG, and Corrective RAG (CRAG) with both conceptual explanations and working implementations.
