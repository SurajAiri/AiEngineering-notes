# 2.3 Retrieval (Core of RAG)

> This is the heart of RAG. Most real-world quality problems come from weak retrieval, not weak generation.

## 📌 Key Lesson

Retrieval is not just "vector search." A production retrieval layer combines dense and sparse search, rewrites queries, knows when to abstain, and understands the tradeoffs of approximate nearest neighbor algorithms.

## Learning Order

Learn the basics (how search works), then why it fails, then how to fix it:

```
Vector search → BM25 → Hybrid (combine both)
→ Metadata filtering (scope your search)
→ Query-document mismatch (WHY retrieval fails)
→ Query rewriting (FIX mismatches) → HyDE (advanced fix)
→ Adaptive retrieval (when to return nothing)
→ Index algorithms & quantization (scale it)
→ Vector databases (hands-on)
```

## Files

| #   | File                                                                                 | Topic                           | Key Concepts                                                            |
| --- | ------------------------------------------------------------------------------------ | ------------------------------- | ----------------------------------------------------------------------- |
| 01  | [01_vector_similarity_search.md](./01_vector_similarity_search.md)                   | Vector Similarity               | Cosine similarity, dot product, L2 distance, embedding models           |
| 02  | [02_bm25_keyword_search.md](./02_bm25_keyword_search.md)                             | BM25 / Keyword Search           | TF-IDF evolution, BM25 scoring, when keywords beat vectors              |
| 03  | [03_hybrid_retrieval.md](./03_hybrid_retrieval.md)                                   | Hybrid Retrieval                | Dense + sparse fusion, RRF (Reciprocal Rank Fusion), metadata filtering |
| 04  | [04_metadata_filtering.md](./04_metadata_filtering.md)                               | Metadata Filtering              | Pre vs post filtering, multi-field filters, filter design patterns      |
| 05  | [05_query_document_mismatch.md](./05_query_document_mismatch.md)                     | Query-Document Mismatch         | Vocabulary mismatch, underspecified queries, diagnosis & remediation    |
| 06  | [06_query_rewriting_and_expansion.md](./06_query_rewriting_and_expansion.md)         | Query Rewriting & Expansion     | Multi-query expansion, query decomposition, LLM-based rewriting         |
| 07  | [07_hyde.md](./07_hyde.md)                                                           | HyDE                            | Hypothetical Document Embeddings, when HyDE helps vs hurts              |
| 08  | [08_adaptive_retrieval_and_abstention.md](./08_adaptive_retrieval_and_abstention.md) | Adaptive Retrieval & Abstention | Dynamic k, score-based abstention, when to return nothing               |
| 09  | [09_index_algorithms.md](./09_index_algorithms.md)                                   | Index Algorithms                | HNSW vs IVF vs DiskANN, approximate vs exact search, index refresh      |
| 10  | [10_quantization_aware_embeddings.md](./10_quantization_aware_embeddings.md)         | Quantization                    | Binary, scalar, product quantization, storage vs quality tradeoffs      |
| 11  | [11_vector_databases_hands_on.md](./11_vector_databases_hands_on.md)                 | Vector Databases                | Pinecone, Weaviate, PGVector, Qdrant, Milvus — hands-on comparisons     |
| 12  | [12_retrieval_routing.md](./12_retrieval_routing.md)                                 | Retrieval Routing               | Query classification, strategy selection, LlamaIndex RouterQueryEngine  |
| 13  | [13_semantic_caching.md](./13_semantic_caching.md)                                   | Semantic Caching                | Embedding-based cache, TTL, LangChain cache, GPTCache                   |
| 00  | [00_retrieval_with_libraries.md](./00_retrieval_with_libraries.md)                   | 📚 Library Cookbook             | LangChain retrievers, LlamaIndex retrievers, vector DB integrations     |

> **Files 09-11 are "good to read" topics** — important for scaling and production, but you can build a working RAG system without them. Come back to these when you need to optimize.

## Popular Libraries

| Library               | Purpose                     | Notes                                              |
| --------------------- | --------------------------- | -------------------------------------------------- |
| FAISS                 | Vector similarity search    | Facebook's library, CPU/GPU, great for prototyping |
| sentence-transformers | Embedding models            | Free, open-source, many pre-trained models         |
| rank_bm25             | BM25 keyword search         | Simple Python BM25 implementation                  |
| Pinecone/Qdrant/etc.  | Managed vector databases    | Production-ready, see file 11 for details          |
| LangChain Retrievers  | Unified retrieval interface | Supports vector, BM25, hybrid, ensemble            |
| LlamaIndex Retrievers | Document-aware retrieval    | Built-in hybrid, auto-merging, fusion              |

## Common Questions

### Q: Do I use the retriever first, then filter by metadata? Or filter first, then retrieve?

**A:** Most vector databases do **in-search filtering** — they filter and search simultaneously. This is the best approach. If your database doesn't support it, prefer **pre-filtering** (filter first, then search the filtered subset) for selective filters, and **post-filtering** (search first, then filter results) when filters are broad. See [04_metadata_filtering.md](./04_metadata_filtering.md) for details.

### Q: How is metadata connected to retrieved data?

**A:** Each chunk is stored with its text, embedding vector, AND metadata (source, date, section, etc.). When retrieved, the metadata travels with the chunk. You use metadata for:

1. **Filtering** — narrow search scope before/during retrieval
2. **Attribution** — show users where the answer came from
3. **Ranking** — boost current versions over old ones

### Q: When should I use vector search vs BM25 vs hybrid?

**A:**

- **Vector search**: When queries are conversational or use different words than documents
- **BM25**: When queries contain exact terms that must match (error codes, product names, IDs)
- **Hybrid** (recommended default): Combines both. Start here unless you have a reason not to.

### Q: How many chunks should I retrieve (what k)?

**A:** Start with **k=5-10**. Too few = might miss relevant info. Too many = noise confuses the LLM. In production, use **adaptive k** — retrieve more and let re-ranking pick the best. See [08_adaptive_retrieval_and_abstention.md](./08_adaptive_retrieval_and_abstention.md).

### Q: What if the retriever returns nothing relevant?

**A:** That's a feature, not a bug — your system should **abstain** rather than hallucinate. Use score thresholds to detect when nothing good was found, and return "I don't have information about that" instead of making something up.

## Syllabus Mapping

Maps to **§2.3** in `p2_rag_depth.md` — covers vector similarity, BM25, hybrid retrieval, metadata filtering, query rewriting, multi-query expansion, HyDE, query-document mismatch mechanics, adaptive retrieval depth, retrieval abstention, HNSW/IVF/DiskANN tradeoffs, approximate vs exact search, index refresh strategies, quantization-aware embeddings, and hands-on with vector database systems.
