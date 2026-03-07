# 2.3 Retrieval (Core of RAG)

> This is the heart of RAG. Most real-world quality problems come from weak retrieval, not weak generation.

## 📌 Key Lesson

Retrieval is not just "vector search." A production retrieval layer combines dense and sparse search, rewrites queries, knows when to abstain, and understands the tradeoffs of approximate nearest neighbor algorithms.

## Files

| File                                                                                 | Topic                           | Key Concepts                                                            |
| ------------------------------------------------------------------------------------ | ------------------------------- | ----------------------------------------------------------------------- |
| [01_vector_similarity_search.md](./01_vector_similarity_search.md)                   | Vector Similarity               | Cosine similarity, dot product, L2 distance, embedding models           |
| [02_bm25_keyword_search.md](./02_bm25_keyword_search.md)                             | BM25 / Keyword Search           | TF-IDF evolution, BM25 scoring, when keywords beat vectors              |
| [03_hybrid_retrieval.md](./03_hybrid_retrieval.md)                                   | Hybrid Retrieval                | Dense + sparse fusion, RRF (Reciprocal Rank Fusion), metadata filtering |
| [04_metadata_filtering.md](./04_metadata_filtering.md)                               | Metadata Filtering              | Pre vs post filtering, multi-field filters, filter design patterns      |
| [05_query_rewriting_and_expansion.md](./05_query_rewriting_and_expansion.md)         | Query Rewriting & Expansion     | Multi-query expansion, query decomposition, LLM-based rewriting         |
| [06_hyde.md](./06_hyde.md)                                                           | HyDE                            | Hypothetical Document Embeddings, when HyDE helps vs hurts              |
| [07_query_document_mismatch.md](./07_query_document_mismatch.md)                     | Query-Document Mismatch         | Vocabulary mismatch, underspecified queries, diagnosis & remediation    |
| [08_adaptive_retrieval_and_abstention.md](./08_adaptive_retrieval_and_abstention.md) | Adaptive Retrieval & Abstention | Dynamic k, score-based abstention, when to return nothing               |
| [09_index_algorithms.md](./09_index_algorithms.md)                                   | Index Algorithms                | HNSW vs IVF vs DiskANN, approximate vs exact search, index refresh      |
| [10_quantization_aware_embeddings.md](./10_quantization_aware_embeddings.md)         | Quantization                    | Binary, scalar, product quantization, storage vs quality tradeoffs      |
| [11_vector_databases_hands_on.md](./11_vector_databases_hands_on.md)                 | Vector Databases                | Pinecone, Weaviate, PGVector, Qdrant, Milvus — hands-on comparisons     |

## Syllabus Mapping

Maps to **§2.3** in `p2_rag_depth.md` — covers vector similarity, BM25, hybrid retrieval, metadata filtering, query rewriting, multi-query expansion, HyDE, query-document mismatch mechanics, adaptive retrieval depth, retrieval abstention, HNSW/IVF/DiskANN tradeoffs, approximate vs exact search, index refresh strategies, quantization-aware embeddings, and hands-on with vector database systems.
