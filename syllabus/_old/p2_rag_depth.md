# JOB-FOCUSED PRODUCTION RAG SYLLABUS

_(RAG only. Lean enough to start, production-aware enough to matter. Split into **2(A)** build and **2(B)** validate so you learn both how to construct and how to trust a RAG system.)_

---

# **PHASE 2(A) - BUILDING REAL RAG SYSTEMS**

> Goal: _Given documents and a query, reliably retrieve the right information and assemble it into a usable context._

---

## **2.1 Data Ingestion (Pre-Retrieval Reality)**

- [ ] Clean vs noisy data handling
- [ ] Deduplication strategies
  - exact
  - near-duplicate
  - semantic
- [ ] Versioned documents
- [ ] Metadata design
- [ ] Source attribution
- [ ] Document canonicalization
  - normalized formatting
  - stable IDs
  - section path normalization
- [ ] Chunk-level provenance tracking
- [ ] Parser / OCR failure awareness
  - reading-order breaks
  - missing tables or captions
  - bad extraction from scanned documents

📌 _You do not need to build parser internals from scratch to get started. Using tools like LlamaParse, Unstructured, or LangExtract is fine, but you must understand when extraction quality breaks retrieval._

---

## **2.2 Chunking (Critical Layer)**

- [ ] Fixed-size chunking (baseline)
- [ ] Sliding window chunking
- [ ] Semantic chunking
- [ ] Hierarchical chunking
- [ ] Chunk overlap tradeoffs
- [ ] Chunk size vs recall vs cost
- [ ] Chunk-query alignment failure modes
- [ ] Structure-aware chunking
  - headings
  - tables
  - code blocks

📌 _Chunking is where you decide the retrieval unit. Good chunking preserves meaning; bad chunking destroys it before retrieval starts._

---

## **2.3 Retrieval (Expanded, Core of RAG)**

- [ ] Vector similarity search
- [ ] BM25 / keyword search
- [ ] Hybrid retrieval
- [ ] Metadata filtering
- [ ] Query rewriting
- [ ] Multi-query expansion
- [ ] Hypothetical Document Embeddings (HyDE)
- [ ] Query-document mismatch mechanics
  - vocabulary mismatch
  - underspecified queries
- [ ] Adaptive retrieval depth (dynamic `k`)
- [ ] Retrieval abstention (return nothing)
- [ ] HNSW vs IVF vs DiskANN tradeoffs
- [ ] Approximate vs exact search error bounds
- [ ] Index refresh strategies for dynamic data
- [ ] Quantization-aware embeddings
  - binary
  - product quantization
- [ ] Hands-on with systems:
  - Pinecone
  - Weaviate
  - PGVector
  - Qdrant
  - Milvus

📌 _This is the heart of RAG. Most real-world quality problems come from weak retrieval, not weak generation._

---

## **2.4 Re-ranking & Context Assembly**

- [ ] Cross-encoder re-ranking
- [ ] Score normalization
- [ ] Context deduplication
- [ ] Token budget allocation
- [ ] Citation alignment
- [ ] Context packing strategies
  - breadth-first vs depth-first
- [ ] Diversity vs relevance tradeoffs
- [ ] Ordering context for answerability

📌 _This section is about what text enters the model context and in what order, not about prompting theory._

---

## **2.4.5 Advanced RAG Architectures (Optional Later)**

- [ ] GraphRAG (knowledge graph + RAG)
  - entity extraction and relationship mapping
  - graph traversal for retrieval
  - combining vector and graph retrieval
- [ ] Agentic RAG (optional later)
  - self-reflective retrieval
  - iterative retrieval refinement
  - query-time decision making
- [ ] Multimodal RAG
  - image and table retrieval
  - cross-modal retrieval
  - vision-language model integration
- [ ] Corrective RAG (CRAG)
  - retrieval quality assessment
  - fallback strategies
  - web search augmentation

📌 _Learn these only after you can build and debug a strong standard RAG pipeline._

---

## **2.5 Failure Modes (Build-Time Awareness)**

- [ ] Over-retrieval hallucination
- [ ] Missing context hallucination
- [ ] Retrieval noise amplification
- [ ] Stale data errors
- [ ] Chunk boundary hallucinations
- [ ] Authority inversion (low-quality sources outranking high-quality sources)

📌 _This phase teaches what can go wrong while building, before formal evaluation starts._

---

# **PHASE 2(B) - VALIDATING & TRUSTING RAG SYSTEMS**

> Goal: _Prove that the RAG system is correct, reliable, and improving._

---

## **2.6 Ground Truth & Evaluation Data**

- [ ] Golden query sets
- [ ] Document-answer alignment
- [ ] Query difficulty stratification
- [ ] Temporal evaluation sets

---

## **2.7 Retrieval Evaluation**

- [ ] Recall@k
- [ ] MRR
- [ ] nDCG
- [ ] Coverage vs precision tradeoffs
- [ ] Hybrid retrieval attribution

---

## **2.8 Faithfulness & Correctness**

- [ ] Context-only answering constraints
- [ ] Answer-context attribution
- [ ] Citation accuracy
- [ ] Refusal conditions
- [ ] Partial answer detection

---

## **2.9 Reliability & Drift**

- [ ] Index drift
- [ ] Data freshness checks
- [ ] Silent degradation detection
- [ ] Regression testing for RAG

---

## **2.10 Cost, Latency & Stability**

- [ ] Token cost modeling
- [ ] Retrieval latency budgets
- [ ] p95 / p99 behavior
- [ ] Failure under load

---

## **2.11 RAG Observability & Debugging**

- [ ] Retrieval trace logging
- [ ] Chunk contribution analysis
- [ ] Source attribution tracking
- [ ] A/B testing for retrieval strategies
- [ ] User feedback integration
- [ ] Retrieval quality dashboards

---

## **2.12 Advanced Embedding Strategies (Optional Later)**

- [ ] Late-interaction models (ColBERT, ColPali)
- [ ] Matryoshka embeddings (variable dimensions)
- [ ] Domain-specific embedding fine-tuning
- [ ] Multi-vector representations

---

# FINAL ASSESSMENT

### 2(A) - Building RAG

**Status:** ✅ Focused and complete enough to build real systems  
**Quality:** Job-focused, production-aware, no unnecessary depth

### 2(B) - Trusting RAG

**Status:** ✅ Clear separation of evaluation from implementation  
**Quality:** Strong enough to learn how to measure trust, not just build pipelines

---

## The Most Important Insight

- 2(A) answers: _"Can I build it?"_
- 2(B) answers: _"Can I trust it?"_

Most engineers mix these together and end up with neither. Keeping them separate is what makes this syllabus practical.
