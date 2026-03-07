# PRODUCTION-FIRST DEEP RAG SYLLABUS

_(RAG only. Production-first by default. Split into **2(A)** build and **2(B)** validate so you learn both how to construct and how to trust a RAG system.)_

---

# **PHASE 2(A) — BUILDING REAL RAG SYSTEMS**

> Goal: _Given a corpus and a query, build a retrieval pipeline that can find the right evidence, preserve document structure, and assemble grounded context for generation._

---

## **2.0 RAG Foundations & Decision Framework**

- [ ] What RAG is actually solving
- [ ] When to use RAG vs long-context prompting
- [ ] When to use RAG vs SQL / structured querying
- [ ] When to use RAG vs graph search
- [ ] When to use RAG vs fine-tuning
- [ ] Task taxonomy
  - lookup / fact retrieval
  - comparison across sources
  - aggregation / synthesis
  - corpus-level or global reasoning
  - multimodal document QA
- [ ] Retrieval unit choices
  - chunk
  - page
  - section
  - document
  - table
  - figure
- [ ] Static vs dynamic corpus implications
- [ ] Freshness, recall, precision, latency, and cost tradeoffs
- [ ] Citation and grounding requirements by use case

📌 _RAG is not the default answer to every knowledge problem. Learn when it is the right system and when it is unnecessary complexity._

---

## **2.1 Document Ingestion & Understanding**

- [ ] Source inventory and corpus characterization
- [ ] Loader / parser selection for:
  - PDF
  - DOCX
  - HTML
  - Markdown
  - spreadsheets
  - slides
  - images / scans
- [ ] Born-digital vs scanned PDFs
- [ ] OCR pipelines and quality limits
- [ ] PDF reading-order problems
  - multi-column layouts
  - headers / footers
  - footnotes
  - sidebars / callouts
- [ ] Preserving document structure
  - titles / headings / subheadings
  - section paths
  - page numbers
  - lists
  - code blocks
  - formulas / equations
- [ ] Tables
  - table detection
  - cell extraction
  - merged cells / spanning headers
  - row / column semantics
- [ ] Figures / charts / diagrams
  - captions
  - nearby explanatory text
  - chart labels / legends
  - when text-only extraction loses meaning
- [ ] Internal references
  - hyperlinks
  - citations
  - figure / table references
  - appendices
- [ ] Bounding boxes and layout coordinates
- [ ] Raw-to-structured mapping
- [ ] Canonicalization and normalization
  - encoding cleanup
  - whitespace / boilerplate cleanup
  - normalized section paths
  - stable document and chunk IDs
- [ ] Provenance tracking
  - source file
  - page / span / box
  - parser version
  - ingestion timestamp
- [ ] Metadata schema design
- [ ] Versioning and re-ingestion strategy
- [ ] Deduplication
  - exact
  - near-duplicate
  - semantic duplicate
- [ ] Parsing QA and validation
- [ ] Output formats
  - plain text
  - markdown
  - structured JSON / document model
- [ ] When to keep both raw and normalized forms

📌 _If document understanding is wrong, retrieval starts broken. PDF handling is a first-class RAG skill, not a preprocessing footnote._

---

## **2.2 Chunking (Critical Layer)**

- [ ] Fixed-size chunking (baseline)
- [ ] Sliding window chunking
- [ ] Semantic chunking
- [ ] Hierarchical chunking
- [ ] Structure-aware chunking
  - heading / section aware
  - page-aware
  - table-aware
  - code-block aware
- [ ] Parent-child chunking
- [ ] Small-to-big retrieval patterns
- [ ] Table chunking strategies
- [ ] Figure-caption coupling
- [ ] Formula and code chunk handling
- [ ] Chunk overlap tradeoffs
- [ ] Chunk size vs recall vs cost
- [ ] Chunk-query alignment failure modes
- [ ] Parser/layout-induced chunking errors
- [ ] Chunk-level provenance and parent lineage

📌 _Good chunking preserves meaning and structure. Bad chunking destroys evidence before retrieval ever starts._

---

## **2.3 Representations & Indexing**

- [ ] Sparse indexes (BM25 / inverted index)
- [ ] Dense vector indexes
- [ ] Hybrid representations
- [ ] Multi-vector / late-interaction representations
- [ ] Multimodal / page-level representations for visually rich documents
- [ ] Graph-based representations
- [ ] Vectorless / reasoning-oriented indexes
- [ ] Embedding model selection
- [ ] Domain adaptation of embeddings
- [ ] Multilingual retrieval implications
- [ ] Embedding dimensionality vs quality vs cost
- [ ] Quantization and compression
  - binary
  - product quantization
  - matryoshka / truncation ideas
- [ ] Exact vs approximate search
- [ ] HNSW vs IVF vs DiskANN tradeoffs
- [ ] Metadata and filter schema design
- [ ] Index partitioning / sharding choices
- [ ] Index freshness, refresh, and backfill strategies
- [ ] Dynamic-corpus update paths
- [ ] Hands-on with systems:
  - Pinecone
  - Weaviate
  - PGVector
  - Qdrant
  - Milvus

📌 _“Advanced embedding strategies” belong here, because representation choices determine what retrieval can and cannot see._

---

## **2.4 Retrieval**

- [ ] Vector similarity search
- [ ] BM25 / keyword search
- [ ] Hybrid retrieval and fusion
- [ ] Metadata filtering
- [ ] Query rewriting
- [ ] Query decomposition
- [ ] Multi-query expansion
- [ ] Hypothetical Document Embeddings (HyDE)
- [ ] Query-document mismatch mechanics
  - vocabulary mismatch
  - underspecified queries
  - ambiguous queries
- [ ] Dynamic retrieval depth (`k`)
- [ ] Retrieval abstention (return nothing)
- [ ] Parent-child retrieval
- [ ] In-document reference following
- [ ] Cross-document evidence collection
- [ ] Graph traversal retrieval
- [ ] Vectorless / reasoning-based retrieval
- [ ] Freshness-aware and version-aware retrieval

📌 _Retrieval is not just “search top-k vectors.” It is query understanding, evidence selection, and deciding when not to retrieve._

---

## **2.5 Re-ranking & Context Assembly**

- [ ] Cross-encoder re-ranking
- [ ] Late-interaction re-ranking
- [ ] Score normalization and fusion calibration
- [ ] Context deduplication
- [ ] Diversity vs relevance tradeoffs
- [ ] Token budget allocation
- [ ] Context packing strategies
  - breadth-first vs depth-first
  - local evidence vs global evidence
- [ ] Context compression / selective summarization
- [ ] Lost-in-the-middle mitigation
- [ ] Ordering evidence for answerability
- [ ] Citation grounding and source alignment
- [ ] Final context quality checks before generation

📌 _The generator only sees the context you assemble. Context assembly is where retrieval quality becomes answer quality._

---

## **2.6 Advanced RAG Architectures**

_(Learn these after dense + sparse + hybrid RAG are already clear. These are specialized tools, not your default starting point.)_

- [ ] RAPTOR
  - recursive abstraction trees over documents
  - useful when answers depend on higher-level summaries
- [ ] GraphRAG
  - entity / relation extraction plus graph traversal
  - strongest for global, relationship-heavy corpora
- [ ] Self-RAG
  - retrieve, critique, and revise via self-reflection
  - useful when retrieval decisions should adapt during generation
- [ ] Corrective RAG (CRAG)
  - retrieval quality evaluation and fallback correction
  - useful when retrieval confidence varies widely
- [ ] Multimodal / page-level retrieval
  - ColPali / ColQwen-style page embeddings
  - useful for tables, figures, charts, and visually rich PDFs
- [ ] PageIndex / reasoning-based vectorless retrieval
  - page-structured reasoning instead of pure embedding lookup
  - emerging approach for complex document understanding
- [ ] When not to use these architectures by default

📌 _Study the frontier, but earn it in the right order. Most production systems still win with strong ingestion, hybrid retrieval, reranking, and disciplined evaluation._

---

## **2.7 Failure Modes, Security & Ops**

- [ ] Over-retrieval hallucination
- [ ] Missing-context hallucination
- [ ] Retrieval noise amplification
- [ ] OCR / layout corruption
- [ ] Multi-column reading-order errors
- [ ] Table / figure extraction loss
- [ ] Chunk-boundary loss
- [ ] Stale or partial indexes
- [ ] Authority inversion
- [ ] Metadata / filter bugs
- [ ] Citation drift
- [ ] Prompt injection from retrieved documents
- [ ] Access-control leakage
- [ ] Cross-tenant data exposure
- [ ] Latency blowups and expensive retrieval paths
- [ ] Index refresh race conditions

📌 _This section teaches what can go wrong before you measure it formally in 2(B)._

---

# **PHASE 2(B) — VALIDATING & TRUSTING RAG SYSTEMS**

> Goal: _Prove that the RAG system is correct, grounded, reliable, fresh, and improving over time._

---

## **2.8 Ground Truth & Evaluation Design**

- [ ] Golden query sets
- [ ] Document-answer alignment
- [ ] Query difficulty stratification
- [ ] Task-type stratification
- [ ] Temporal evaluation sets
- [ ] Multilingual and multimodal slices
- [ ] Negative / abstention query sets
- [ ] Access-control and permission-aware test cases

---

## **2.9 Ingestion & Parsing Evaluation**

- [ ] OCR spot checks
- [ ] Reading-order validation
- [ ] Table extraction fidelity
- [ ] Figure-caption linkage checks
- [ ] Structure preservation audits
- [ ] Provenance validation
- [ ] Parser regression suites
- [ ] Raw vs normalized output diff checks
- [ ] Document-level failure labeling

📌 _If parsing quality is unmeasured, retrieval metrics alone can mislead you about where the real failure starts._

---

## **2.10 Retrieval Evaluation**

- [ ] Recall@k
- [ ] Hit@k
- [ ] MRR
- [ ] nDCG
- [ ] Coverage vs precision tradeoffs
- [ ] Hybrid retrieval attribution
- [ ] Filter correctness
- [ ] Page / section / table / figure retrieval accuracy
- [ ] Dynamic-k evaluation
- [ ] Failure-bucket analysis by query type

---

## **2.11 Faithfulness & Answer Correctness**

- [ ] Context-only answering constraints
- [ ] Answer-context attribution
- [ ] Citation accuracy
- [ ] Unsupported claim detection
- [ ] Abstention correctness
- [ ] Partial answer detection
- [ ] Contradictory evidence handling
- [ ] Completeness vs faithfulness tradeoffs

---

## **2.12 Reliability, Latency & Cost**

- [ ] Index drift
- [ ] Data freshness checks
- [ ] Silent degradation detection
- [ ] Regression testing for RAG
- [ ] Token cost modeling
- [ ] Retrieval latency budgets
- [ ] End-to-end latency budgets
- [ ] p95 / p99 behavior
- [ ] Failure under load
- [ ] Cache behavior and invalidation correctness
- [ ] Fallback behavior under degraded retrieval

---

## **2.13 RAG Observability & Debugging**

- [ ] Retrieval trace logging
- [ ] Ingestion and parser trace logging
- [ ] Chunk contribution analysis
- [ ] Source attribution tracking
- [ ] A/B testing for retrieval strategies
- [ ] User feedback integration
- [ ] Retrieval quality dashboards
- [ ] Bad-case triage workflow
- [ ] Root-cause tagging
  - ingestion
  - chunking
  - retrieval
  - re-ranking
  - context assembly

📌 _Observability is what turns “the system feels worse” into a debuggable engineering problem._

---

# FINAL ASSESSMENT (UPDATED)

### 2(A) — Building RAG

**Status:** ✅ Now covers document intelligence, representations, retrieval, reranking, and advanced architectures  
**Quality:** Production-first and complete enough to build real-world RAG systems

### 2(B) — Trusting RAG

**Status:** ✅ Now validates parsing as well as retrieval  
**Quality:** Strong enough to measure whether the system is grounded, fresh, stable, and trustworthy

---

## The Most Important Insight

- Build quality starts before embeddings: if parsing, layout understanding, or structure preservation is wrong, the rest of the pipeline inherits that error.
- Modern RAG is no longer just chunks + vectors: you need to understand sparse, dense, multimodal, graph, and emerging vectorless retrieval patterns, then know when each is worth the extra complexity.

---

