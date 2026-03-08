# What is RAG? (Retrieval-Augmented Generation)

## The Problem RAG Solves

LLMs are powerful, but they have critical limitations:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM LIMITATIONS                          │
│                                                             │
│  1. KNOWLEDGE CUTOFF                                        │
│     LLM was trained on data up to a certain date.           │
│     It doesn't know about anything after that.              │
│                                                             │
│  2. NO ACCESS TO YOUR DATA                                  │
│     It hasn't seen your internal docs, policies, or code.   │
│                                                             │
│  3. HALLUCINATION                                           │
│     When it doesn't know, it makes things up confidently.   │
│                                                             │
│  4. NO CITATIONS                                            │
│     Even when correct, you can't verify WHERE it got info.  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**RAG fixes this by giving the LLM relevant context from YOUR data at query time.**

---

## RAG vs Fine-Tuning vs Prompt Engineering

```
┌────────────────────┬─────────────────────────────────────────┐
│ Approach           │ When to Use                             │
├────────────────────┼─────────────────────────────────────────┤
│ Prompt Engineering │ Simple tasks, no custom data needed     │
│ RAG                │ Need access to specific/changing data   │
│ Fine-Tuning        │ Need to change model behavior/style     │
│ RAG + Fine-Tuning  │ Custom behavior + custom data           │
└────────────────────┴─────────────────────────────────────────┘

KEY INSIGHT:
  - Fine-tuning teaches the model HOW to respond
  - RAG tells the model WHAT to respond with
  - They solve different problems and can be combined
```

---

## The RAG Pipeline — Big Picture

```
USER QUERY: "What's our refund policy for enterprise customers?"
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                      RAG PIPELINE                            │
│                                                              │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │ 1.INGEST│   │2.CHUNK   │   │3.EMBED   │   │4.STORE   │  │
│  │         │   │          │   │          │   │          │  │
│  │Raw docs │──▶│Split into│──▶│Convert to│──▶│Save in   │  │
│  │→ clean  │   │smaller   │   │vectors   │   │vector DB │  │
│  │text     │   │pieces    │   │(numbers) │   │          │  │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘  │
│       ▲              OFFLINE (done once / periodically)      │
│  ─────┼──────────────────────────────────────────────────── │
│       │              ONLINE (every query)                    │
│       │                                                      │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │
│  │5.QUERY  │   │6.RETRIEVE│   │7.RERANK  │   │8.GENERATE│  │
│  │         │   │          │   │          │   │          │  │
│  │User     │──▶│Find      │──▶│Pick best │──▶│LLM makes │  │
│  │question │   │similar   │   │matches   │   │answer    │  │
│  │         │   │chunks    │   │          │   │+ cite    │  │
│  └─────────┘   └──────────┘   └──────────┘   └──────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    │
    ▼
ANSWER: "Enterprise customers can request a full refund within 30 days
         of purchase. [Source: enterprise_policy_v3.pdf, Section 4.2]"
```

---

## Each Stage Explained Simply

### Stage 1: Data Ingestion

**What:** Take raw documents (PDFs, web pages, databases) and clean them.
**Why:** Garbage in → garbage out. Noise in your data = wrong answers.

### Stage 2: Chunking

**What:** Split documents into smaller pieces (chunks).
**Why:** LLMs have context limits. You need to send only the relevant parts, not entire documents.

### Stage 3: Embedding

**What:** Convert each chunk of text into a vector (array of numbers).
**Why:** Vectors let you compute mathematical similarity between text.

### Stage 4: Indexing/Storage

**What:** Store vectors in a vector database (Pinecone, Qdrant, Weaviate, etc.).
**Why:** Enables fast similarity search over millions of vectors.

### Stage 5: Query Processing

**What:** When a user asks a question, process and optionally rewrite it.
**Why:** Users' questions may not match document vocabulary.

### Stage 6: Retrieval

**What:** Find the top-k most similar chunks to the query.
**Why:** This narrows millions of chunks to the ~5-20 most relevant.

### Stage 7: Re-ranking

**What:** Use a more expensive model to re-score and pick the best chunks.
**Why:** Initial retrieval is fast but imprecise. Re-ranking improves precision.

### Stage 8: Generation

**What:** Send the query + retrieved chunks to the LLM to generate an answer.
**Why:** The LLM now has the right context to answer accurately.

---

## When to Use RAG

```
USE RAG WHEN:
  ✅ You need answers grounded in specific documents
  ✅ Your data changes frequently (policies, docs, APIs)
  ✅ You need citations ("Where did this come from?")
  ✅ You have domain-specific data the LLM hasn't seen
  ✅ You need to control what the model can access (security)

DON'T USE RAG WHEN:
  ❌ You need creative/open-ended generation
  ❌ Your task is about style/format, not knowledge
  ❌ All needed knowledge is already in the LLM
  ❌ You only have a handful of documents (just put them in the prompt)
```

---

## Common Beginner Questions

### Q: Why not just put all documents in the LLM's prompt?

**A:** Context windows have limits (and cost money per token). Even with 1M token context, embedding + retrieval gives better accuracy than dumping everything in. The LLM is better at answering when given 5 focused chunks vs 500 pages of text. Research shows LLMs struggle with "lost in the middle" — they pay more attention to the start and end of long contexts.

### Q: Is RAG just semantic search + ChatGPT?

**A:** At its simplest, yes. But production RAG adds cleaning, chunking strategies, hybrid search, re-ranking, evaluation, and monitoring. The gap between a demo and a production system is everything after basic retrieval.

### Q: How is RAG different from a search engine?

**A:** A search engine returns documents. RAG returns **answers** synthesized from documents. The LLM reads the retrieved content and generates a natural language response, not just a ranked list of links.

### Q: Can I use RAG with open-source models?

**A:** Absolutely. RAG works with any LLM — GPT-4, Claude, Llama, Mistral, etc. The retrieval pipeline is model-agnostic. You just swap the generation step.

### Q: What's the minimum I need to build a working RAG system?

**A:**

1. Documents (even just a few text files)
2. An embedding model (e.g., OpenAI's `text-embedding-3-small` or free `all-MiniLM-L6-v2`)
3. A vector store (even FAISS in-memory works for demos)
4. An LLM for generation

---

## Popular Libraries & Frameworks

```
┌──────────────────────────────────────────────────────────────┐
│                  RAG ECOSYSTEM (2024-2026)                    │
│                                                              │
│  FRAMEWORKS (build RAG pipelines):                           │
│  ├── LangChain      — Most popular, huge ecosystem           │
│  ├── LlamaIndex     — Best for data ingestion & indexing     │
│  ├── Haystack       — Production-focused, by deepset         │
│  └── Semantic Kernel — Microsoft's framework                 │
│                                                              │
│  VECTOR DATABASES:                                           │
│  ├── Pinecone       — Managed, easiest to start              │
│  ├── Weaviate       — Open-source, hybrid search built-in    │
│  ├── Qdrant         — Open-source, Rust-based, fast          │
│  ├── Milvus         — Open-source, GPU-accelerated           │
│  ├── ChromaDB       — Simple, great for prototyping          │
│  └── PGVector       — PostgreSQL extension (no new infra)    │
│                                                              │
│  EMBEDDING MODELS:                                           │
│  ├── OpenAI         — text-embedding-3-small/large           │
│  ├── Cohere         — embed-v3                               │
│  ├── sentence-transformers — Free, open-source               │
│  └── Voyage AI      — Domain-optimized embeddings            │
│                                                              │
│  EVALUATION:                                                 │
│  ├── RAGAS          — RAG-specific evaluation framework      │
│  ├── DeepEval       — LLM evaluation with many metrics       │
│  └── LangSmith      — Tracing + evaluation by LangChain      │
│                                                              │
│  DOCUMENT PARSING:                                           │
│  ├── Unstructured   — Multi-format document parsing          │
│  ├── LlamaParse     — LLM-powered document parsing           │
│  └── PyMuPDF        — Fast PDF text extraction               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## What's Next?

Now that you understand the RAG pipeline:

1. **[Embedding Basics](./02_embedding_basics.md)** — Understand how text becomes vectors
2. **[End-to-End Example](./03_rag_pipeline_end_to_end.md)** — Build a working RAG pipeline
3. **[Data Ingestion](../01_data_ingestion/)** — Deep dive into the first stage
