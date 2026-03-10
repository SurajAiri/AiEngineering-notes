# Retrieval with Libraries — Practical Cookbook

## 🟢 How to Approach This Topic

> **Why this matters for your job:** Retrieval is the heart of RAG. In production, you'll use LangChain retrievers, LlamaIndex query engines, or direct vector DB clients — not raw FAISS code. This cookbook shows how to build retrieval pipelines with production libraries, including hybrid search, ensemble retrieval, and auto-merging patterns.

**Prerequisites:** Read [01_vector_similarity_search.md](./01_vector_similarity_search.md) and [03_hybrid_retrieval.md](./03_hybrid_retrieval.md) to understand the underlying concepts.

**Reading order:**

1. LangChain retriever examples (most common in industry)
2. LlamaIndex retriever examples
3. Hybrid & ensemble retrieval patterns
4. Advanced patterns (auto-merging, query fusion)

**⏱️ Core concept: 1 hour | Full exploration: 3 hours**

---

## Retrieval Architectures at a Glance

```
                        ┌─────────────┐
                        │  User Query │
                        └──────┬──────┘
                               │
                    ┌──────────▼──────────┐
                    │   Query Processing  │
                    │  (rewrite, expand,  │
                    │   decompose)        │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        ┌─────▼─────┐   ┌─────▼─────┐   ┌──────▼─────┐
        │  Dense     │   │  Sparse   │   │  Metadata  │
        │  (Vector)  │   │  (BM25)   │   │  (Filter)  │
        └─────┬─────┘   └─────┬─────┘   └──────┬─────┘
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │       Fusion        │
                    │   (RRF, weighted)   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     Re-ranking      │
                    │  (cross-encoder)    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Top-K Results      │
                    └─────────────────────┘
```

---

## LangChain — Retrieval Patterns

### Basic Vector Retriever

```python
"""
The simplest retrieval pattern: embed → search → return top-k.
"""
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Assume documents are already loaded
# docs = loader.load()

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=50,
)
chunks = splitter.split_documents(docs)

# Create vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Basic retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",       # or "mmr" for diversity
    search_kwargs={"k": 5},
)

# Search
results = retriever.invoke("What is the authentication flow?")
for doc in results:
    print(f"[{doc.metadata.get('source', '?')}] {doc.page_content[:200]}")
```

### MMR Retriever (Maximal Marginal Relevance — Diversity)

```python
"""
MMR balances relevance with diversity.
Prevents returning 5 near-identical chunks.
"""
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,                # return 5 results
        "fetch_k": 20,         # consider top 20 candidates
        "lambda_mult": 0.7,    # 0 = max diversity, 1 = max relevance
    },
)

results = retriever.invoke("How does the payment system work?")
```

### BM25 Retriever (Keyword Search)

```python
"""
Keyword-based retrieval using BM25.
Better than vector search for exact term matching.
"""
from langchain_community.retrievers import BM25Retriever

# Create from documents (not a vectorstore — it's in-memory)
bm25_retriever = BM25Retriever.from_documents(
    chunks,
    k=5,
)

results = bm25_retriever.invoke("OAuth2 client_credentials grant type")
# BM25 excels when users search for specific terms/acronyms
```

### Ensemble Retriever (Hybrid Search)

```python
"""
Combine vector + BM25 using Reciprocal Rank Fusion (RRF).
This is the production standard for most RAG systems.
"""
from langchain.retrievers import EnsembleRetriever

# Vector retriever
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# BM25 retriever
bm25_retriever = BM25Retriever.from_documents(chunks, k=10)

# Ensemble with equal weights
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5],  # 50% vector, 50% BM25
)

results = ensemble_retriever.invoke("payment processing timeout errors")
print(f"Got {len(results)} results from hybrid search")
```

### Multi-Query Retriever (Query Expansion)

```python
"""
Generates multiple query variations using an LLM,
retrieves for each, then merges results. Handles vague/ambiguous queries.
"""
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm,
)

# "How does auth work?" might generate:
# 1. "What is the authentication mechanism?"
# 2. "How are users authenticated in the system?"
# 3. "What authentication protocols are used?"
results = multi_query_retriever.invoke("How does auth work?")
```

### Contextual Compression (Retriever + Reranker)

```python
"""
Retrieve broadly, then compress/filter using a reranker.
Reduces noise while keeping relevant passages.
"""
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Cross-encoder reranker
model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=model, top_n=5)

# Retrieve 20, rerank to top 5
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

results = compression_retriever.invoke("What are the SLA requirements?")
```

### Self-Query Retriever (Natural Language Metadata Filtering)

```python
"""
Converts natural language queries into structured filters.
"Show me Python tutorials from 2024" → filter: language=Python, year=2024
"""
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(name="source", description="The document source file", type="string"),
    AttributeInfo(name="year", description="Year the document was published", type="integer"),
    AttributeInfo(name="category", description="Document category", type="string"),
    AttributeInfo(name="language", description="Programming language", type="string"),
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
    vectorstore=vectorstore,
    document_contents="Technical documentation and tutorials",
    metadata_field_info=metadata_field_info,
)

# Automatically generates metadata filter from query
results = retriever.invoke("Python tutorials published after 2024")
```

---

## LlamaIndex — Retrieval Patterns

### Basic VectorStoreIndex Query

```python
"""
LlamaIndex wraps retrieval + generation in one query engine.
"""
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.llm = OpenAI(model="gpt-4o-mini")

# Load and index
documents = SimpleDirectoryReader("docs/").load_data()
index = VectorStoreIndex.from_documents(documents)

# Query (retrieval + generation)
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact",  # or "tree_summarize", "refine"
)
response = query_engine.query("What is the authentication flow?")
print(response)

# Just retrieval (no generation)
retriever = index.as_retriever(similarity_top_k=5)
nodes = retriever.retrieve("What is the authentication flow?")
for node in nodes:
    print(f"Score: {node.score:.3f} | {node.text[:200]}")
```

### Hybrid Search (BM25 + Vector)

```python
"""
LlamaIndex hybrid retrieval using QueryFusionRetriever.
"""
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever

# Build BM25 retriever from the same nodes
bm25_retriever = BM25Retriever.from_defaults(
    nodes=nodes,  # same nodes used for vector index
    similarity_top_k=10,
)

# Vector retriever
vector_retriever = index.as_retriever(similarity_top_k=10)

# Fusion retriever (RRF by default)
hybrid_retriever = QueryFusionRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    similarity_top_k=5,
    num_queries=1,       # set to 4 for multi-query expansion
    mode="reciprocal_rerank",  # RRF
)

results = hybrid_retriever.retrieve("OAuth2 client_credentials")
for r in results:
    print(f"Score: {r.score:.3f} | {r.text[:150]}")
```

### Auto-Merging Retriever (Hierarchical Expansion)

```python
"""
Retrieves small chunks, then automatically expands to parent
if enough children match. Best of both: precision + context.
"""
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.retrievers import AutoMergingRetriever

# 1. Create hierarchical nodes
node_parser = HierarchicalNodeParser.from_defaults(
    chunk_sizes=[2048, 512, 128],
)
nodes = node_parser.get_nodes_from_documents(documents)
leaf_nodes = get_leaf_nodes(nodes)

# 2. Store all nodes (parents + children)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

# 3. Index only leaf nodes (small, precise)
index = VectorStoreIndex(leaf_nodes, storage_context=storage_context)

# 4. Auto-merging retriever expands to parent when appropriate
base_retriever = index.as_retriever(similarity_top_k=12)
retriever = AutoMergingRetriever(
    base_retriever,
    storage_context,
    simple_ratio_thresh=0.4,  # merge if 40%+ children match
)

results = retriever.retrieve("Explain the authentication architecture")
for r in results:
    print(f"Score: {r.score:.3f} | Length: {len(r.text)} | {r.text[:150]}")
```

---

## Vector Database Retrievers

### Qdrant

```python
"""
Qdrant: high-performance vector DB with advanced filtering.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer

# Connect
client = QdrantClient(url="http://localhost:6333")  # or ":memory:" for testing
model = SentenceTransformer("all-MiniLM-L6-v2")

# With LangChain
from langchain_qdrant import QdrantVectorStore

vectorstore = QdrantVectorStore.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    url="http://localhost:6333",
    collection_name="my_docs",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### Pinecone

```python
"""
Pinecone: managed vector DB, scales to billions.
"""
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_API_KEY")

# With LangChain
from langchain_pinecone import PineconeVectorStore

vectorstore = PineconeVectorStore.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    index_name="my-index",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

### PGVector (PostgreSQL)

```python
"""
PGVector: vector search in PostgreSQL. Great for existing Postgres users.
"""
from langchain_postgres import PGVector

CONNECTION_STRING = "postgresql+psycopg://user:pass@localhost:5432/mydb"

vectorstore = PGVector.from_documents(
    chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    connection=CONNECTION_STRING,
    collection_name="my_docs",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
```

---

## Full RAG Chain (Retrieval + Generation)

### LangChain LCEL Chain

```python
"""
Complete RAG chain using LangChain Expression Language (LCEL).
This is the modern LangChain pattern.
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Setup
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context.
If the context doesn't contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


# LCEL chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Query
answer = rag_chain.invoke("What are the SLA requirements?")
print(answer)
```

### LlamaIndex Query Engine

```python
"""
LlamaIndex handles retrieval + generation in one call.
"""
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)

# Simple query engine
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="compact",
)
response = query_engine.query("What are the SLA requirements?")

print(response)
print("\nSources:")
for node in response.source_nodes:
    print(f"  Score: {node.score:.3f} | {node.node.metadata}")
```

---

## Common Pitfalls

| Pitfall                             | Impact                                           | Fix                                  |
| ----------------------------------- | ------------------------------------------------ | ------------------------------------ |
| Only using vector search            | Misses exact keyword matches (acronyms, IDs)     | Use hybrid (vector + BM25) retrieval |
| Too low k (top_k=3)                 | Missing relevant docs                            | Start with k=10, rerank to 5         |
| Too high k (top_k=50)               | Noise overwhelms signal                          | Cap at 10-20, use reranker           |
| Not using MMR for diversity         | Getting 5 near-identical chunks                  | Use `search_type="mmr"`              |
| Ignoring metadata filtering         | Searching all docs when user wants specific ones | Use self-query or manual filtering   |
| Same embedding model for everything | Domain mismatch                                  | Evaluate models on your data first   |

---

## 📚 Additional Reading

- [LangChain Retrievers docs](https://python.langchain.com/docs/how_to/#retrievers) — Full list of retriever types
- [LlamaIndex Retrievers](https://docs.llamaindex.ai/en/stable/module_guides/querying/retriever/) — Retriever module guide
- [Chroma docs](https://docs.trychroma.com/) — Simple vector DB for prototyping
- See also: [12_retrieval_routing.md](./12_retrieval_routing.md) for query-aware retrieval strategy selection

---

## Syllabus Mapping

Maps to **§2.3** in `p2_rag_depth.md`. This cookbook complements the concept files (01–11) by showing production library usage for retrieval patterns.
