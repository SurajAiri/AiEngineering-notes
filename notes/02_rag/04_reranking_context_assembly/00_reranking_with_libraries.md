# Re-ranking with Libraries — Practical Cookbook

## 🟢 How to Approach This Topic

> **Why this matters for your job:** Retrieval gets you candidates; re-ranking picks the winners. In production, you'll use Cohere Rerank, FlashRank, or cross-encoder models through LangChain/LlamaIndex — not raw PyTorch code. This cookbook shows how.

**Prerequisites:** Read [01_cross_encoder_reranking.md](./01_cross_encoder_reranking.md) to understand why re-ranking matters.

**Reading order:**

1. Understand the re-ranking concept (5 min review)
2. Try FlashRank (free, local, fast) — 10 min
3. Try Cohere Rerank (best quality, paid API) — 10 min
4. LangChain/LlamaIndex integration — 20 min
5. Compare and decide — 10 min

**⏱️ Core concept: 30 min | Full exploration: 1.5 hours**

---

## Re-ranking in the RAG Pipeline

```
Query: "How does payment processing handle timeouts?"
                    │
                    ▼
        ┌───────────────────────┐
        │ Retrieve Top 20       │  ← Broad recall (cheap, fast)
        │ (Vector + BM25)       │
        └───────────┬───────────┘
                    │
          20 candidate chunks
                    │
                    ▼
        ┌───────────────────────┐
        │ Re-rank with          │  ← Precise relevance (slower, smarter)
        │ Cross-Encoder         │
        │ (query, chunk) pairs  │
        └───────────┬───────────┘
                    │
          Top 5 most relevant
                    │
                    ▼
        ┌───────────────────────┐
        │ Pack into LLM Context │
        └───────────────────────┘
```

**Key insight:** Retrieve many (cheap), re-rank few (expensive). Bi-encoders are fast but imprecise. Cross-encoders are slow but accurate. Use both.

---

## Option 1: FlashRank (Free, Local, Fast)

```python
"""
FlashRank: lightweight, fast, no API key needed.
Great for prototyping and cost-sensitive production.
pip install flashrank
"""
from flashrank import Ranker, RerankRequest

# Initialize (downloads model on first use)
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

# Prepare passages
query = "How does payment processing handle timeouts?"
passages = [
    {"id": 1, "text": "Payment timeouts occur when the gateway doesn't respond within 30 seconds. The system retries up to 3 times with exponential backoff."},
    {"id": 2, "text": "The user authentication flow uses OAuth2 with PKCE for mobile clients."},
    {"id": 3, "text": "When a payment request times out, the order is placed in a 'pending' state and a background job verifies the payment status."},
    {"id": 4, "text": "Our monitoring dashboard tracks p95 latency across all API endpoints."},
    {"id": 5, "text": "Timeout handling in the payment module follows the circuit breaker pattern to prevent cascade failures."},
]

# Re-rank
request = RerankRequest(query=query, passages=passages)
results = ranker.rerank(request)

for r in results:
    print(f"Score: {r['score']:.4f} | {r['text'][:100]}")

# Output (re-ranked by relevance):
# Score: 0.9847 | Payment timeouts occur when the gateway doesn't respond...
# Score: 0.9621 | Timeout handling in the payment module follows...
# Score: 0.9234 | When a payment request times out, the order...
# Score: 0.0012 | Our monitoring dashboard tracks p95 latency...
# Score: 0.0003 | The user authentication flow uses OAuth2...
```

### FlashRank Model Options

| Model                     | Size  | Speed     | Quality         | Use Case                  |
| ------------------------- | ----- | --------- | --------------- | ------------------------- |
| `ms-marco-MiniLM-L-12-v2` | 33MB  | ⚡ Fast   | ⭐⭐⭐ Good     | Default, general purpose  |
| `ms-marco-MultiBERT-L-12` | 167MB | 🔄 Medium | ⭐⭐⭐⭐ Better | When quality matters more |
| `rank-T5-flan`            | 220MB | 🐌 Slower | ⭐⭐⭐⭐⭐ Best | Maximum quality, local    |

---

## Option 2: Cohere Rerank (Best Quality, Paid API)

```python
"""
Cohere Rerank: highest quality, simple API.
pip install cohere
"""
import cohere

co = cohere.Client(api_key="YOUR_COHERE_API_KEY")

query = "How does payment processing handle timeouts?"
documents = [
    "Payment timeouts occur when the gateway doesn't respond within 30 seconds.",
    "The user authentication flow uses OAuth2 with PKCE.",
    "When a payment request times out, the order enters a 'pending' state.",
    "Our monitoring dashboard tracks p95 latency.",
    "Timeout handling follows the circuit breaker pattern.",
]

response = co.rerank(
    model="rerank-v3.5",        # latest model
    query=query,
    documents=documents,
    top_n=3,                    # return top 3
    return_documents=True,
)

for result in response.results:
    print(f"Score: {result.relevance_score:.4f} | Index: {result.index}")
    print(f"  {result.document.text[:100]}")
```

### Cohere Pricing (as of 2025)

| Model                      | Cost               | Latency | Best For                      |
| -------------------------- | ------------------ | ------- | ----------------------------- |
| `rerank-v3.5`              | ~$0.002 per search | ~200ms  | Production, multilingual      |
| `rerank-english-v3.0`      | ~$0.002 per search | ~150ms  | English-only, slightly faster |
| `rerank-multilingual-v3.0` | ~$0.002 per search | ~200ms  | Multi-language support        |

---

## Option 3: Cross-Encoder with sentence-transformers (Free, Flexible)

```python
"""
Direct cross-encoder usage. Full control, free, no API.
pip install sentence-transformers
"""
from sentence_transformers import CrossEncoder

# Load model
model = CrossEncoder("BAAI/bge-reranker-v2-m3")  # multilingual, high quality
# Alternatives:
# "cross-encoder/ms-marco-MiniLM-L-6-v2"  — fast, English
# "cross-encoder/ms-marco-MiniLM-L-12-v2" — balanced
# "BAAI/bge-reranker-base"                 — good quality

query = "How does payment processing handle timeouts?"
passages = [
    "Payment timeouts occur when the gateway doesn't respond within 30 seconds.",
    "The user authentication flow uses OAuth2 with PKCE.",
    "Timeout handling follows the circuit breaker pattern.",
]

# Score each (query, passage) pair
pairs = [(query, passage) for passage in passages]
scores = model.predict(pairs)

# Sort by score
ranked = sorted(zip(scores, passages), reverse=True)
for score, passage in ranked:
    print(f"Score: {score:.4f} | {passage[:100]}")
```

---

## LangChain Integration

### ContextualCompressionRetriever + CrossEncoder

```python
"""
LangChain's built-in pattern: retrieve broadly, compress with reranker.
"""
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1. Base retriever (broad, cheap)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# 2. Reranker (precise, slower)
cross_encoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)

# 3. Combined retriever
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# Use in chain
results = retriever.invoke("How does payment processing handle timeouts?")
for doc in results:
    print(f"{doc.page_content[:200]}")
```

### With Cohere Rerank

```python
"""
Cohere reranker in LangChain.
pip install langchain-cohere
"""
from langchain_cohere import CohereRerank

compressor = CohereRerank(
    model="rerank-v3.5",
    top_n=5,
)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

results = retriever.invoke("How does payment processing handle timeouts?")
```

### Full RAG Chain with Re-ranking

```python
"""
Complete RAG chain: retrieve → rerank → generate.
"""
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Setup reranking retriever (from above)
reranking_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_template("""
Answer based only on the context below. Cite the source for each claim.

Context:
{context}

Question: {question}

Answer:""")


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


rag_chain = (
    {"context": reranking_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("How does payment processing handle timeouts?")
print(answer)
```

---

## LlamaIndex Integration

### SentenceTransformerRerank

```python
"""
LlamaIndex built-in reranker node postprocessor.
"""
from llama_index.core.postprocessor import SentenceTransformerRerank

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-v2-m3",
    top_n=5,
)

# Use with query engine
query_engine = index.as_query_engine(
    similarity_top_k=20,             # retrieve 20
    node_postprocessors=[reranker],  # rerank to top 5
)
response = query_engine.query("How does payment processing handle timeouts?")
```

### Cohere Rerank in LlamaIndex

```python
"""
Cohere reranker as a LlamaIndex node postprocessor.
pip install llama-index-postprocessor-cohere-rerank
"""
from llama_index.postprocessor.cohere_rerank import CohereRerank

reranker = CohereRerank(
    api_key="YOUR_COHERE_API_KEY",
    model="rerank-v3.5",
    top_n=5,
)

query_engine = index.as_query_engine(
    similarity_top_k=20,
    node_postprocessors=[reranker],
)
response = query_engine.query("How does payment processing handle timeouts?")
```

---

## Decision Framework

```
Should I use a reranker?
    │
    ├── Retrieval quality is already good (>0.9 recall@5)?
    │   └── NO reranker needed. Don't add complexity.
    │
    ├── On a tight budget?
    │   └── FlashRank (free, local, fast)
    │
    ├── Need best possible quality?
    │   └── Cohere Rerank API (paid, highest quality)
    │
    ├── Need multilingual support?
    │   └── BAAI/bge-reranker-v2-m3 (free) or Cohere (paid)
    │
    └── Default recommendation?
        └── Start with FlashRank for prototyping.
            Move to Cohere for production if budget allows.
            Cross-encoder (sentence-transformers) is the middle ground.
```

---

## Common Pitfalls

| Pitfall                                      | Impact                              | Fix                                                       |
| -------------------------------------------- | ----------------------------------- | --------------------------------------------------------- |
| Re-ranking all retrieved docs (k=100+)       | Very slow, O(n) cross-encoder calls | Retrieve top 20-50, rerank to top 5                       |
| Ignoring max sequence length                 | Long chunks get truncated silently  | Ensure chunks fit within model limit (512 tokens typical) |
| Not batching cross-encoder calls             | Sequential = 10x slower             | Use batch_size=32+                                        |
| Using reranker without measuring improvement | Adding latency for no gain          | Compare recall@5 with/without reranker                    |
| Reranker model mismatch                      | English model on non-English data   | Use multilingual model (bge-reranker-v2-m3)               |

---

## Syllabus Mapping

Maps to **§2.4** in `p2_rag_depth.md`. This cookbook complements the concept files (01–04) by showing production library usage for re-ranking.
