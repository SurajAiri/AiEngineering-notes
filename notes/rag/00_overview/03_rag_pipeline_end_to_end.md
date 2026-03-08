# End-to-End RAG Pipeline — Working Examples

> Build a complete RAG system from scratch. Two versions: one with LangChain, one with LlamaIndex.

## Minimal RAG — No Framework (Understand the Fundamentals)

```python
"""
Minimal RAG pipeline with no framework.
Understand what's happening before using LangChain/LlamaIndex.

Requirements: pip install openai sentence-transformers faiss-cpu numpy
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ── Step 1: Your documents ──
documents = [
    "Our return policy allows full refunds within 30 days of purchase with a receipt.",
    "Enterprise customers get dedicated support with a 4-hour SLA response time.",
    "The API rate limit is 1000 requests per minute per API key.",
    "Password reset can be done via Settings > Security > Reset Password.",
    "Pricing starts at $29/month for the Starter plan, $99/month for Pro.",
]

# ── Step 2: Create embeddings ──
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embed_model.encode(documents, normalize_embeddings=True)

# ── Step 3: Store in vector index (FAISS) ──
dimension = doc_embeddings.shape[1]  # 384
index = faiss.IndexFlatIP(dimension)  # Inner product (= cosine for normalized vectors)
index.add(doc_embeddings.astype('float32'))

# ── Step 4: Query ──
query = "How much does the Pro plan cost?"
query_embedding = embed_model.encode([query], normalize_embeddings=True)

# ── Step 5: Retrieve top-k similar chunks ──
k = 3
scores, indices = index.search(query_embedding.astype('float32'), k)

retrieved_chunks = []
for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
    retrieved_chunks.append(documents[idx])
    print(f"  [{i+1}] (score={score:.3f}) {documents[idx][:80]}...")

# ── Step 6: Generate answer with LLM ──
context = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(retrieved_chunks))

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Answer based ONLY on the provided context. Cite sources using [1], [2], etc."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ],
    temperature=0,
)

print(f"\nAnswer: {response.choices[0].message.content}")
```

---

## LangChain Version

```python
"""
RAG pipeline using LangChain — the most popular RAG framework.

Requirements: pip install langchain langchain-openai langchain-community
              pip install faiss-cpu sentence-transformers
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Step 1: Load and chunk documents ──
raw_text = """
# Company FAQ

## Return Policy
Our return policy allows full refunds within 30 days of purchase with a receipt.
After 30 days, store credit is offered. Enterprise customers have a 60-day window.

## API Rate Limits
The API rate limit is 1000 requests per minute per API key.
Enterprise API keys have a limit of 10,000 requests per minute.
Rate limit headers are included in every response.

## Pricing
Pricing starts at $29/month for the Starter plan.
The Pro plan costs $99/month and includes priority support.
Enterprise pricing is custom — contact sales.

## Support
Standard support: 24-hour response time via email.
Enterprise customers get dedicated support with a 4-hour SLA response time.
"""

# Chunk the text
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separators=["\n\n", "\n", ". ", " "],
)
chunks = splitter.split_text(raw_text)
print(f"Created {len(chunks)} chunks")

# ── Step 2: Embed and store ──
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_texts(chunks, embeddings)

# ── Step 3: Create retrieval chain ──
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

prompt = PromptTemplate.from_template("""
Answer the question based ONLY on the following context.
If the context doesn't contain the answer, say "I don't have that information."
Cite your sources.

Context:
{context}

Question: {question}

Answer:""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True,
)

# ── Step 4: Query ──
result = qa_chain.invoke({"query": "What's the rate limit for enterprise?"})
print(f"Answer: {result['result']}")
print(f"Sources: {[doc.page_content[:50] for doc in result['source_documents']]}")
```

---

## LlamaIndex Version

```python
"""
RAG pipeline using LlamaIndex — best for document-heavy applications.

Requirements: pip install llama-index llama-index-llms-openai
              pip install llama-index-embeddings-openai
"""

from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ── Step 1: Configure ──
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-small")
Settings.chunk_size = 200
Settings.chunk_overlap = 30

# ── Step 2: Load documents ──
documents = [
    Document(text="Our return policy allows full refunds within 30 days of purchase with a receipt. After 30 days, store credit is offered."),
    Document(text="The API rate limit is 1000 requests per minute per API key. Enterprise API keys have a limit of 10,000 requests per minute."),
    Document(text="Pricing starts at $29/month for Starter. Pro is $99/month. Enterprise pricing is custom."),
    Document(text="Enterprise customers get dedicated support with a 4-hour SLA response time."),
]

# ── Step 3: Build index (embeds + stores automatically) ──
index = VectorStoreIndex.from_documents(documents)

# ── Step 4: Query ──
query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query("What's the rate limit for enterprise?")

print(f"Answer: {response}")
print(f"Sources: {[node.text[:50] for node in response.source_nodes]}")
```

---

## Comparing the Frameworks

```
┌──────────────────┬──────────────────────────┬──────────────────────────┐
│                  │ LangChain                │ LlamaIndex               │
├──────────────────┼──────────────────────────┼──────────────────────────┤
│ Best for         │ General LLM apps,        │ Document-heavy RAG,      │
│                  │ complex chains           │ structured data          │
├──────────────────┼──────────────────────────┼──────────────────────────┤
│ Learning curve   │ Medium (many concepts)   │ Lower (focused API)      │
├──────────────────┼──────────────────────────┼──────────────────────────┤
│ Chunking control │ Manual (TextSplitters)   │ Built-in (Settings)      │
├──────────────────┼──────────────────────────┼──────────────────────────┤
│ Ecosystem        │ Huge (many integrations) │ Focused (data/retrieval) │
├──────────────────┼──────────────────────────┼──────────────────────────┤
│ Production use   │ Very common              │ Growing fast             │
├──────────────────┼──────────────────────────┼──────────────────────────┤
│ Recommended for  │ Multi-step agents +      │ Pure RAG systems,        │
│                  │ RAG combinations         │ document pipelines       │
└──────────────────┴──────────────────────────┴──────────────────────────┘

Both are great choices. Pick one and learn it well.
Start with LangChain if you're unsure — it has the most community resources.
```

---

## Common Questions

### Q: Do I need a framework at all?

**A:** No. The "no framework" version above works perfectly. Frameworks save time on boilerplate and provide integrations (vector DBs, embedding providers, etc.). For learning, start without a framework. For production, frameworks help with reliability and maintainability.

### Q: Can I use a free/local LLM instead of OpenAI?

**A:** Yes. Replace OpenAI with Ollama + Llama/Mistral for fully local RAG:

```python
# LangChain with Ollama
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")

# LlamaIndex with Ollama
from llama_index.llms.ollama import Ollama
Settings.llm = Ollama(model="llama3")
```

### Q: Where does the data actually get stored?

**A:** In the examples above, everything is in-memory (RAM). For production you'd use a persistent vector database like Pinecone, Qdrant, or Weaviate. The retrieval logic stays the same — only the storage backend changes.

---

## What's Next?

Now you know how the full pipeline works. The rest of these notes deep-dive into each stage:

1. **[Data Ingestion](../01_data_ingestion/)** — How to clean and prepare documents (the hardest part in practice)
2. **[Chunking](../02_chunking/)** — How to split documents into good retrieval units
3. **[Retrieval](../03_retrieval/)** — How to find the right chunks for a query
