# Deployment Patterns for RAG Systems

## 🟢 How to Approach This Topic

> **Why this matters for your job:** Building a RAG prototype in a notebook is step 1. Deploying it as a reliable service that handles real users is step 2 — and it's where most of the engineering challenge lives. Interviewers want to know you can take a prototype to production.

**Prerequisites:** Have a working RAG pipeline (any approach from earlier sections).

**Reading order:**

1. Architecture overview — which components need to be deployed — 10 min
2. FastAPI service pattern — 15 min
3. LangServe (LangChain deployment) — 10 min
4. Docker + containerization — 10 min
5. Scaling considerations — 10 min

**⏱️ Core concept: 30 min | Full implementation: 2 hours**

---

## Deployment Architecture

```
                       ┌─────────────────────────────────────┐
                       │          Load Balancer               │
                       │        (nginx / ALB / CloudFlare)    │
                       └──────────┬──────────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
             ┌──────────┐  ┌──────────┐  ┌──────────┐
             │ RAG API  │  │ RAG API  │  │ RAG API  │
             │ (FastAPI)│  │ (replica)│  │ (replica)│
             └────┬─────┘  └────┬─────┘  └────┬─────┘
                  │              │              │
         ┌────────┴──────────────┴──────────────┘
         │
    ┌────┼──────────────────────────────┐
    │    ▼                              │
    │  ┌──────────┐  ┌───────────────┐  │
    │  │ Vector   │  │ Redis Cache   │  │
    │  │ DB       │  │ (optional)    │  │
    │  │ (Qdrant/ │  │               │  │
    │  │ PGVector)│  └───────────────┘  │
    │  └──────────┘                     │
    │           Shared Infrastructure   │
    └───────────────────────────────────┘
```

---

## Pattern 1: FastAPI RAG Service

```python
"""
app.py — Production RAG API with FastAPI.
This is the most common deployment pattern.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# These would be your actual RAG components
# from src.rag.pipeline import RAGPipeline


# ─── Models ───

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    collection: str = Field(default="default")


class Source(BaseModel):
    text: str
    metadata: dict
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    vector_db: str
    llm: str


# ─── App ───

# Global pipeline instance (initialized once at startup)
pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize expensive resources once at startup."""
    global pipeline
    # pipeline = RAGPipeline(
    #     vector_store="qdrant://localhost:6333",
    #     embedding_model="text-embedding-3-small",
    #     llm_model="gpt-4o-mini",
    # )
    print("RAG pipeline initialized")
    yield
    # Cleanup
    print("Shutting down")


app = FastAPI(
    title="RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG pipeline."""
    import time
    start = time.time()

    try:
        # result = pipeline.query(
        #     question=request.question,
        #     top_k=request.top_k,
        #     collection=request.collection,
        # )
        result = {
            "answer": "Placeholder answer",
            "sources": [{"text": "source text", "metadata": {}, "score": 0.95}],
        }

        latency = (time.time() - start) * 1000
        return QueryResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
            latency_ms=round(latency, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for load balancer."""
    return HealthResponse(
        status="healthy",
        vector_db="connected",
        llm="available",
    )


# Run: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### Streaming Responses

```python
"""
Streaming is critical for UX — users see partial answers immediately.
"""
from fastapi.responses import StreamingResponse
import json


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream RAG response token by token."""

    async def generate():
        # 1. Retrieve (non-streaming)
        # chunks = pipeline.retrieve(request.question, top_k=request.top_k)

        # 2. Send sources first
        sources_event = {
            "type": "sources",
            "data": [{"text": "chunk...", "score": 0.95}],
        }
        yield f"data: {json.dumps(sources_event)}\n\n"

        # 3. Stream LLM response
        # async for token in pipeline.stream_generate(request.question, chunks):
        for token in ["The ", "answer ", "is ", "streaming."]:
            token_event = {"type": "token", "data": token}
            yield f"data: {json.dumps(token_event)}\n\n"

        # 4. Done
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Pattern 2: LangServe (LangChain Deployment)

```python
"""
LangServe automatically creates a REST API from your LangChain chain.
pip install langserve[all] langchain-openai
"""
from fastapi import FastAPI
from langserve import add_routes
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Build your chain
prompt = ChatPromptTemplate.from_template(
    """Answer based on the context:
    Context: {context}
    Question: {question}
    Answer:"""
)

llm = ChatOpenAI(model="gpt-4o-mini")

# Mock retriever (replace with your actual retriever)
# retriever = vector_store.as_retriever(search_kwargs={"k": 5})

chain = (
    {"context": lambda x: "retrieved context here", "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Create FastAPI app with LangServe
app = FastAPI(title="RAG with LangServe")

add_routes(
    app,
    chain,
    path="/rag",
    # Enables: POST /rag/invoke, POST /rag/stream, POST /rag/batch
    # Also creates: GET /rag/playground (interactive UI)
)

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
# Try: http://localhost:8000/rag/playground
```

---

## Pattern 3: LlamaIndex Deployment

```python
"""
Deploy LlamaIndex query engine as an API.
"""
from fastapi import FastAPI
from pydantic import BaseModel

# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.vector_stores.qdrant import QdrantVectorStore


app = FastAPI(title="RAG with LlamaIndex")


class Query(BaseModel):
    question: str


# index = VectorStoreIndex.from_vector_store(
#     QdrantVectorStore(collection_name="docs", url="http://localhost:6333")
# )
# query_engine = index.as_query_engine(similarity_top_k=5)


@app.post("/query")
async def query(q: Query):
    # response = query_engine.query(q.question)
    response_text = "placeholder"
    source_nodes = []  # response.source_nodes

    return {
        "answer": str(response_text),
        "sources": [
            {
                "text": node.text[:200],
                "score": node.score,
                "metadata": node.metadata,
            }
            for node in source_nodes
        ],
    }
```

---

## Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY app.py .

# Don't run as root
RUN useradd -m appuser
USER appuser

EXPOSE 8000

# Uvicorn with multiple workers
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

```yaml
# docker-compose.yml
# Full stack: API + Vector DB + Cache
version: "3.8"

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
    depends_on:
      - qdrant
      - redis

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  qdrant_data:
```

```bash
# Build and run
docker compose up -d

# Test
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

---

## Scaling Considerations

### Horizontal Scaling

```
                 ┌────────────────┐
                 │  Load Balancer │
                 └───────┬────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ Worker 1 │    │ Worker 2 │    │ Worker 3 │
   │ (CPU)    │    │ (CPU)    │    │ (CPU)    │
   └────┬─────┘    └────┬─────┘    └────┬─────┘
        │               │               │
        └───────────────┬┘               │
                        │                │
                 ┌──────┴────────┐  ┌────┴──────┐
                 │ Vector DB     │  │ LLM API   │
                 │ (scales       │  │ (external,│
                 │ separately)   │  │ no deploy)│
                 └───────────────┘  └───────────┘
```

| Component           | Scaling Strategy               | Notes                          |
| ------------------- | ------------------------------ | ------------------------------ |
| **API workers**     | Horizontal (add replicas)      | Stateless, easy to scale       |
| **Vector DB**       | Vertical first, then shard     | Qdrant Cloud auto-scales       |
| **Embedding model** | Use API (OpenAI) or GPU node   | Batching reduces cost          |
| **LLM**             | Use API (no scaling needed)    | Rate limits are the bottleneck |
| **Cache (Redis)**   | Single node usually sufficient | Cluster for >100K RPM          |

### Latency Budget

```
Typical RAG latency budget (target: <3s total)

  Query embedding:        50-100ms   (API call)
  Vector search:          10-50ms    (in-memory index)
  Reranking:              100-200ms  (cross-encoder)
  LLM generation:         500-2000ms (varies by model)
  Network overhead:       50-100ms
  ─────────────────────────────────
  Total:                  710-2450ms

Optimization levers:
  - Cache frequent queries (skip embedding + search)
  - Use smaller reranker (FlashRank: ~50ms vs cross-encoder: ~200ms)
  - Stream LLM response (perceived latency drops dramatically)
  - Use gpt-4o-mini instead of gpt-4o (~3x faster)
```

---

## Common Pitfalls

| Pitfall                           | Impact                              | Fix                                                          |
| --------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
| Initializing pipeline per request | 5-10s startup overhead per query    | Use lifespan/startup events                                  |
| No health check endpoint          | Load balancer can't detect failures | Add `/health` endpoint                                       |
| Blocking I/O in async handler     | Entire server blocks                | Use `asyncio` or run in thread pool                          |
| Secrets in code                   | Security vulnerability              | Use env vars, secrets manager                                |
| No rate limiting                  | Single user can overwhelm API       | Add rate limits (slowapi, nginx)                             |
| No response timeout               | Hung LLM calls block workers        | Set timeout on LLM calls (30s max)                           |
| Not logging traces                | Can't debug production issues       | Use LangFuse/LangSmith (see `03_observability_debugging.md`) |

---

## 📚 Additional Reading

- [FastAPI docs](https://fastapi.tiangolo.com/) — async Python web framework
- [LangServe docs](https://python.langchain.com/docs/langserve/) — LangChain deployment
- [Uvicorn](https://www.uvicorn.org/) — ASGI server
- [Docker best practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/)

---

## Syllabus Mapping

Maps to `p2_rag_depth.md` §2.8 (Production) and `01_ai_engineering_checklist.md` §5 (Containerization & DevOps).
