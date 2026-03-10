# Capstone: Production RAG with LlamaIndex

## 🟢 How to Approach This

> **Goal:** Build a complete, production-quality RAG system using LlamaIndex. Start from raw documents, end with a deployed API that you can show in interviews or use as a template at work.

**Time estimate:** 2–3 hours for full build

**What you need:**

- Python 3.10+
- OpenAI API key (or any supported LLM)
- Some PDF/Markdown documents to index

```bash
pip install llama-index llama-index-vector-stores-qdrant \
    llama-index-embeddings-openai llama-index-llms-openai \
    llama-index-postprocessor-cohere-rerank \
    qdrant-client ragas fastapi uvicorn
```

---

## Architecture

```
Documents (PDF/MD/HTML)
    │
    ▼
┌──────────────┐
│ SimpleDirectoryReader │──── Load & parse
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ IngestionPipeline │──── Transform → chunk → embed
│ • SentenceSplitter │
│ • OpenAIEmbedding  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ QdrantVectorStore │──── Store embeddings
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ VectorStoreIndex │──── Query engine + reranker
│ + CohereRerank   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ FastAPI       │──── Serve as API
└──────────────┘
```

---

## Step 1: Project Structure

```
my_rag/
├── app.py              # FastAPI app
├── config.py           # Configuration
├── ingest.py           # Document ingestion
├── query_engine.py     # Query engine setup
├── evaluate.py         # RAGAS evaluation
├── docs/               # Your documents
│   ├── guide.pdf
│   └── faq.md
└── tests/
    └── eval_data/
        └── golden_v1.json
```

---

## Step 2: Configuration

```python
"""
config.py — centralized configuration.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # LLM
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    similarity_top_k: int = 10
    rerank_top_n: int = 5

    # Vector store
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = "documents"

    # API keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    cohere_api_key: str = os.getenv("COHERE_API_KEY", "")


config = Config()
```

---

## Step 3: Document Ingestion

```python
"""
ingest.py — load, chunk, embed, and index documents.
"""
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import config


def create_vector_store() -> QdrantVectorStore:
    """Connect to Qdrant and create/get collection."""
    client = QdrantClient(url=config.qdrant_url)
    return QdrantVectorStore(
        client=client,
        collection_name=config.collection_name,
        enable_hybrid=True,  # enables BM25 + vector hybrid search
    )


def ingest_documents(docs_dir: str = "docs/") -> VectorStoreIndex:
    """
    Full ingestion pipeline:
    1. Load documents from directory
    2. Split into chunks
    3. Embed chunks
    4. Store in Qdrant
    """
    # 1. Load
    print(f"Loading documents from {docs_dir}...")
    reader = SimpleDirectoryReader(
        input_dir=docs_dir,
        recursive=True,
        required_exts=[".pdf", ".md", ".txt", ".html"],
    )
    documents = reader.load_data()
    print(f"  Loaded {len(documents)} documents")

    # 2. Create ingestion pipeline
    vector_store = create_vector_store()
    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            ),
            OpenAIEmbedding(model=config.embedding_model),
        ],
        vector_store=vector_store,
    )

    # 3. Run pipeline (chunks, embeds, and stores)
    print("Running ingestion pipeline...")
    nodes = pipeline.run(documents=documents, show_progress=True)
    print(f"  Indexed {len(nodes)} chunks")

    # 4. Create index from vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    return index


if __name__ == "__main__":
    index = ingest_documents()
    print("Ingestion complete!")
```

---

## Step 4: Query Engine

```python
"""
query_engine.py — build the query engine with reranking.
"""
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import config

# Optional: Cohere reranker (better quality, requires API key)
# from llama_index.postprocessor.cohere_rerank import CohereRerank


def build_query_engine():
    """
    Build production query engine with:
    - Hybrid search (vector + BM25)
    - Reranking (cross-encoder)
    - Configured LLM
    """
    # Configure global settings
    Settings.llm = OpenAI(model=config.llm_model, temperature=config.llm_temperature)
    Settings.embed_model = OpenAIEmbedding(model=config.embedding_model)

    # Connect to existing vector store
    client = QdrantClient(url=config.qdrant_url)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=config.collection_name,
        enable_hybrid=True,
    )

    # Build index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    # Reranker (local, free — no API key needed)
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=config.rerank_top_n,
    )

    # Alternative: Cohere reranker (better quality)
    # reranker = CohereRerank(
    #     api_key=config.cohere_api_key,
    #     top_n=config.rerank_top_n,
    # )

    # Build query engine
    query_engine = index.as_query_engine(
        similarity_top_k=config.similarity_top_k,
        node_postprocessors=[reranker],
        response_mode="compact",  # compact context into fewer LLM calls
    )

    return query_engine


# Custom prompt (optional but recommended)
from llama_index.core import PromptTemplate

QA_PROMPT = PromptTemplate(
    """You are a helpful document assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information to answer that."

Context:
{context_str}

Question: {query_str}

Answer:"""
)


def build_query_engine_with_custom_prompt():
    """Query engine with custom prompt template."""
    engine = build_query_engine()
    engine.update_prompts({"response_synthesizer:text_qa_template": QA_PROMPT})
    return engine
```

---

## Step 5: API Server

```python
"""
app.py — FastAPI server wrapping the query engine.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from query_engine import build_query_engine_with_custom_prompt


# Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceNode(BaseModel):
    text: str
    score: float
    file_name: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceNode]


# App
engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = build_query_engine_with_custom_prompt()
    print("Query engine ready")
    yield


app = FastAPI(title="RAG API (LlamaIndex)", lifespan=lifespan)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        response = engine.query(request.question)

        sources = []
        for node in response.source_nodes:
            sources.append(SourceNode(
                text=node.text[:500],
                score=node.score or 0.0,
                file_name=node.metadata.get("file_name", "unknown"),
            ))

        return QueryResponse(answer=str(response), sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "llamaindex"}
```

---

## Step 6: Evaluation

```python
"""
evaluate.py — evaluate the pipeline with RAGAS.
"""
import json

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from query_engine import build_query_engine_with_custom_prompt


def run_evaluation(golden_path: str = "tests/eval_data/golden_v1.json"):
    """Run RAGAS evaluation against golden test set."""
    engine = build_query_engine_with_custom_prompt()

    with open(golden_path) as f:
        golden = json.load(f)

    questions, answers, contexts, truths = [], [], [], []

    for item in golden:
        response = engine.query(item["question"])
        questions.append(item["question"])
        answers.append(str(response))
        contexts.append([node.text for node in response.source_nodes])
        truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": truths,
    })

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    print("\n📊 Evaluation Results:")
    print(f"  Faithfulness:       {results['faithfulness']:.3f}")
    print(f"  Answer Relevancy:   {results['answer_relevancy']:.3f}")
    print(f"  Context Precision:  {results['context_precision']:.3f}")
    print(f"  Context Recall:     {results['context_recall']:.3f}")

    return results


if __name__ == "__main__":
    run_evaluation()
```

---

## Step 7: Run It

```bash
# 1. Start Qdrant (docker)
docker run -d -p 6333:6333 qdrant/qdrant

# 2. Set API keys
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="..."  # optional, for Cohere reranker

# 3. Ingest documents
python ingest.py

# 4. Test query engine
python -c "from query_engine import build_query_engine_with_custom_prompt; e = build_query_engine_with_custom_prompt(); print(e.query('What is the return policy?'))"

# 5. Start API
uvicorn app:app --host 0.0.0.0 --port 8000

# 6. Test API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'

# 7. Run evaluation
python evaluate.py
```

---

## What to Highlight in Interviews

When discussing this project, emphasize:

1. **Hybrid search** — "I use both vector similarity and BM25 keyword search for better recall."
2. **Reranking** — "I retrieve 10 candidates, then rerank to top 5 with a cross-encoder."
3. **Evaluation** — "I evaluate with RAGAS metrics — faithfulness, relevancy, precision, recall."
4. **Production patterns** — "Documents are ingested through a pipeline, served via FastAPI, with health checks."
5. **Observability** — "I can trace every query through the pipeline to debug issues."

---

## Syllabus Mapping

This capstone integrates all sections: §2.1 (Data Ingestion) → §2.2 (Chunking) → §2.3 (Retrieval) → §2.4 (Reranking) → §2.7 (Evaluation) → §2.8 (Production).
