# Capstone: Production RAG with LangChain

## 🟢 How to Approach This

> **Goal:** Build a complete, production-quality RAG system using LangChain's LCEL (LangChain Expression Language). Same quality as the LlamaIndex capstone, different framework — knowing both makes you more versatile.

**Time estimate:** 2–3 hours for full build

**What you need:**

- Python 3.10+
- OpenAI API key
- Some PDF/Markdown documents to index

```bash
pip install langchain langchain-openai langchain-community langchain-qdrant \
    langchain-text-splitters flashrank qdrant-client \
    ragas fastapi uvicorn python-multipart
```

---

## Architecture

```
Documents (PDF/MD/HTML)
    │
    ▼
┌────────────────────┐
│ Document Loaders   │──── PyMuPDFLoader, TextLoader, etc.
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Text Splitter      │──── RecursiveCharacterTextSplitter
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Qdrant VectorStore │──── OpenAI embeddings → Qdrant
└──────┬─────────────┘
       │
       ▼
┌────────────────────────────────────────────┐
│ LCEL Chain                                  │
│                                             │
│ retriever → reranker → prompt → llm → parse│
└──────┬──────────────────────────────────────┘
       │
       ▼
┌────────────────────┐
│ FastAPI             │──── Serve as API
└────────────────────┘
```

---

## Step 1: Project Structure

```
my_rag_lc/
├── app.py              # FastAPI app
├── config.py           # Configuration
├── ingest.py           # Document ingestion
├── chain.py            # LCEL RAG chain
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
config.py
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

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Retrieval
    search_k: int = 10        # initial retrieval
    rerank_top_n: int = 5     # after reranking

    # Vector store
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = "documents_lc"

    # API keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


config = Config()
```

---

## Step 3: Document Ingestion

```python
"""
ingest.py — load, split, embed, and index documents using LangChain.
"""
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from config import config


def load_documents(docs_dir: str = "docs/") -> list:
    """Load documents from directory with format-specific loaders."""
    all_docs = []

    # PDF files
    pdf_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
    )
    all_docs.extend(pdf_loader.load())

    # Markdown files
    md_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
    )
    all_docs.extend(md_loader.load())

    # Text files
    txt_loader = DirectoryLoader(
        docs_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )
    all_docs.extend(txt_loader.load())

    print(f"Loaded {len(all_docs)} documents")
    return all_docs


def split_documents(documents: list) -> list:
    """Split documents into chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_vector_store(chunks: list) -> QdrantVectorStore:
    """Embed chunks and store in Qdrant."""
    embeddings = OpenAIEmbeddings(model=config.embedding_model)

    # Create collection
    client = QdrantClient(url=config.qdrant_url)
    if not client.collection_exists(config.collection_name):
        client.create_collection(
            collection_name=config.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    # Add documents
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=config.qdrant_url,
        collection_name=config.collection_name,
    )

    print(f"Indexed {len(chunks)} chunks in Qdrant")
    return vector_store


def ingest(docs_dir: str = "docs/"):
    """Full ingestion pipeline."""
    documents = load_documents(docs_dir)
    chunks = split_documents(documents)
    vector_store = create_vector_store(chunks)
    return vector_store


if __name__ == "__main__":
    ingest()
    print("Ingestion complete!")
```

---

## Step 4: RAG Chain (LCEL)

```python
"""
chain.py — LCEL RAG chain with reranking.
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import config

# Optional: FlashRank for free local reranking
# pip install flashrank
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False


class RAGChain:
    """Production RAG chain with retrieval, reranking, and generation."""

    def __init__(self):
        # Embeddings
        self.embeddings = OpenAIEmbeddings(model=config.embedding_model)

        # Vector store
        client = QdrantClient(url=config.qdrant_url)
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=config.collection_name,
            embedding=self.embeddings,
        )

        # Retriever
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.search_k},
        )

        # Reranker (FlashRank — free, local, fast)
        self.reranker = None
        if FLASHRANK_AVAILABLE:
            self.reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

        # LLM
        self.llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
        )

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful document assistant. Answer the question based ONLY
on the provided context. If the context doesn't contain the answer, say
"I don't have enough information to answer that."

Always be concise and cite relevant parts of the context."""),
            ("human", """Context:
{context}

Question: {question}"""),
        ])

        # Build chain
        self.chain = self._build_chain()

    def _rerank(self, docs_and_query: dict) -> str:
        """Rerank retrieved documents and format as context string."""
        docs = docs_and_query["docs"]
        query = docs_and_query["question"]

        if self.reranker and docs:
            rerank_request = RerankRequest(
                query=query,
                passages=[{"text": doc.page_content} for doc in docs],
            )
            reranked = self.reranker.rerank(rerank_request)
            # Get top N docs by reranked score
            top_indices = [r["corpus_id"] for r in reranked[:config.rerank_top_n]]
            docs = [docs[i] for i in top_indices]
        else:
            docs = docs[:config.rerank_top_n]

        # Format context
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def _build_chain(self):
        """Build LCEL chain."""
        return (
            {
                "context": {
                    "docs": self.retriever,
                    "question": RunnablePassthrough(),
                } | RunnableLambda(self._rerank),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> dict:
        """Query with full context returned."""
        # Get retrieved docs for evaluation
        docs = self.retriever.invoke(question)

        # Rerank
        if self.reranker and docs:
            rerank_request = RerankRequest(
                query=question,
                passages=[{"text": doc.page_content} for doc in docs],
            )
            reranked = self.reranker.rerank(rerank_request)
            top_indices = [r["corpus_id"] for r in reranked[:config.rerank_top_n]]
            top_docs = [docs[i] for i in top_indices]
        else:
            top_docs = docs[:config.rerank_top_n]

        # Format context
        context = "\n\n---\n\n".join(doc.page_content for doc in top_docs)

        # Generate
        messages = self.prompt.invoke({"context": context, "question": question})
        response = self.llm.invoke(messages)
        answer = response.content

        return {
            "answer": answer,
            "contexts": [doc.page_content for doc in top_docs],
            "sources": [
                {
                    "text": doc.page_content[:200],
                    "metadata": doc.metadata,
                }
                for doc in top_docs
            ],
        }

    def stream(self, question: str):
        """Stream the response."""
        return self.chain.stream(question)


# ─── Usage ───
if __name__ == "__main__":
    rag = RAGChain()
    result = rag.query("What is the return policy?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")
```

---

## Step 5: API Server

```python
"""
app.py — FastAPI server wrapping the LangChain RAG chain.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chain import RAGChain


# Models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class Source(BaseModel):
    text: str
    metadata: dict


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


# App
rag = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = RAGChain()
    print("RAG chain ready")
    yield


app = FastAPI(title="RAG API (LangChain)", lifespan=lifespan)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    try:
        result = rag.query(request.question)
        return QueryResponse(
            answer=result["answer"],
            sources=[Source(**s) for s in result["sources"]],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream the response token by token."""

    async def generate():
        for chunk in rag.stream(request.question):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "healthy", "engine": "langchain"}
```

---

## Step 6: Evaluation

```python
"""
evaluate.py — RAGAS evaluation of the LangChain RAG pipeline.
"""
import json

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

from chain import RAGChain


def run_evaluation(golden_path: str = "tests/eval_data/golden_v1.json"):
    """Run RAGAS evaluation."""
    rag = RAGChain()

    with open(golden_path) as f:
        golden = json.load(f)

    questions, answers, contexts, truths = [], [], [], []

    for item in golden:
        result = rag.query(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
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
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        print(f"  {metric}: {results[metric]:.3f}")

    # Per-sample analysis
    df = results.to_pandas()
    low_faith = df[df["faithfulness"] < 0.7]
    if len(low_faith) > 0:
        print(f"\n⚠️  {len(low_faith)} samples with low faithfulness:")
        for _, row in low_faith.iterrows():
            print(f"  Q: {row['question'][:80]}... → faithfulness: {row['faithfulness']:.3f}")

    return results


if __name__ == "__main__":
    run_evaluation()
```

---

## Step 7: Run It

```bash
# 1. Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# 2. Set API keys
export OPENAI_API_KEY="sk-..."

# 3. Ingest
python ingest.py

# 4. Test chain directly
python chain.py

# 5. Start API
uvicorn app:app --host 0.0.0.0 --port 8000

# 6. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'

# 7. Stream
curl -N http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'

# 8. Evaluate
python evaluate.py
```

---

## LlamaIndex vs LangChain: Side-by-Side

| Aspect             | LlamaIndex                      | LangChain                           |
| ------------------ | ------------------------------- | ----------------------------------- |
| **RAG focus**      | Purpose-built for RAG           | General-purpose LLM framework       |
| **Ingestion**      | `IngestionPipeline` (one-liner) | Manual loader → splitter → embed    |
| **Retrieval**      | Built-in hybrid, auto-merging   | More manual, but flexible           |
| **Chain syntax**   | `index.as_query_engine()`       | LCEL pipe syntax                    |
| **Customization**  | Moderate                        | Very high (every step customizable) |
| **Learning curve** | Easier for RAG                  | Steeper, but more versatile         |
| **When to use**    | RAG is your main use case       | RAG + agents + tools + chains       |

**Interview tip:** Say "I've used both — LlamaIndex is faster to prototype RAG, LangChain gives more control for complex pipelines. I choose based on the project requirements."

---

## Syllabus Mapping

This capstone integrates all sections: §2.1 (Data Ingestion) → §2.2 (Chunking) → §2.3 (Retrieval) → §2.4 (Reranking) → §2.7 (Evaluation) → §2.8 (Production).
