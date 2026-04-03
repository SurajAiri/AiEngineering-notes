# Vector Database Hands-On: Pinecone, Weaviate, PGVector, Qdrant, Milvus

## Why It Matters

Building a RAG prototype with FAISS is easy. Moving to production requires a **managed or self-hosted vector database** that handles persistence, scaling, filtering, updates, and monitoring. Each DB has different strengths — this guide gives you working code for each.

---

## Quick Comparison

```
┌──────────────┬───────────┬─────────┬─────────────┬───────────────┐
│              │ Hosting   │ Filter  │ Hybrid      │ Best for      │
│              │           │ Support │ Search      │               │
├──────────────┼───────────┼─────────┼─────────────┼───────────────┤
│ Pinecone     │ Managed   │ ✅ Rich │ ✅ Sparse+  │ Quickest to   │
│              │ (cloud)   │         │    Dense    │ production    │
├──────────────┼───────────┼─────────┼─────────────┼───────────────┤
│ Qdrant       │ Both      │ ✅ Rich │ ✅ Built-in │ Feature-rich  │
│              │           │         │             │ self-hosted   │
├──────────────┼───────────┼─────────┼─────────────┼───────────────┤
│ Weaviate     │ Both      │ ✅ Rich │ ✅ BM25+vec │ Schema-first  │
│              │           │         │             │ approach      │
├──────────────┼───────────┼─────────┼─────────────┼───────────────┤
│ pgvector     │ Self &    │ ✅ SQL  │ ⚠️ Manual   │ Already using │
│              │ managed   │ power   │             │ PostgreSQL    │
├──────────────┼───────────┼─────────┼─────────────┼───────────────┤
│ Milvus       │ Both      │ ✅ Rich │ ✅ Built-in │ Large-scale   │
│              │ (Zilliz)  │         │             │ deployments   │
└──────────────┴───────────┴─────────┴─────────────┴───────────────┘
```

---

## 1. Pinecone

```python
"""
Pinecone — fully managed, serverless option available.

pip install pinecone-client sentence-transformers
"""

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# ─── Setup ───
pc = Pinecone(api_key="YOUR_API_KEY")
model = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_NAME = "rag-demo"

# Create index (serverless)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(INDEX_NAME)

# ─── Ingest ───
documents = [
    {"id": "doc-1", "text": "Kubernetes uses RBAC for access control",
     "metadata": {"source": "k8s-docs", "topic": "security", "year": 2024}},
    {"id": "doc-2", "text": "HPA scales pods based on CPU metrics",
     "metadata": {"source": "k8s-docs", "topic": "scaling", "year": 2024}},
    {"id": "doc-3", "text": "Network policies control pod traffic",
     "metadata": {"source": "k8s-docs", "topic": "security", "year": 2023}},
]

# Batch upsert
vectors = []
for doc in documents:
    embedding = model.encode(doc["text"]).tolist()
    vectors.append({
        "id": doc["id"],
        "values": embedding,
        "metadata": {**doc["metadata"], "text": doc["text"]},
    })

index.upsert(vectors=vectors)

# ─── Search with metadata filter ───
query = "access control in kubernetes"
query_vec = model.encode(query).tolist()

results = index.query(
    vector=query_vec,
    top_k=3,
    include_metadata=True,
    filter={
        "topic": {"$eq": "security"},
        "year": {"$gte": 2024},
    },
)

for match in results.matches:
    print(f"  [{match.score:.4f}] {match.metadata['text']}")
```

---

## 2. Qdrant

```python
"""
Qdrant — feature-rich, runs locally or in cloud.

pip install qdrant-client sentence-transformers
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue, Range,
)
from sentence_transformers import SentenceTransformer

# ─── Setup ───
client = QdrantClient(url="http://localhost:6333")  # or use :memory: for testing
# client = QdrantClient(":memory:")  # in-memory for quick testing
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION = "rag-demo"

# Create collection
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# ─── Ingest ───
documents = [
    {"id": 1, "text": "Kubernetes uses RBAC for access control",
     "metadata": {"source": "k8s-docs", "topic": "security", "year": 2024}},
    {"id": 2, "text": "HPA scales pods based on CPU metrics",
     "metadata": {"source": "k8s-docs", "topic": "scaling", "year": 2024}},
    {"id": 3, "text": "Network policies control pod traffic",
     "metadata": {"source": "k8s-docs", "topic": "security", "year": 2023}},
]

points = []
for doc in documents:
    embedding = model.encode(doc["text"]).tolist()
    points.append(PointStruct(
        id=doc["id"],
        vector=embedding,
        payload={"text": doc["text"], **doc["metadata"]},
    ))

client.upsert(collection_name=COLLECTION, points=points)

# ─── Search with filter ───
query = "access control in kubernetes"
query_vec = model.encode(query).tolist()

results = client.search(
    collection_name=COLLECTION,
    query_vector=query_vec,
    limit=3,
    query_filter=Filter(
        must=[
            FieldCondition(key="topic", match=MatchValue(value="security")),
            FieldCondition(key="year", range=Range(gte=2024)),
        ]
    ),
)

for r in results:
    print(f"  [{r.score:.4f}] {r.payload['text']}")
```

---

## 3. Weaviate

```python
"""
Weaviate — schema-first, built-in hybrid search.

pip install weaviate-client sentence-transformers
"""

import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import MetadataQuery, Filter

# ─── Setup ───
client = weaviate.connect_to_local()  # or weaviate.connect_to_wcs(...)

# Create collection (schema-first approach)
collection = client.collections.create(
    name="Document",
    properties=[
        Property(name="text", data_type=DataType.TEXT),
        Property(name="source", data_type=DataType.TEXT),
        Property(name="topic", data_type=DataType.TEXT),
        Property(name="year", data_type=DataType.INT),
    ],
    vectorizer_config=Configure.Vectorizer.none(),  # we provide our own
)

# ─── Ingest ───
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    {"text": "Kubernetes uses RBAC for access control",
     "source": "k8s-docs", "topic": "security", "year": 2024},
    {"text": "HPA scales pods based on CPU metrics",
     "source": "k8s-docs", "topic": "scaling", "year": 2024},
]

collection = client.collections.get("Document")
with collection.batch.dynamic() as batch:
    for doc in documents:
        vector = model.encode(doc["text"]).tolist()
        batch.add_object(properties=doc, vector=vector)

# ─── Search ───
query = "access control in kubernetes"
query_vec = model.encode(query).tolist()

results = collection.query.near_vector(
    near_vector=query_vec,
    limit=3,
    filters=Filter.by_property("topic").equal("security"),
    return_metadata=MetadataQuery(distance=True),
)

for obj in results.objects:
    print(f"  [{obj.metadata.distance:.4f}] {obj.properties['text']}")

client.close()
```

---

## 4. pgvector (PostgreSQL)

```python
"""
pgvector — vector search inside PostgreSQL.
Best when you already use PostgreSQL and want to avoid a separate DB.

pip install psycopg2-binary pgvector sentence-transformers

PostgreSQL setup:
  CREATE EXTENSION vector;
"""

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import numpy as np

# ─── Setup ───
conn = psycopg2.connect(
    host="localhost", port=5432,
    dbname="ragdb", user="postgres", password="postgres",
)
register_vector(conn)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Create table
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            source TEXT,
            topic TEXT,
            year INTEGER,
            embedding vector(384)
        );

        CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    """)
    conn.commit()

# ─── Ingest ───
documents = [
    ("Kubernetes uses RBAC for access control", "k8s-docs", "security", 2024),
    ("HPA scales pods based on CPU metrics", "k8s-docs", "scaling", 2024),
    ("Network policies control pod traffic", "k8s-docs", "security", 2023),
]

with conn.cursor() as cur:
    for text, source, topic, year in documents:
        embedding = model.encode(text).tolist()
        cur.execute(
            """INSERT INTO documents (text, source, topic, year, embedding)
               VALUES (%s, %s, %s, %s, %s)""",
            (text, source, topic, year, embedding),
        )
    conn.commit()

# ─── Search with SQL filters ───
query = "access control in kubernetes"
query_vec = model.encode(query).tolist()

with conn.cursor() as cur:
    cur.execute(
        """SELECT text, 1 - (embedding <=> %s::vector) AS similarity
           FROM documents
           WHERE topic = %s AND year >= %s
           ORDER BY embedding <=> %s::vector
           LIMIT 3""",
        (query_vec, "security", 2024, query_vec),
    )

    for text, sim in cur.fetchall():
        print(f"  [{sim:.4f}] {text}")

conn.close()
```

---

## 5. Milvus

```python
"""
Milvus — designed for billion-scale vector search.
Zilliz Cloud for managed, self-hosted via Docker.

pip install pymilvus sentence-transformers
"""

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility,
)
from sentence_transformers import SentenceTransformer

# ─── Setup ───
connections.connect("default", host="localhost", port="19530")
model = SentenceTransformer("all-MiniLM-L6-v2")

COLLECTION = "rag_demo"

# Drop if exists
if utility.has_collection(COLLECTION):
    utility.drop_collection(COLLECTION)

# Define schema
schema = CollectionSchema(fields=[
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="year", dtype=DataType.INT64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
])

collection = Collection(COLLECTION, schema)

# ─── Ingest ───
documents = [
    {"text": "Kubernetes uses RBAC for access control", "topic": "security", "year": 2024},
    {"text": "HPA scales pods based on CPU metrics", "topic": "scaling", "year": 2024},
    {"text": "Network policies control pod traffic", "topic": "security", "year": 2023},
]

texts = [d["text"] for d in documents]
embeddings = model.encode(texts).tolist()

collection.insert([
    texts,
    [d["topic"] for d in documents],
    [d["year"] for d in documents],
    embeddings,
])

# Build index
collection.create_index(
    field_name="embedding",
    index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
)
collection.load()

# ─── Search with filter ───
query = "access control in kubernetes"
query_vec = model.encode(query).tolist()

results = collection.search(
    data=[query_vec],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"ef": 128}},
    limit=3,
    expr='topic == "security" and year >= 2024',
    output_fields=["text", "topic", "year"],
)

for hits in results:
    for hit in hits:
        print(f"  [{hit.distance:.4f}] {hit.entity.get('text')}")
```

---

## Decision Guide

```
START HERE:

  Already using PostgreSQL?
  ├── Yes → pgvector (simplest integration)
  └── No
      │
      Want managed (no ops)?
      ├── Yes → Pinecone (or Zilliz Cloud for Milvus)
      └── No
          │
          Scale?
          ├── <10M vectors → Qdrant (best features/perf ratio)
          ├── 10M-1B → Milvus (designed for scale)
          └── Need hybrid search? → Weaviate or Qdrant (built-in BM25)
```

---

## Key Takeaways

1. **pgvector** is the easiest if you already use PostgreSQL — no new infrastructure.
2. **Pinecone** is the fastest to production — fully managed, no ops.
3. **Qdrant** has the best feature set for self-hosted — rich filtering, hybrid search.
4. **Milvus** is designed for scale — use it for billion-vector deployments.
5. **Weaviate** is best if you want schema-first design with built-in vectorization.
6. **All support metadata filtering** — the API differs, the concept is the same.
