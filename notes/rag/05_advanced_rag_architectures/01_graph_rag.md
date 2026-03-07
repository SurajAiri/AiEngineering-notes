# GraphRAG — Knowledge Graph + RAG

## Why It Matters

Standard vector RAG retrieves isolated chunks. If a user asks "How are Drug A and Drug B related through their shared targets?", no single chunk might contain that answer — the connection exists **across documents**. GraphRAG combines knowledge graphs with vector retrieval to answer multi-hop and relational queries.

```
STANDARD RAG — Isolated Chunks:

  Query: "How does Enzyme X relate to Disease Y?"

  Chunk 1: "Enzyme X phosphorylates Protein A"
  Chunk 2: "Protein A regulates Gene B"
  Chunk 3: "Gene B mutations cause Disease Y"

  Vector search might return Chunk 1 and Chunk 3, but
  the connection through Protein A is LOST.

GRAPH RAG — Connected Knowledge:

  Enzyme X ──phosphorylates──▶ Protein A
                                  │
                              regulates
                                  ▼
                               Gene B ──mutated_in──▶ Disease Y

  Graph traversal finds the FULL PATH from Enzyme X to Disease Y.
```

---

## Core Concepts

### Knowledge Graph Basics

```
A knowledge graph stores information as TRIPLES:

  (Subject) ──[Relationship]──▶ (Object)

  Examples:
  ("Python",     "is_a",          "Programming Language")
  ("FastAPI",    "built_with",    "Python")
  ("FastAPI",    "used_for",      "REST APIs")
  ("Kubernetes", "orchestrates",  "Containers")

  Nodes = Entities (Python, FastAPI, Kubernetes)
  Edges = Relationships (is_a, built_with, used_for)
```

### GraphRAG Pipeline

```
                    ┌─────────────┐
                    │  Documents  │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌──────────────────┐    ┌───────────────────┐
    │ Entity Extraction │    │ Chunk + Embed     │
    │ (NER / LLM)      │    │ (standard RAG)    │
    └────────┬─────────┘    └────────┬──────────┘
             │                       │
             ▼                       ▼
    ┌──────────────────┐    ┌───────────────────┐
    │ Relationship     │    │ Vector Store      │
    │ Mapping          │    │ (FAISS / Qdrant)  │
    └────────┬─────────┘    └────────┬──────────┘
             │                       │
             ▼                       │
    ┌──────────────────┐             │
    │ Knowledge Graph  │             │
    │ (Neo4j / NetworkX│)            │
    └────────┬─────────┘             │
             │                       │
             └───────────┬───────────┘
                         ▼
               ┌───────────────────┐
               │ Combined Retrieval│
               │ Graph + Vector    │
               └───────────────────┘
```

---

## Step 1: Entity Extraction & Relationship Mapping

### Simple Version — LLM-based Extraction

```python
"""
Extract entities and relationships from text using an LLM.

Requirements: pip install openai
"""

import json
from openai import OpenAI

client = OpenAI()

EXTRACTION_PROMPT = """Extract entities and relationships from the text below.

Output JSON with this exact structure:
{
  "entities": [
    {"name": "EntityName", "type": "PERSON|ORG|TECH|CONCEPT|PRODUCT"}
  ],
  "relationships": [
    {"source": "Entity1", "target": "Entity2", "relation": "verb_phrase"}
  ]
}

Rules:
- Normalize entity names (e.g., "K8s" → "Kubernetes")
- Use lowercase snake_case for relations (e.g., "is_built_with")
- Only extract clearly stated relationships, not inferences

Text:
{text}
"""


def extract_triples(text: str) -> dict:
    """Extract knowledge graph triples from text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You extract structured knowledge from text."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(response.choices[0].message.content)


# Example
text = """
FastAPI is a modern Python web framework built on top of Starlette and Pydantic.
It was created by Sebastián Ramírez. FastAPI supports automatic API documentation
through Swagger UI and ReDoc. It is commonly used with Uvicorn as the ASGI server.
"""

result = extract_triples(text)
print(json.dumps(result, indent=2))
# {
#   "entities": [
#     {"name": "FastAPI", "type": "TECH"},
#     {"name": "Python", "type": "TECH"},
#     {"name": "Starlette", "type": "TECH"},
#     {"name": "Pydantic", "type": "TECH"},
#     {"name": "Sebastián Ramírez", "type": "PERSON"},
#     ...
#   ],
#   "relationships": [
#     {"source": "FastAPI", "target": "Python", "relation": "is_built_with"},
#     {"source": "FastAPI", "target": "Starlette", "relation": "built_on"},
#     {"source": "Sebastián Ramírez", "target": "FastAPI", "relation": "created"},
#     ...
#   ]
# }
```

---

## Step 2: Build the Knowledge Graph

```python
"""
Build and query a knowledge graph using NetworkX.

Requirements: pip install networkx
"""

import networkx as nx
from dataclasses import dataclass


@dataclass
class Triple:
    source: str
    relation: str
    target: str
    source_chunk: str = ""  # provenance: which chunk this came from


class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_triple(self, triple: Triple):
        """Add a relationship to the graph."""
        # Add nodes with type info
        self.graph.add_node(triple.source)
        self.graph.add_node(triple.target)

        # Add edge with relationship and provenance
        self.graph.add_edge(
            triple.source,
            triple.target,
            relation=triple.relation,
            source_chunk=triple.source_chunk,
        )

    def add_triples_from_extraction(self, extraction: dict, chunk_text: str = ""):
        """Add all triples from an LLM extraction result."""
        for rel in extraction.get("relationships", []):
            self.add_triple(Triple(
                source=rel["source"],
                relation=rel["relation"],
                target=rel["target"],
                source_chunk=chunk_text[:200],
            ))

    def get_neighbors(self, entity: str, depth: int = 1) -> list[Triple]:
        """Get all relationships within N hops of an entity."""
        if entity not in self.graph:
            return []

        triples = []
        visited = set()
        queue = [(entity, 0)]

        while queue:
            node, current_depth = queue.pop(0)
            if current_depth >= depth or node in visited:
                continue
            visited.add(node)

            # Outgoing edges
            for _, target, data in self.graph.out_edges(node, data=True):
                triples.append(Triple(
                    source=node,
                    relation=data["relation"],
                    target=target,
                    source_chunk=data.get("source_chunk", ""),
                ))
                queue.append((target, current_depth + 1))

            # Incoming edges
            for source, _, data in self.graph.in_edges(node, data=True):
                triples.append(Triple(
                    source=source,
                    relation=data["relation"],
                    target=node,
                    source_chunk=data.get("source_chunk", ""),
                ))
                queue.append((source, current_depth + 1))

        return triples

    def find_path(self, source: str, target: str) -> list[Triple] | None:
        """Find the shortest path between two entities."""
        try:
            path_nodes = nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Try undirected
            try:
                path_nodes = nx.shortest_path(
                    self.graph.to_undirected(), source, target
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return None

        triples = []
        for i in range(len(path_nodes) - 1):
            s, t = path_nodes[i], path_nodes[i + 1]
            if self.graph.has_edge(s, t):
                data = self.graph.edges[s, t]
                triples.append(Triple(s, data["relation"], t))
            elif self.graph.has_edge(t, s):
                data = self.graph.edges[t, s]
                triples.append(Triple(t, data["relation"], s))
        return triples

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "components": nx.number_weakly_connected_components(self.graph),
        }


# ─── Build example graph ───
kg = KnowledgeGraph()

triples = [
    Triple("FastAPI", "is_built_with", "Python"),
    Triple("FastAPI", "built_on", "Starlette"),
    Triple("FastAPI", "uses", "Pydantic"),
    Triple("Starlette", "is_a", "ASGI Framework"),
    Triple("Pydantic", "validates", "Data Models"),
    Triple("Django", "is_built_with", "Python"),
    Triple("Django", "uses", "Django ORM"),
    Triple("Flask", "is_built_with", "Python"),
]

for t in triples:
    kg.add_triple(t)

# Query: What's related to FastAPI?
print("=== Neighbors of FastAPI (depth=1) ===")
for t in kg.get_neighbors("FastAPI", depth=1):
    print(f"  {t.source} --[{t.relation}]--> {t.target}")

# Query: How are Pydantic and Python connected?
print("\n=== Path: Pydantic → Python ===")
path = kg.find_path("Pydantic", "Python")
if path:
    for t in path:
        print(f"  {t.source} --[{t.relation}]--> {t.target}")

print(f"\nGraph stats: {kg.stats()}")
```

---

## Step 3: Combined Graph + Vector Retrieval

```python
"""
GraphRAG retriever: combines vector search with graph traversal.

Requirements: pip install sentence-transformers numpy networkx faiss-cpu
"""

import numpy as np
import faiss
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer


@dataclass
class GraphRAGResult:
    """A retrieved result combining vector and graph sources."""
    text: str
    score: float
    source: str  # "vector" | "graph" | "both"
    graph_context: list[str] = field(default_factory=list)  # related triples


class GraphRAGRetriever:
    """
    Retrieval that combines:
    1. Vector similarity search (standard RAG)
    2. Graph traversal (for relational / multi-hop queries)
    """

    def __init__(
        self,
        kg: "KnowledgeGraph",
        model_name: str = "all-MiniLM-L6-v2",
        graph_depth: int = 2,
    ):
        self.kg = kg
        self.model = SentenceTransformer(model_name)
        self.graph_depth = graph_depth
        self.chunks: list[str] = []
        self.index = None

    def ingest(self, chunks: list[str]):
        """Index chunks for vector search."""
        self.chunks = chunks
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def retrieve(
        self,
        query: str,
        k: int = 5,
        use_graph: bool = True,
    ) -> list[GraphRAGResult]:
        """
        1. Vector search for top-k chunks
        2. Extract entities from query
        3. Traverse graph around those entities
        4. Combine and deduplicate results
        """
        results = []

        # ── Vector search ──
        if self.index is not None:
            query_emb = self.model.encode(query, normalize_embeddings=True)
            scores, indices = self.index.search(
                query_emb.reshape(1, -1).astype(np.float32), k
            )
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.chunks):
                    results.append(GraphRAGResult(
                        text=self.chunks[idx],
                        score=float(score),
                        source="vector",
                    ))

        # ── Graph traversal ──
        if use_graph:
            # Extract mentioned entities (simple: check which graph nodes
            # appear in the query)
            query_lower = query.lower()
            mentioned_entities = [
                node for node in self.kg.graph.nodes
                if node.lower() in query_lower
            ]

            graph_triples = []
            for entity in mentioned_entities:
                triples = self.kg.get_neighbors(entity, depth=self.graph_depth)
                graph_triples.extend(triples)

            # Convert triples to text context
            if graph_triples:
                triple_texts = list({
                    f"{t.source} → [{t.relation}] → {t.target}"
                    for t in graph_triples
                })

                # Add graph context as a synthesized chunk
                graph_text = "Related knowledge:\n" + "\n".join(triple_texts)
                results.append(GraphRAGResult(
                    text=graph_text,
                    score=0.5,  # fixed score for graph results
                    source="graph",
                    graph_context=triple_texts,
                ))

                # Also enrich vector results with graph context
                for result in results:
                    if result.source == "vector":
                        result.graph_context = triple_texts[:3]

        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]


# ─── Full Example ───
if __name__ == "__main__":
    # Build knowledge graph
    kg = KnowledgeGraph()
    triples = [
        Triple("FastAPI", "is_built_with", "Python"),
        Triple("FastAPI", "built_on", "Starlette"),
        Triple("FastAPI", "uses", "Pydantic"),
        Triple("FastAPI", "supports", "async/await"),
        Triple("Starlette", "is_a", "ASGI Framework"),
        Triple("Pydantic", "validates", "Data Models"),
        Triple("Python", "has_framework", "Django"),
        Triple("Python", "has_framework", "Flask"),
    ]
    for t in triples:
        kg.add_triple(t)

    # Build retriever
    retriever = GraphRAGRetriever(kg)
    chunks = [
        "FastAPI is a modern web framework for building APIs with Python.",
        "Starlette provides the ASGI foundation for FastAPI.",
        "Pydantic handles data validation and serialization in FastAPI.",
        "Django is a full-stack web framework with ORM and admin panel.",
        "Flask is a lightweight WSGI framework for Python.",
    ]
    retriever.ingest(chunks)

    # Query that benefits from graph
    results = retriever.retrieve("What is FastAPI built on?", k=3)
    for r in results:
        print(f"[{r.source}] score={r.score:.3f}: {r.text[:80]}")
        if r.graph_context:
            for gc in r.graph_context:
                print(f"    graph: {gc}")
```

---

## When to Use GraphRAG vs Standard RAG

```
USE GRAPH RAG WHEN:
  ✅ Multi-hop reasoning: "How are A and C connected?"
  ✅ Relational queries: "What depends on X?"
  ✅ Entity-centric domains: biomedical, legal, financial
  ✅ Cross-document connections matter
  ✅ Users ask about relationships, not just facts

USE STANDARD RAG WHEN:
  ✅ Factual lookups: "What is the timeout setting?"
  ✅ Single-hop queries: answer is in one chunk
  ✅ Fast iteration speed needed
  ✅ Data doesn't have strong relational structure
```

---

## Pitfalls & Common Mistakes

| Mistake                                   | Impact                                   | Fix                                                          |
| ----------------------------------------- | ---------------------------------------- | ------------------------------------------------------------ |
| **Entity extraction quality is poor**     | Garbage graph, wrong relationships       | Use LLM with few-shot examples; validate extractions         |
| **Graph too sparse**                      | No useful connections found              | Ensure enough documents cover related topics                 |
| **Graph too dense**                       | Too much irrelevant context returned     | Limit traversal depth; filter by relationship type           |
| **Ignoring entity normalization**         | "K8s" and "Kubernetes" as separate nodes | Canonicalize entity names before insertion                   |
| **No provenance on graph edges**          | Can't trace back to source document      | Store source_chunk on every edge                             |
| **Building graph upfront for everything** | Massive cost, slow iteration             | Start with the subdomain where relational queries are common |

---

## Key Takeaways

1. **GraphRAG fills the multi-hop gap** — connects facts across documents that vector search can't.
2. **Entity extraction quality is the bottleneck** — bad entities = bad graph.
3. **Combine, don't replace** — use graph results alongside vector results.
4. **Start small** — graph a focused domain first, not your entire corpus.
5. **Neo4j for production, NetworkX for prototyping** — choose the right tool for scale.
