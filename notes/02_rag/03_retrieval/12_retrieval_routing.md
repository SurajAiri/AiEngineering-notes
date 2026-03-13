# Retrieval Routing — Query-Aware Strategy Selection

## 🟢 How to Approach This Topic

> **Why this matters for your job:** Not every query should hit vector search. Some need keyword search, some need metadata filters, some need multi-hop retrieval, and some need no retrieval at all. A retrieval router decides which strategy to use per query — and this routing layer is often **more important than embedding quality**.

**Prerequisites:** Read [01_vector_similarity_search.md](./01_vector_similarity_search.md), [02_bm25_keyword_search.md](./02_bm25_keyword_search.md), and [03_hybrid_retrieval.md](./03_hybrid_retrieval.md).

**Reading order:**

1. Understand why routing matters (concept)
2. Study query types and which retrieval strategy fits
3. Simple rule-based router (beginner)
4. LLM-based router (production)
5. LangChain/LlamaIndex router patterns

**⏱️ Core concept: 45 min | Full exploration: 2 hours**

---

## Why Routing Matters

```
WITHOUT ROUTING:
┌──────────┐     ┌──────────────┐     ┌──────────┐
│  Query   │────▶│ Vector Search│────▶│ Results  │
└──────────┘     └──────────────┘     └──────────┘
                   (one size fits all)

PROBLEM: "What is error code E-4012?" → Vector search finds semantically
         similar but wrong results. BM25 would nail it instantly.


WITH ROUTING:
┌──────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────┐
│  Query   │────▶│ Query Classifier │────▶│ Best Strategy│────▶│ Results  │
└──────────┘     └──────────────────┘     └──────────────┘     └──────────┘
                         │
                    ┌────┼────┬────────┬───────────┐
                    │    │    │        │           │
                 Vector BM25 Hybrid  Metadata   No Search
                 Search      Search  Filter     (LLM only)
```

---

## Query Types and Matching Strategies

| Query Type              | Example                     | Best Strategy            | Why                          |
| ----------------------- | --------------------------- | ------------------------ | ---------------------------- |
| **Factual lookup**      | "What is HNSW?"             | Vector search            | Semantic matching works well |
| **Exact term / code**   | "Error E-4012"              | BM25 / keyword           | Exact match needed           |
| **Scoped question**     | "Python docs from 2024"     | Metadata filter + vector | Narrow scope first           |
| **Broad / exploratory** | "How does auth work?"       | Hybrid (vector + BM25)   | Cast wide net                |
| **Multi-hop**           | "Compare auth in v1 vs v2"  | Multi-query expansion    | Needs multiple retrievals    |
| **Conversational**      | "What about the other one?" | Memory + retrieval       | Needs context resolution     |
| **Simple / general**    | "What is machine learning?" | No retrieval (LLM only)  | LLM knows this already       |
| **Temporal**            | "Latest release notes"      | Metadata filter (date)   | Time-based filtering         |

---

## Simple Code — Rule-Based Router

```python
"""
Rule-based retrieval router. Good starting point.
Routes queries to the best retrieval strategy using heuristics.
"""
import re
from enum import Enum


class RetrievalStrategy(Enum):
    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    METADATA_FILTER = "metadata_filter"
    NO_RETRIEVAL = "no_retrieval"
    MULTI_QUERY = "multi_query"


def classify_query(query: str) -> RetrievalStrategy:
    """Route query to the best retrieval strategy using heuristics."""
    query_lower = query.lower().strip()

    # Pattern: exact codes, IDs, error messages → BM25
    if re.search(r'[A-Z]{1,5}[-_]\d{3,}', query):
        return RetrievalStrategy.BM25
    if re.search(r'"[^"]{3,}"', query):  # quoted exact phrase
        return RetrievalStrategy.BM25

    # Pattern: temporal queries → metadata filter
    temporal_words = ["latest", "recent", "last week", "yesterday",
                      "this month", "updated", "newest", "2024", "2025", "2026"]
    if any(word in query_lower for word in temporal_words):
        return RetrievalStrategy.METADATA_FILTER

    # Pattern: comparison / multi-entity → multi-query
    comparison_words = ["compare", "difference between", "vs", "versus",
                        "how does .* differ", "contrast"]
    if any(re.search(pattern, query_lower) for pattern in comparison_words):
        return RetrievalStrategy.MULTI_QUERY

    # Pattern: very simple / general knowledge → no retrieval
    general_patterns = [
        r"^what is [a-z\s]{3,20}\?$",  # "What is machine learning?"
        r"^define [a-z\s]{3,20}$",
        r"^explain [a-z\s]{3,20}$",
    ]
    if any(re.search(p, query_lower) for p in general_patterns):
        return RetrievalStrategy.NO_RETRIEVAL

    # Default: hybrid (safest general-purpose choice)
    return RetrievalStrategy.HYBRID


# Test
test_queries = [
    "What is error code E-4012?",          # → BM25
    "How does authentication work?",        # → HYBRID
    "Compare REST vs GraphQL",              # → MULTI_QUERY
    "Latest release notes",                 # → METADATA_FILTER
    "What is machine learning?",            # → NO_RETRIEVAL
    'Find the function "calculate_tax"',    # → BM25
]

for q in test_queries:
    strategy = classify_query(q)
    print(f"  {strategy.value:18s} ← {q}")
```

---

## Production Code — LLM-Based Router

```python
"""
LLM-based query classification for retrieval routing.
More accurate than rules, handles edge cases better.
"""
from openai import OpenAI
import json

client = OpenAI()

ROUTER_PROMPT = """You are a retrieval strategy router. Given a user query,
classify which retrieval strategy would work best.

Strategies:
- "vector": Semantic/conceptual questions where meaning matters more than exact words
- "bm25": Exact term lookup, error codes, specific function names, quoted phrases
- "hybrid": Broad questions that benefit from both semantic + keyword matching
- "metadata_filter": Time-scoped, category-scoped, or source-scoped queries
- "multi_query": Comparison questions, multi-hop reasoning, complex queries
- "no_retrieval": Simple general knowledge the LLM already knows

Respond with JSON: {"strategy": "<strategy>", "reason": "<brief reason>"}

Query: {query}"""


def route_query_llm(query: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": ROUTER_PROMPT.format(query=query)}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# Test
result = route_query_llm("What is error code E-4012?")
print(result)
# {"strategy": "bm25", "reason": "Query contains a specific error code that needs exact matching"}
```

---

## Full Router with Strategy Execution

```python
"""
Complete retrieval router: classify → execute → return results.
Production pattern combining routing with actual retrieval.
"""
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    documents: list
    strategy_used: str
    query_original: str
    query_modified: str | None = None


class RetrievalRouter:
    """Route queries to optimal retrieval strategies."""

    def __init__(self, vector_retriever, bm25_retriever, ensemble_retriever,
                 metadata_retriever=None, llm=None):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.ensemble_retriever = ensemble_retriever
        self.metadata_retriever = metadata_retriever
        self.llm = llm

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """Route and execute retrieval."""
        strategy = self._classify(query)

        if strategy == "no_retrieval":
            return RetrievalResult(
                documents=[],
                strategy_used="no_retrieval",
                query_original=query,
            )

        retriever_map = {
            "vector": self.vector_retriever,
            "bm25": self.bm25_retriever,
            "hybrid": self.ensemble_retriever,
            "metadata_filter": self.metadata_retriever or self.ensemble_retriever,
            "multi_query": self.ensemble_retriever,  # use multi-query expansion
        }

        retriever = retriever_map.get(strategy, self.ensemble_retriever)
        docs = retriever.invoke(query)[:k]

        return RetrievalResult(
            documents=docs,
            strategy_used=strategy,
            query_original=query,
        )

    def _classify(self, query: str) -> str:
        """Classify query into retrieval strategy."""
        # Use rules first (fast, free)
        rule_result = classify_query(query)  # from above

        # For ambiguous cases, use LLM
        if rule_result == RetrievalStrategy.HYBRID and self.llm:
            llm_result = route_query_llm(query)
            return llm_result["strategy"]

        return rule_result.value


# Usage with LangChain:
# router = RetrievalRouter(
#     vector_retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
#     bm25_retriever=BM25Retriever.from_documents(chunks, k=10),
#     ensemble_retriever=EnsembleRetriever(
#         retrievers=[vector_retriever, bm25_retriever], weights=[0.5, 0.5]
#     ),
# )
# result = router.retrieve("What is error code E-4012?")
# print(f"Strategy: {result.strategy_used}, Docs: {len(result.documents)}")
```

---

## With Libraries

### LlamaIndex RouterQueryEngine

```python
"""
LlamaIndex has built-in query routing to different indexes/engines.
"""
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool

# Create different query engines for different data
vector_engine = vector_index.as_query_engine(similarity_top_k=5)
keyword_engine = keyword_index.as_query_engine()
summary_engine = summary_index.as_query_engine(response_mode="tree_summarize")

# Define tools (engines with descriptions)
tools = [
    QueryEngineTool.from_defaults(
        query_engine=vector_engine,
        description="Best for specific factual questions about the documentation.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=keyword_engine,
        description="Best for queries with exact terms, error codes, or specific names.",
    ),
    QueryEngineTool.from_defaults(
        query_engine=summary_engine,
        description="Best for broad overview questions or summaries of topics.",
    ),
]

# Router picks the best engine per query
router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=tools,
)

response = router_engine.query("What is error code E-4012?")
# Automatically routes to keyword_engine
```

### LangChain with RunnableBranch

```python
"""
LangChain routing using RunnableBranch.
"""
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def is_keyword_query(query: str) -> bool:
    """Check if query needs keyword search."""
    import re
    return bool(re.search(r'[A-Z]{1,5}[-_]\d{3,}|"[^"]{3,}"', query))


def is_temporal_query(query: str) -> bool:
    temporal = ["latest", "recent", "last", "newest", "updated"]
    return any(w in query.lower() for w in temporal)


# Route to different retrieval chains
retrieval_branch = RunnableBranch(
    (lambda x: is_keyword_query(x["query"]), bm25_chain),
    (lambda x: is_temporal_query(x["query"]), metadata_filter_chain),
    hybrid_chain,  # default
)

# Use:
# result = retrieval_branch.invoke({"query": "Error E-4012"})
```

---

## Trade-offs

| Approach                          | Accuracy   | Latency   | Cost | Maintenance         |
| --------------------------------- | ---------- | --------- | ---- | ------------------- |
| **No routing** (always hybrid)    | ⭐⭐⭐     | ⚡ Low    | Free | None                |
| **Rule-based router**             | ⭐⭐⭐⭐   | ⚡ Low    | Free | Rules need updating |
| **LLM-based router**              | ⭐⭐⭐⭐⭐ | 🐌 +200ms | $$   | Prompt needs tuning |
| **Hybrid (rules + LLM fallback)** | ⭐⭐⭐⭐⭐ | ⚡/🐌     | $    | Best balance        |

**Recommendation:** Start with hybrid-always (no routing). Add rule-based routing when you see specific query types failing. Graduate to LLM routing only if rules can't handle the diversity.

---

## Common Pitfalls

| Pitfall                                  | Impact                                | Fix                                                          |
| ---------------------------------------- | ------------------------------------- | ------------------------------------------------------------ |
| Over-engineering routing from day 1      | Complexity without data to justify it | Start with hybrid-always, measure, then add routing          |
| Not logging which route was taken        | Can't debug retrieval failures        | Log strategy per query                                       |
| LLM router adding latency to every query | Slower retrieval                      | Use rules first, LLM only for ambiguous cases                |
| No fallback when routed strategy fails   | Silent failure                        | Always fallback to hybrid if selected strategy returns empty |
| Hardcoded temporal keywords              | Misses new patterns                   | Periodically review routing accuracy                         |

---

## Syllabus Mapping

Maps to **2.6.3** in `p2b_context_retriever.md` — Retrieval Strategy Selection (Routing Layer). This is a production-critical pattern that determines retrieval quality at the system level.
