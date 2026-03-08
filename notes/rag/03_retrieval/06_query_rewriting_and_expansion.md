# Query Rewriting & Multi-Query Expansion

## Why It Matters

User queries are often **vague, ambiguous, or poorly worded**. The embedding of a bad query will be far from the relevant documents — even if those documents exist in the index. **Query rewriting** transforms the user's query into a better one before retrieval. **Multi-query expansion** generates multiple variations to cast a wider net.

---

## Core Concept

```
Original query: "why is it slow?"

                    ┌─────────────────────────────────────┐
                    │         QUERY REWRITING              │
                    ├─────────────────────────────────────┤
                    │                                      │
                    │  "why is it slow?"                   │
                    │         │                             │
                    │         ▼                             │
                    │  ┌─────────────┐                     │
                    │  │   Rewriter  │ (LLM or rules)      │
                    │  └─────────────┘                     │
                    │         │                             │
                    │    ┌────┴────┐                       │
                    │    ▼         ▼                       │
                    │  Single    Multiple                  │
                    │  rewrite   expansions                │
                    │    │         │                       │
                    │    ▼         ▼                       │
                    │  "What causes   "latency issues"    │
                    │   high latency   "performance       │
                    │   in the          degradation"      │
                    │   system?"       "slow response     │
                    │                   time causes"      │
                    │                                      │
                    │  → Retrieve with each query          │
                    │  → Merge results (RRF / union)       │
                    └─────────────────────────────────────┘
```

### Types of Query Transformation

```
┌─────────────────────────────────────────────────────────────────┐
│  TYPE              │ WHAT IT DOES              │ EXAMPLE         │
├────────────────────┼───────────────────────────┼─────────────────┤
│ Rewriting          │ Rephrase for clarity      │ "slow" → "what  │
│                    │                           │ causes latency?" │
├────────────────────┼───────────────────────────┼─────────────────┤
│ Expansion          │ Generate multiple queries │ One query → 3-5  │
│                    │                           │ sub-queries      │
├────────────────────┼───────────────────────────┼─────────────────┤
│ Decomposition      │ Break complex into parts  │ "Compare A & B" │
│                    │                           │ → query A, B     │
├────────────────────┼───────────────────────────┼─────────────────┤
│ Step-back          │ Ask a more general Q      │ "Fix ERR_04" →  │
│                    │ for background context    │ "Error handling  │
│                    │                           │  architecture"   │
└────────────────────┴───────────────────────────┴─────────────────┘
```

---

## Simple Code — Rule-Based Query Rewriting

```python
"""
Simple rule-based query rewriter.
No LLM needed — handles common patterns.
"""

import re


def rewrite_query(query: str) -> str:
    """
    Apply simple rules to improve query quality.
    Good for handling common patterns before spending tokens on an LLM.
    """
    q = query.strip()

    # Remove filler words at the start
    filler_starts = [
        r"^(hey|hi|hello|please|can you|could you|i want to know|tell me)\s+",
        r"^(i need help with|help me with|what about)\s+",
    ]
    for pattern in filler_starts:
        q = re.sub(pattern, "", q, flags=re.IGNORECASE)

    # Expand common abbreviations
    abbreviations = {
        "k8s": "kubernetes",
        "tf": "terraform",
        "db": "database",
        "auth": "authentication",
        "config": "configuration",
        "env": "environment",
        "prod": "production",
        "deps": "dependencies",
        "infra": "infrastructure",
    }
    words = q.split()
    words = [abbreviations.get(w.lower(), w) for w in words]
    q = " ".join(words)

    # If query is very short, make it a question
    if len(q.split()) <= 3 and not q.endswith("?"):
        q = f"What is {q}?"

    return q


# Examples
test_queries = [
    "hey can you tell me about k8s scaling",
    "auth errors",
    "please help me with db config",
    "slow",
]

for query in test_queries:
    rewritten = rewrite_query(query)
    print(f"  '{query}' → '{rewritten}'")
```

---

## Production Code — LLM-Based Query Rewriting & Expansion

```python
"""
LLM-based query rewriting and multi-query expansion.
Uses an LLM to generate better queries, then retrieves with each
and fuses results.

Requirements: pip install openai sentence-transformers faiss-cpu numpy
"""

import json
import logging
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class ExpandedQuery:
    original: str
    rewritten: str
    sub_queries: list[str]
    step_back_query: str | None = None


class QueryRewriter:
    """
    LLM-based query rewriter and expander.

    Produces:
    1. A rewritten (improved) query
    2. Multiple sub-queries for multi-query expansion
    3. Optional step-back query for complex topics
    """

    REWRITE_PROMPT = """You are a search query optimizer for a document retrieval system.

Given a user query, produce:
1. "rewritten": A clearer, more specific version of the query (keep same intent)
2. "sub_queries": 3 alternative phrasings that might match different relevant documents
3. "step_back": A broader background query (null if the query is already clear)

Rules:
- Preserve the original intent exactly
- Sub-queries should cover different aspects or phrasings
- Step-back query should be more general to get background context
- Output valid JSON only

User query: "{query}"

Output JSON:"""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def expand(self, query: str) -> ExpandedQuery:
        """Expand a user query into multiple search queries."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You output JSON only."},
                {"role": "user", "content": self.REWRITE_PROMPT.format(query=query)},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )

        result = json.loads(response.choices[0].message.content)

        return ExpandedQuery(
            original=query,
            rewritten=result.get("rewritten", query),
            sub_queries=result.get("sub_queries", []),
            step_back_query=result.get("step_back"),
        )


class MultiQueryRetriever:
    """
    Retriever that expands queries and fuses results.

    Pipeline:
    1. Expand user query into multiple queries
    2. Retrieve top-k for each query
    3. Fuse results using Reciprocal Rank Fusion
    """

    def __init__(
        self,
        rewriter: QueryRewriter,
        search_fn,  # callable(query, k) -> list of results
        rrf_k: int = 60,
    ):
        self.rewriter = rewriter
        self.search_fn = search_fn
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        k: int = 5,
        per_query_k: int = 10,
    ) -> list[dict]:
        """
        Expand query, retrieve for each expansion, fuse results.
        """
        # Step 1: Expand
        expanded = self.rewriter.expand(query)
        logger.info(
            f"Query expanded: '{query}' → rewritten='{expanded.rewritten}', "
            f"{len(expanded.sub_queries)} sub-queries"
        )

        # Step 2: Retrieve for each query
        all_queries = [expanded.rewritten] + expanded.sub_queries
        if expanded.step_back_query:
            all_queries.append(expanded.step_back_query)

        ranked_lists: list[list[str]] = []  # list of doc_id lists
        doc_map: dict[str, dict] = {}  # doc_id → result data

        for q in all_queries:
            results = self.search_fn(q, per_query_k)
            ranked_list = []
            for r in results:
                doc_id = str(r.get("id", r.get("index", "")))
                ranked_list.append(doc_id)
                doc_map[doc_id] = r
            ranked_lists.append(ranked_list)

        # Step 3: Fuse with RRF
        fused_scores: dict[str, float] = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list, start=1):
                fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)

        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:k]

        results = []
        for doc_id in sorted_ids:
            result = doc_map[doc_id].copy()
            result["fused_score"] = fused_scores[doc_id]
            result["queries_matched"] = sum(
                1 for rl in ranked_lists if doc_id in rl
            )
            results.append(result)

        return results


# ─── Usage ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rewriter = QueryRewriter(model="gpt-4o-mini")

    # Test query expansion
    expanded = rewriter.expand("why is auth slow in prod?")
    print(f"Original:    {expanded.original}")
    print(f"Rewritten:   {expanded.rewritten}")
    print(f"Sub-queries: {expanded.sub_queries}")
    print(f"Step-back:   {expanded.step_back_query}")
```

---

## Multi-Query Expansion — Practical Example

```python
"""
Concrete example showing how multi-query expansion
improves recall over a single query.
"""

# Original query: "how to fix OOM errors"
#
# Without expansion:
#   Vector search finds: docs about "out of memory" and "memory management"
#   Misses: docs about "container resource limits", "JVM heap settings"
#
# With expansion:
#   Query 1: "How to resolve out of memory errors"
#   Query 2: "Container memory limits and resource allocation"
#   Query 3: "JVM heap size configuration for memory issues"
#   Query 4: "Kubernetes pod memory requests and limits"  (step-back)
#
# RRF fusion combines all results → much higher recall

# This is especially powerful for:
# 1. Technical queries with multiple valid phrasings
# 2. Questions spanning multiple document types
# 3. Vague queries that need to be made specific
```

---

## When NOT to Rewrite

```
DON'T REWRITE:
  ❌ Already specific queries ("What is the max_connections setting?")
  ❌ Exact-match queries (error codes, IDs, paths)
  ❌ Quoted terms ("authentication service" — user means exactly this)

DO REWRITE:
  ✅ Vague queries ("why slow?", "not working", "help with auth")
  ✅ Conversational queries ("I was wondering if the system supports SSO")
  ✅ Complex multi-part questions
  ✅ Queries with abbreviations or informal language
```

---

## Pitfalls & Common Mistakes

| Mistake                           | Impact                          | Fix                                                           |
| --------------------------------- | ------------------------------- | ------------------------------------------------------------- |
| **Rewriting changes the intent**  | Returns irrelevant results      | Set temperature low, validate rewrite matches original intent |
| **Too many sub-queries**          | High latency, token cost        | 3-5 sub-queries max                                           |
| **Not including original query**  | Lose the user's exact wording   | Always include original alongside rewrites                    |
| **Rewriting exact-match queries** | "ERR_0x4A" becomes "error code" | Detect exact patterns and skip rewriting                      |
| **No caching of rewrites**        | Same query rewritten repeatedly | Cache rewrites by query hash                                  |

---

## Key Takeaways

1. **Query rewriting is cheap insurance** — catches vague/badly worded queries that would otherwise miss.
2. **Multi-query expansion boosts recall** by searching with multiple phrasings.
3. **Start with rules**, add LLM rewriting only for complex cases.
4. **Always keep the original query** — don't replace it entirely.
5. **Fuse multi-query results with RRF** — same technique as hybrid retrieval.
