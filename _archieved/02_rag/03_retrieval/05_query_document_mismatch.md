# Query-Document Mismatch Mechanics

## Why It Matters

Even with a good retrieval system, **queries and documents often don't match** — not because the answer is missing, but because they describe the same concept in fundamentally different ways. Understanding **why** they don't match is the first step to fixing retrieval failures.

---

## The Two Types of Mismatch

```
┌─────────────────────────────────────────────────────────────────┐
│                   QUERY-DOCUMENT MISMATCH                       │
├───────────────────────────────┬─────────────────────────────────┤
│   VOCABULARY MISMATCH         │   UNDERSPECIFIED QUERIES        │
│                               │                                 │
│  Query and document use       │  Query is too vague to match    │
│  different words for the      │  any specific document well     │
│  same concept                 │                                 │
│                               │                                 │
│  "fix memory issues"          │  "it's not working"             │
│  vs                           │  → What's "it"?                 │
│  "resolve OOM errors"         │  → What does "working" mean?    │
│                               │                                 │
│  ┌─────┐         ┌─────┐     │       ┌─────┐                   │
│  │Query│ ·       │ Doc │     │       │Query│ ·                  │
│  │     │    gap  │     │     │       │     │      · · · · ·     │
│  └─────┘         └─────┘     │       └─────┘    many possible   │
│  Same meaning, far apart     │       Too far from everything    │
└───────────────────────────────┴─────────────────────────────────┘
```

---

## Vocabulary Mismatch in Detail

```python
"""
Demonstrate vocabulary mismatch — same concepts, different words.
Shows how cosine similarity fails for lexical variants.

Requirements: pip install sentence-transformers numpy
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def show_similarity(text1: str, text2: str) -> float:
    """Show cosine similarity between two texts."""
    emb = model.encode([text1, text2], normalize_embeddings=True)
    sim = float(np.dot(emb[0], emb[1]))
    return sim


# Vocabulary mismatch examples
mismatch_pairs = [
    # (query, document content, mismatch type)
    ("fix memory issues",
     "Resolve OOM errors by increasing pod resource limits",
     "synonym mismatch"),

    ("API is slow",
     "High p95 latency on the /users endpoint",
     "abstraction level"),

    ("how to log in",
     "Authentication flow uses OAuth2 PKCE with OIDC provider",
     "expert vs novice vocabulary"),

    ("cancel my subscription",
     "Churn prevention: Initiate retention workflow before processing cancellation",
     "user vs internal terminology"),

    ("the button doesn't work",
     "onClick handler throws TypeError: Cannot read property 'id' of null",
     "user description vs root cause"),
]

print("Vocabulary Mismatch Analysis\n")
for query, doc, mismatch_type in mismatch_pairs:
    sim = show_similarity(query, doc)
    print(f"  [{sim:.3f}] {mismatch_type}")
    print(f"    Query: '{query}'")
    print(f"    Doc:   '{doc}'")
    print()
```

---

## Underspecified Queries in Detail

```python
"""
Demonstrate underspecified queries — too vague to retrieve well.
"""

# Underspecified queries and what's missing
underspecified = [
    {
        "query": "it's not working",
        "missing": ["What is 'it'?", "What does 'working' mean?", "What's the error?"],
        "better": "The payment processing API returns 500 errors intermittently",
    },
    {
        "query": "how to fix this?",
        "missing": ["Fix what?", "What system?", "What's the symptom?"],
        "better": "How to fix connection timeout errors in the database pool",
    },
    {
        "query": "update the config",
        "missing": ["Which config?", "What value?", "Which environment?"],
        "better": "Update the max_connections setting in production PostgreSQL config",
    },
    {
        "query": "is it supported?",
        "missing": ["What is 'it'?", "Supported by what?", "What version?"],
        "better": "Does the v3.2 API support batch document upload?",
    },
]

print("Underspecified Query Analysis\n")
for item in underspecified:
    print(f"  Query: '{item['query']}'")
    print(f"  Missing: {', '.join(item['missing'])}")
    print(f"  Better:  '{item['better']}'")
    print()
```

---

## Production Code — Mismatch Detection & Mitigation

```python
"""
Detect and mitigate query-document mismatch in a retrieval pipeline.

Strategies:
1. Score threshold detection — low top scores indicate mismatch
2. Score gap analysis — if top scores are clustered, query is ambiguous
3. Query classification — detect underspecified queries
4. Mitigation routing — different strategies for different mismatch types

Requirements: pip install sentence-transformers numpy
"""

import re
import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MismatchType(Enum):
    NONE = "none"
    VOCABULARY = "vocabulary_mismatch"
    UNDERSPECIFIED = "underspecified_query"
    NO_RELEVANT_DOCS = "no_relevant_documents"
    AMBIGUOUS = "ambiguous_query"


@dataclass
class MismatchDiagnosis:
    type: MismatchType
    confidence: float
    evidence: str
    mitigation: str


class MismatchDetector:
    """
    Analyzes retrieval results to detect query-document mismatch.
    """

    def __init__(
        self,
        low_score_threshold: float = 0.3,
        gap_threshold: float = 0.05,    # max gap between top scores
        min_query_tokens: int = 4,
    ):
        self.low_score_threshold = low_score_threshold
        self.gap_threshold = gap_threshold
        self.min_query_tokens = min_query_tokens

    def diagnose(
        self,
        query: str,
        scores: list[float],
        texts: list[str],
    ) -> MismatchDiagnosis:
        """
        Analyze retrieval results and diagnose mismatch type.

        Args:
            query: The user's original query
            scores: Similarity scores (highest first)
            texts: Retrieved text snippets (same order as scores)
        """
        # Check 1: Is the query underspecified?
        if self._is_underspecified(query):
            return MismatchDiagnosis(
                type=MismatchType.UNDERSPECIFIED,
                confidence=0.9,
                evidence=f"Query has only {len(query.split())} tokens and lacks specificity",
                mitigation="Ask user to clarify, or expand query with context",
            )

        # Check 2: Are all scores low?
        if not scores or scores[0] < self.low_score_threshold:
            # Could be vocabulary mismatch or no relevant docs
            if self._has_keyword_overlap(query, texts):
                return MismatchDiagnosis(
                    type=MismatchType.VOCABULARY,
                    confidence=0.7,
                    evidence=f"Top score {scores[0]:.3f} < threshold, "
                             f"but some keyword overlap exists",
                    mitigation="Use hybrid search (add BM25) or query rewriting",
                )
            else:
                return MismatchDiagnosis(
                    type=MismatchType.NO_RELEVANT_DOCS,
                    confidence=0.8,
                    evidence=f"Top score {scores[0]:.3f} < threshold, "
                             f"no keyword overlap",
                    mitigation="Abstain or inform user that no relevant docs found",
                )

        # Check 3: Are scores too clustered? (ambiguous query)
        if len(scores) >= 3:
            top_gap = scores[0] - scores[2]
            if top_gap < self.gap_threshold:
                return MismatchDiagnosis(
                    type=MismatchType.AMBIGUOUS,
                    confidence=0.6,
                    evidence=f"Score gap between #1 and #3 is only {top_gap:.3f}",
                    mitigation="Use multi-query expansion or ask user to clarify",
                )

        return MismatchDiagnosis(
            type=MismatchType.NONE,
            confidence=0.9,
            evidence=f"Top score {scores[0]:.3f} is healthy",
            mitigation="No mitigation needed",
        )

    def _is_underspecified(self, query: str) -> bool:
        """Detect vague/underspecified queries."""
        tokens = query.lower().split()
        if len(tokens) < self.min_query_tokens:
            return True

        vague_patterns = [
            r"^(it|this|that)\s+(is|isn't|doesn't|won't|can't)",
            r"^how\s+to\s+(fix|solve|handle)\s+(this|it|that)$",
            r"^(not working|broken|help|issue|problem)$",
        ]
        for pattern in vague_patterns:
            if re.search(pattern, query.lower()):
                return True
        return False

    def _has_keyword_overlap(self, query: str, texts: list[str]) -> bool:
        """Check if any query terms appear in retrieved texts."""
        query_terms = set(query.lower().split())
        trivial = {"the", "a", "an", "is", "are", "to", "in", "of", "and", "or", "how", "what"}
        query_terms -= trivial

        for text in texts[:3]:
            text_lower = text.lower()
            if any(term in text_lower for term in query_terms):
                return True
        return False


class MismatchMitigator:
    """
    Routes different mismatch types to appropriate mitigation strategies.
    """

    def mitigate(
        self,
        query: str,
        diagnosis: MismatchDiagnosis,
        search_fn,   # callable(query, k) -> results
    ) -> dict:
        """
        Apply mitigation strategy based on diagnosis.
        """
        if diagnosis.type == MismatchType.NONE:
            return {"strategy": "none", "results": search_fn(query, 5)}

        if diagnosis.type == MismatchType.VOCABULARY:
            return self._mitigate_vocabulary(query, search_fn)

        if diagnosis.type == MismatchType.UNDERSPECIFIED:
            return self._mitigate_underspecified(query, search_fn)

        if diagnosis.type == MismatchType.AMBIGUOUS:
            return self._mitigate_ambiguous(query, search_fn)

        if diagnosis.type == MismatchType.NO_RELEVANT_DOCS:
            return {"strategy": "abstain", "results": [], "message": "No relevant documents found."}

        return {"strategy": "fallback", "results": search_fn(query, 5)}

    def _mitigate_vocabulary(self, query: str, search_fn):
        """
        For vocabulary mismatch: try BM25, query expansion,
        or synonym-based search.
        """
        # Strategy: search with expanded terms
        expanded_queries = [
            query,
            query.replace("fix", "resolve").replace("issue", "error"),
            query.replace("slow", "latency").replace("fast", "low latency"),
        ]

        all_results = []
        seen = set()
        for q in expanded_queries:
            results = search_fn(q, 5)
            for r in results:
                key = r.get("text", "")[:50]
                if key not in seen:
                    seen.add(key)
                    all_results.append(r)

        return {"strategy": "vocabulary_expansion", "results": all_results[:5]}

    def _mitigate_underspecified(self, query: str, search_fn):
        """
        For underspecified queries: search broadly,
        or request clarification.
        """
        return {
            "strategy": "clarification_needed",
            "results": search_fn(query, 3),
            "clarification": f"Your query '{query}' is vague. "
                           f"Can you specify what system or error you're referring to?",
        }

    def _mitigate_ambiguous(self, query: str, search_fn):
        """
        For ambiguous queries: return diverse results from
        different apparent topics.
        """
        results = search_fn(query, 10)
        return {
            "strategy": "diverse_results",
            "results": results,
            "note": "Multiple equally relevant results found. "
                   "Results may cover different interpretations.",
        }


# ─── Usage ───
if __name__ == "__main__":
    detector = MismatchDetector()

    # Simulated retrieval results
    test_cases = [
        {
            "query": "fix memory issues",
            "scores": [0.25, 0.23, 0.22],
            "texts": ["OOM errors can be resolved by...", "Memory leak detection...", "Heap dump analysis..."],
        },
        {
            "query": "it's not working",
            "scores": [0.41, 0.40, 0.39],
            "texts": ["Service health check...", "Error handling...", "Debugging guide..."],
        },
        {
            "query": "How to configure RBAC in Kubernetes?",
            "scores": [0.85, 0.72, 0.65],
            "texts": ["RBAC configuration...", "ClusterRole setup...", "Service accounts..."],
        },
    ]

    for case in test_cases:
        diagnosis = detector.diagnose(
            case["query"], case["scores"], case["texts"]
        )
        print(f"\nQuery: '{case['query']}'")
        print(f"  Diagnosis: {diagnosis.type.value}")
        print(f"  Confidence: {diagnosis.confidence}")
        print(f"  Evidence: {diagnosis.evidence}")
        print(f"  Mitigation: {diagnosis.mitigation}")
```

---

## Mitigation Strategy Summary

```
┌──────────────────────┬──────────────────────────────────────────┐
│ MISMATCH TYPE        │ MITIGATION STRATEGIES                    │
├──────────────────────┼──────────────────────────────────────────┤
│ Vocabulary           │ 1. Hybrid search (add BM25)              │
│                      │ 2. Query rewriting/expansion             │
│                      │ 3. HyDE (hypothetical document)          │
│                      │ 4. Synonym expansion                     │
├──────────────────────┼──────────────────────────────────────────┤
│ Underspecified       │ 1. Ask for clarification                 │
│                      │ 2. Use conversation context              │
│                      │ 3. Multi-query with different guesses    │
│                      │ 4. Show diverse results, let user pick   │
├──────────────────────┼──────────────────────────────────────────┤
│ No relevant docs     │ 1. Abstain (say "I don't know")          │
│                      │ 2. Web search fallback (CRAG)            │
│                      │ 3. Suggest related queries               │
├──────────────────────┼──────────────────────────────────────────┤
│ Ambiguous            │ 1. Return diverse results                │
│                      │ 2. Ask user to disambiguate              │
│                      │ 3. Multi-query expansion                 │
└──────────────────────┴──────────────────────────────────────────┘
```

---

## Key Takeaways

1. **Vocabulary mismatch** is the #1 retrieval failure mode — users and documents use different words.
2. **Underspecified queries** are inevitable in production — build detection and mitigation.
3. **Hybrid search (BM25 + vector) fixes most vocabulary mismatch** without extra complexity.
4. **Monitor retrieval scores** — low top scores signal mismatch, clustered scores signal ambiguity.
5. **It's better to abstain than hallucinate** — when no docs match, say so.
