# Adaptive Retrieval Depth & Retrieval Abstention

## Why It Matters

Using a fixed `k` for every query is like wearing the same prescription glasses for reading and driving — sometimes you need more results, sometimes fewer, and sometimes you should return **nothing at all**. Adaptive retrieval adjusts `k` based on query complexity, and abstention means knowing when to say "I don't have relevant information."

---

## Core Concept

```
FIXED k (NAIVE):
  Every query → Retrieve exactly k=5 → Feed to LLM

  Problem: "What is 2+2?"         → 5 chunks retrieved (4 useless)
           "Compare all auth methods" → 5 chunks retrieved (needs 15)
           "Latest Mars mission?"    → 5 chunks retrieved (all irrelevant)

ADAPTIVE k:
  Simple factual Q    → k=3  (answer likely in top few)
  Comparison query    → k=15 (needs multiple sources)
  Broad analysis      → k=20 (needs comprehensive coverage)
  Off-topic query     → k=0  (ABSTAIN — no relevant docs)

┌─────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────┐
│  Query  │ ──► │  Classify    │ ──► │ Set k        │ ──► │Retrieve│
│         │     │  complexity  │     │ dynamically  │     │        │
└─────────┘     └──────────────┘     └──────────────┘     └────────┘
                                            │
                                     ┌──────┴───────┐
                                     │ Score Check   │
                                     │ Abstain if    │
                                     │ all scores    │
                                     │ below threshold│
                                     └───────────────┘
```

---

## Simple Code — Score-Based Adaptive k

```python
"""
Simple adaptive retrieval: adjust k based on score distribution.
"""

import numpy as np


def adaptive_retrieve(
    scores: list[float],
    texts: list[str],
    min_score: float = 0.3,
    score_drop_ratio: float = 0.6,
    min_k: int = 1,
    max_k: int = 20,
) -> list[tuple[str, float]]:
    """
    Dynamically select how many results to return based on scores.

    Rules:
    1. Drop everything below min_score
    2. Drop when score drops below score_drop_ratio × top_score
    3. Enforce min_k and max_k bounds
    """
    if not scores:
        return []

    top_score = scores[0]

    # If even the best result is bad, abstain
    if top_score < min_score:
        return []  # ABSTENTION

    results = []
    threshold = top_score * score_drop_ratio

    for score, text in zip(scores, texts):
        if score < min_score:
            break
        if score < threshold and len(results) >= min_k:
            break
        if len(results) >= max_k:
            break
        results.append((text, score))

    return results


# Example
scores = [0.85, 0.82, 0.78, 0.45, 0.42, 0.15, 0.12]
texts = [f"Doc {i}" for i in range(len(scores))]

results = adaptive_retrieve(scores, texts)
print(f"Selected {len(results)} of {len(scores)} results:")
for text, score in results:
    print(f"  [{score:.2f}] {text}")
# Selected 3 — dropped after the score cliff at 0.78 → 0.45
```

---

## Production Code — Full Adaptive Retrieval System

```python
"""
Production adaptive retrieval with query classification,
dynamic k, score-based filtering, and abstention.

Requirements: pip install sentence-transformers faiss-cpu numpy
"""

import logging
import re
import numpy as np
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    SIMPLE = "simple"         # factual, single-answer
    MODERATE = "moderate"     # needs a few sources
    COMPLEX = "complex"       # comparison, analysis, multi-hop
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class AdaptiveConfig:
    """k and threshold settings per complexity level."""
    initial_k: int
    min_score: float
    score_drop_ratio: float


# Default configs per complexity
COMPLEXITY_CONFIGS = {
    QueryComplexity.SIMPLE: AdaptiveConfig(
        initial_k=5, min_score=0.35, score_drop_ratio=0.7
    ),
    QueryComplexity.MODERATE: AdaptiveConfig(
        initial_k=10, min_score=0.30, score_drop_ratio=0.6
    ),
    QueryComplexity.COMPLEX: AdaptiveConfig(
        initial_k=20, min_score=0.25, score_drop_ratio=0.5
    ),
}


@dataclass
class AdaptiveResult:
    texts: list[str]
    scores: list[float]
    k_used: int
    k_returned: int
    complexity: QueryComplexity
    abstained: bool
    abstention_reason: str = ""


class QueryClassifier:
    """
    Rule-based query complexity classifier.
    Replace with an LLM classifier for production.
    """

    COMPLEX_PATTERNS = [
        r"compare|contrast|difference|versus|vs\.?",
        r"list all|enumerate|comprehensive|overview",
        r"how does .+ relate to",
        r"pros and cons|advantages and disadvantages",
        r"step.by.step|walkthrough|tutorial",
    ]

    SIMPLE_PATTERNS = [
        r"^what is\b",
        r"^who is\b",
        r"^when (did|was|is)\b",
        r"^(define|explain)\s+\w+$",
        r"^(yes or no|true or false)",
    ]

    def classify(self, query: str) -> QueryComplexity:
        q = query.lower().strip()

        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, q):
                return QueryComplexity.COMPLEX

        for pattern in self.SIMPLE_PATTERNS:
            if re.search(pattern, q):
                return QueryComplexity.SIMPLE

        # Default: moderate
        word_count = len(q.split())
        if word_count <= 5:
            return QueryComplexity.SIMPLE
        elif word_count <= 15:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX


class AdaptiveRetriever:
    """
    Retriever with dynamic k selection and abstention.

    1. Classifies query complexity
    2. Sets initial k based on complexity
    3. Retrieves k results
    4. Filters by score distribution
    5. Returns only meaningful results (or abstains)
    """

    def __init__(
        self,
        search_fn,  # callable(query: str, k: int) -> (scores, texts)
        classifier: QueryClassifier | None = None,
        configs: dict | None = None,
    ):
        self.search_fn = search_fn
        self.classifier = classifier or QueryClassifier()
        self.configs = configs or COMPLEXITY_CONFIGS

    def retrieve(self, query: str) -> AdaptiveResult:
        """
        Retrieve with adaptive k and optional abstention.
        """
        # Step 1: Classify query
        complexity = self.classifier.classify(query)
        config = self.configs.get(
            complexity,
            COMPLEXITY_CONFIGS[QueryComplexity.MODERATE],
        )

        logger.info(
            f"Query classified as {complexity.value}, "
            f"initial_k={config.initial_k}"
        )

        # Step 2: Retrieve with initial k
        scores, texts = self.search_fn(query, config.initial_k)

        # Step 3: Abstention check
        if not scores or scores[0] < config.min_score:
            logger.info(f"Abstaining: top score {scores[0] if scores else 0:.3f} "
                       f"< threshold {config.min_score}")
            return AdaptiveResult(
                texts=[],
                scores=[],
                k_used=config.initial_k,
                k_returned=0,
                complexity=complexity,
                abstained=True,
                abstention_reason=(
                    f"Top retrieval score ({scores[0]:.3f}) below threshold "
                    f"({config.min_score}). No relevant documents found."
                    if scores else "No results returned from search."
                ),
            )

        # Step 4: Adaptive filtering
        filtered_texts, filtered_scores = self._adaptive_filter(
            scores, texts, config
        )

        return AdaptiveResult(
            texts=filtered_texts,
            scores=filtered_scores,
            k_used=config.initial_k,
            k_returned=len(filtered_texts),
            complexity=complexity,
            abstained=False,
        )

    def _adaptive_filter(
        self,
        scores: list[float],
        texts: list[str],
        config: AdaptiveConfig,
    ) -> tuple[list[str], list[float]]:
        """
        Filter results based on score distribution.

        Uses two strategies:
        1. Absolute threshold: drop below min_score
        2. Relative drop: drop when score drops by score_drop_ratio
        """
        top_score = scores[0]
        drop_threshold = top_score * config.score_drop_ratio

        filtered_texts = []
        filtered_scores = []

        for i, (score, text) in enumerate(zip(scores, texts)):
            # Hard floor
            if score < config.min_score:
                break

            # Relative drop (but keep at least 1 result)
            if i > 0 and score < drop_threshold:
                break

            # Cliff detection: if score drops more than 30% from previous
            if i > 0 and score < scores[i - 1] * 0.7:
                break

            filtered_texts.append(text)
            filtered_scores.append(score)

        return filtered_texts, filtered_scores


# ─── Usage ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simulate a search function
    import faiss
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = [
        "RBAC in Kubernetes uses ClusterRole and RoleBinding objects",
        "NetworkPolicy controls pod-to-pod traffic",
        "HPA scales pods based on CPU and memory metrics",
        "Terraform manages infrastructure as code",
        "The weather in Paris is sunny today",
        "Pod security standards restrict container capabilities",
        "Service accounts provide identity for processes in pods",
        "Custom Resource Definitions extend the Kubernetes API",
    ]

    embeddings = model.encode(documents, normalize_embeddings=True).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    def search_fn(query: str, k: int):
        q_emb = model.encode(query, normalize_embeddings=True).astype(np.float32)
        distances, indices = index.search(np.array([q_emb]), k)
        scores = [float(d) for d in distances[0]]
        texts = [documents[int(i)] for i in indices[0] if i != -1]
        return scores, texts

    retriever = AdaptiveRetriever(search_fn=search_fn)

    # Test cases
    queries = [
        "What is RBAC?",                           # Simple → low k
        "Compare RBAC and NetworkPolicy security",  # Complex → high k
        "Latest SpaceX launch details",             # Off-topic → abstain
    ]

    for query in queries:
        result = retriever.retrieve(query)
        print(f"\nQuery: '{query}'")
        print(f"  Complexity: {result.complexity.value}")
        print(f"  k_used={result.k_used}, k_returned={result.k_returned}")
        print(f"  Abstained: {result.abstained}")
        if result.abstained:
            print(f"  Reason: {result.abstention_reason}")
        else:
            for text, score in zip(result.texts, result.scores):
                print(f"  [{score:.3f}] {text}")
```

---

## Abstention Design

```
WHEN TO ABSTAIN (return nothing):

  ┌─────────────────────────────────────────────────────┐
  │  1. Top score < threshold     → "I don't know"      │
  │  2. All scores clustered low  → "Nothing matches"   │
  │  3. Query is out-of-domain    → "Can't help"        │
  │  4. No docs after filtering   → "Not enough info"   │
  └─────────────────────────────────────────────────────┘

HOW TO COMMUNICATE ABSTENTION:

  ❌ BAD:  Silently pass empty context to LLM
           → LLM hallucinates an answer from its training data

  ✅ GOOD: Return explicit signal to the generation step
           → LLM says "I don't have information about that."

  In practice:
    if retrieval_result.abstained:
        system_prompt += "\nIMPORTANT: No relevant documents were found. "
        system_prompt += "Respond with 'I don't have information about that.'"
```

---

## Pitfalls & Common Mistakes

| Mistake                              | Impact                                  | Fix                                                  |
| ------------------------------------ | --------------------------------------- | ---------------------------------------------------- |
| **Fixed k for all queries**          | Wasted tokens or missed context         | Classify query complexity, set k dynamically         |
| **No abstention**                    | LLM hallucinates when docs don't match  | Check top score, abstain below threshold             |
| **Too aggressive abstention**        | System refuses too many valid queries   | Set threshold conservatively, tune with real queries |
| **Not telling LLM about abstention** | LLM generates answer from training data | Explicitly instruct LLM to say "I don't know"        |
| **Ignoring score distribution**      | Return noisy low-relevance results      | Use cliff detection and score drop ratio             |

---

## Key Takeaways

1. **Fixed k is wrong** — simple queries need fewer results, complex queries need more.
2. **Abstention is a feature** — it's better to say "I don't know" than hallucinate.
3. **Score cliffs are natural split points** — when scores drop sharply, that's where relevant docs end.
4. **Communicate abstention to the LLM** — don't just pass empty context silently.
5. **Tune thresholds per domain** — what counts as "low" depends on your embedding model and data.
