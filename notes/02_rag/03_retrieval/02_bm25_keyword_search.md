# BM25 / Keyword Search

## Why It Matters

BM25 (Best Matching 25) is the gold standard for **lexical/keyword search**. Unlike vector search which finds semantically similar documents, BM25 finds documents that literally contain the query terms. In production RAG, BM25 catches cases that vector search misses — especially for **exact terms, acronyms, IDs, and domain jargon**.

---

## Core Concept

```
BM25 scores a document based on:
1. Term Frequency (TF)  — How often does the query term appear in this doc?
2. Inverse Document Frequency (IDF) — How rare is this term across all docs?
3. Document Length normalization — Don't bias toward longer documents.

Formula:
                               TF(t,d) × (k₁ + 1)
score(q,d) = Σ  IDF(t) × ─────────────────────────────────────
             t∈q           TF(t,d) + k₁ × (1 - b + b × |d|/avgdl)

Where:
  k₁ = 1.2–2.0  (term frequency saturation; higher = TF matters more)
  b  = 0.75      (length normalization; 0 = no normalization, 1 = full)
```

### Intuition

```
Query: "kubernetes pod scaling"

Document A: "Kubernetes pod autoscaling is configured via HPA..."
  → Has "kubernetes" ✓, "pod" ✓, "scaling" ✓ → HIGH score

Document B: "Container orchestration enables elastic workload management..."
  → Has NONE of the query terms → ZERO score (despite being semantically related!)

This is the limitation AND the strength of BM25:
  ✅ Precise: Finds exactly what you asked for
  ❌ No semantic understanding: "car" won't match "automobile"
```

---

## When BM25 Wins Over Vector Search

```
┌────────────────────────────────────────────────────────────┐
│  BM25 WINS                    │  VECTOR SEARCH WINS        │
├───────────────────────────────┼────────────────────────────┤
│ Exact product names           │ Paraphrased questions      │
│ Error codes: "ERR_0x4A2F"     │ "How do I fix memory?"     │
│ Acronyms: "RBAC", "OIDC"     │ Conceptual similarity      │
│ API paths: "/api/v2/users"    │ Multilingual queries       │
│ Config keys: "max_retries"    │ Vague or exploratory Q&A   │
│ Rare domain terms             │ Questions about concepts   │
│ Legal citations: "42 USC 1983"│ "What does X mean?"       │
└───────────────────────────────┴────────────────────────────┘
```

---

## Simple Code — BM25 from Scratch

```python
"""
BM25 implementation from scratch to understand the algorithm.
No external libraries.
"""

import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenization."""
    return text.lower().split()


class SimpleBM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = documents
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths)
        self.n_docs = len(documents)

        # Build document frequency table
        self.df = {}  # term -> number of docs containing it
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] = self.df.get(term, 0) + 1

    def _idf(self, term: str) -> float:
        """Inverse document frequency with smoothing."""
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str) -> list[tuple[int, float]]:
        """Score all documents for a query. Returns [(doc_idx, score)]."""
        query_terms = tokenize(query)
        scores = []

        for doc_idx, doc_tokens in enumerate(self.doc_tokens):
            tf = Counter(doc_tokens)
            doc_len = self.doc_lengths[doc_idx]
            score = 0.0

            for term in query_terms:
                if term not in tf:
                    continue
                term_freq = tf[term]
                idf = self._idf(term)
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avgdl
                )
                score += idf * numerator / denominator

            scores.append((doc_idx, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)


# Example
docs = [
    "Python is a popular programming language for machine learning",
    "BM25 is a ranking function used in information retrieval",
    "Vector databases store embeddings for similarity search",
    "Kubernetes manages container orchestration at scale",
    "The BM25 algorithm scores documents based on term frequency",
]

bm25 = SimpleBM25(docs)
results = bm25.score("BM25 ranking algorithm")

print("Query: 'BM25 ranking algorithm'\n")
for idx, score in results:
    if score > 0:
        print(f"  [{score:.4f}] {docs[idx]}")
```

---

## Production Code — BM25 with rank_bm25

```python
"""
Production BM25 search using the rank_bm25 library.
Includes proper tokenization, stopword removal, and result formatting.

Requirements: pip install rank_bm25 nltk
"""

import re
import logging
from dataclasses import dataclass
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Minimal stopword list — expand as needed
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
    "both", "either", "neither", "each", "every", "all", "any", "few",
    "more", "most", "other", "some", "such", "no", "only", "own",
    "same", "than", "too", "very", "just", "because", "if", "when",
    "where", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "it", "its",
}


@dataclass
class BM25Result:
    index: int
    score: float
    text: str


class ProductionBM25:
    """
    Production-quality BM25 search with proper tokenization.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        remove_stopwords: bool = True,
    ):
        self.k1 = k1
        self.b = b
        self.remove_stopwords = remove_stopwords
        self.bm25: BM25Okapi | None = None
        self.texts: list[str] = []

    def index(self, texts: list[str]):
        """Build BM25 index from a list of texts."""
        self.texts = texts
        tokenized = [self._tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info(f"BM25 index built with {len(texts)} documents")

    def search(self, query: str, k: int = 5) -> list[BM25Result]:
        """Search for top-k documents matching the query."""
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call index() first.")

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = scores.argsort()[::-1][:k]

        results = []
        for idx in top_k_indices:
            score = float(scores[idx])
            if score > 0:  # Only return documents with non-zero scores
                results.append(BM25Result(
                    index=int(idx),
                    score=score,
                    text=self.texts[int(idx)],
                ))

        return results

    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text: lowercase, remove punctuation, split on whitespace.
        Optionally remove stopwords.
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS]
        return tokens


# ─── Usage ───
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    search = ProductionBM25()

    documents = [
        "Configure RBAC policies in Kubernetes using ClusterRole and RoleBinding",
        "The HPA (Horizontal Pod Autoscaler) scales pods based on CPU utilization",
        "Use kubectl apply -f deployment.yaml to deploy applications",
        "Pod scheduling is controlled by node selectors and affinity rules",
        "Container images should be pulled from a private registry with imagePullSecrets",
        "Kubernetes service mesh with Istio provides mTLS between services",
        "Error ERR_0x4A2F occurs when the API gateway times out",
        "Set max_retries=3 in the client configuration for resilience",
    ]

    search.index(documents)

    queries = [
        "RBAC configuration",
        "ERR_0x4A2F",
        "how to scale pods",
        "max_retries setting",
    ]

    for query in queries:
        results = search.search(query, k=3)
        print(f"\nQuery: '{query}'")
        for r in results:
            print(f"  [{r.score:.4f}] {r.text}")
```

---

## BM25 Parameters Deep Dive

```
Parameter k₁ (default: 1.2–2.0)
─────────────────────────────────

Controls term frequency saturation.

  k₁ = 0: Only IDF matters, TF is ignored
  k₁ = 1.5: Default, good for most cases
  k₁ = 3+: Documents with more term occurrences score much higher

  Low k₁ → "Does the term appear?" (boolean-like)
  High k₁ → "How many times does the term appear?"

  For RAG chunks (short texts): k₁ = 1.2 works well
  For long documents: k₁ = 1.5–2.0


Parameter b (default: 0.75)
─────────────────────────────

Controls document length normalization.

  b = 0: No length normalization (longer docs not penalized)
  b = 0.75: Default, moderate length normalization
  b = 1.0: Full length normalization

  For RAG chunks (similar lengths): b = 0.5–0.75
  For mixed-length documents: b = 0.75
```

---

## Pitfalls & Common Mistakes

| Mistake                         | Impact                                      | Fix                                       |
| ------------------------------- | ------------------------------------------- | ----------------------------------------- |
| **Skipping BM25 entirely**      | Missing exact matches for codes, names, IDs | Always include BM25 in hybrid setup       |
| **No stopword removal**         | Common words dominate scores                | Remove stopwords or use sublinear TF      |
| **Not tokenizing consistently** | Query and doc tokens don't match            | Same tokenizer for indexing and search    |
| **Forgetting about stemming**   | "running" doesn't match "run"               | Consider adding a stemmer (PorterStemmer) |
| **Using BM25 alone**            | No semantic understanding                   | Combine with vector search (hybrid)       |

---

## Key Takeaways

1. **BM25 is essential for exact matches** — product names, error codes, config keys, IDs.
2. **It has zero semantic understanding** — "car" won't match "automobile".
3. **Complements vector search perfectly** — what one misses, the other catches.
4. **Tune k₁ and b** for your chunk sizes — defaults work for most cases.
5. **In production**, always pair BM25 with vector search in a hybrid setup.

---

## Popular Libraries

| Library                 | Purpose                 | Install                           |
| ----------------------- | ----------------------- | --------------------------------- |
| rank_bm25               | Pure Python BM25 scorer | `pip install rank_bm25`           |
| Elasticsearch           | Full-text search engine | Docker / managed service          |
| LangChain BM25Retriever | Framework integration   | `pip install langchain-community` |

### Quick Example — rank_bm25

```python
from rank_bm25 import BM25Okapi

# Tokenize documents (simple whitespace split; use a proper tokenizer in production)
corpus = [
    "python programming language syntax",
    "retrieval augmented generation pipeline",
    "BM25 keyword matching algorithm",
    "error code ERR_CONNECTION_REFUSED",
]
tokenized_corpus = [doc.lower().split() for doc in corpus]

# Build BM25 index
bm25 = BM25Okapi(tokenized_corpus)

# Query
query = "error code connection"
tokenized_query = query.lower().split()
scores = bm25.get_scores(tokenized_query)

# Get top results
for idx in scores.argsort()[::-1][:2]:
    print(f"Score: {scores[idx]:.2f} | {corpus[idx]}")
# Output: "error code ERR_CONNECTION_REFUSED" scores highest (exact keyword match)
```

---

## Common Questions

### Q: When does BM25 beat vector search?

**A:** When the query contains **exact terms** that must match: product IDs ("SKU-12345"), error codes ("ERR_404"), config keys ("max_retries"), proper nouns ("LangChain"), or acronyms ("HNSW"). Vector search might match semantically similar but wrong items.

### Q: Do I need Elasticsearch for BM25?

**A:** No. For small-medium datasets (<1M docs), `rank_bm25` in Python is fast enough. Use Elasticsearch when you need: full-text search features (fuzzy matching, query DSL), scalability across machines, or real-time indexing.

### Q: How does hybrid search combine BM25 + vector scores?

**A:** The most common approach is **Reciprocal Rank Fusion (RRF)**: take the top results from both BM25 and vector search, combine their rank positions with the formula `1/(k + rank)`, and return the highest fused scores. See the hybrid retrieval note for details.
