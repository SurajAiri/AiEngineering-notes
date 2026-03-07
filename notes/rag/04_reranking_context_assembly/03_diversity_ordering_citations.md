# Diversity, Relevance, Ordering & Citation Alignment

## Part 1: Diversity vs Relevance

### Why It Matters

Top-k results from vector search often cluster around the same subtopic. If a user asks "What security features does Kubernetes offer?", pure relevance ranking might return 5 chunks all about RBAC — missing network policies, pod security, and secrets management. **Diversity-aware re-ranking** ensures broader coverage.

```
PURE RELEVANCE (no diversity):
  1. [0.95] RBAC uses ClusterRole for access control
  2. [0.93] Configure RBAC with RoleBinding objects
  3. [0.91] RBAC authorization in Kubernetes APIs
  4. [0.88] Role vs ClusterRole differences
  5. [0.85] RBAC best practices
  → All about RBAC! Answer will miss other security features.

WITH DIVERSITY:
  1. [0.95] RBAC uses ClusterRole for access control
  2. [0.88] Network policies control pod-to-pod traffic
  3. [0.82] Pod security standards: Privileged, Baseline, Restricted
  4. [0.79] Secrets management with external secret stores
  5. [0.76] Container security scanning with Trivy
  → Comprehensive answer covering all security areas ✅
```

---

### Maximal Marginal Relevance (MMR)

```
MMR balances relevance and diversity:

  MMR(dᵢ) = λ × sim(dᵢ, q) - (1-λ) × max_j∈S sim(dᵢ, dⱼ)

  Where:
    λ = 0.5-0.7 → higher = more relevance, lower = more diversity
    q = query
    S = already selected documents
    dᵢ = candidate document

  Intuition: Pick documents that are
  1. Similar to the query (relevant)
  2. Different from already-selected documents (diverse)
```

---

### Code — MMR Re-ranking

```python
"""
Maximal Marginal Relevance for diversity-aware re-ranking.

Requirements: pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def mmr_rerank(
    query: str,
    documents: list[str],
    model: SentenceTransformer,
    k: int = 5,
    lambda_param: float = 0.6,
) -> list[tuple[int, float, str]]:
    """
    Re-rank documents using Maximal Marginal Relevance.

    Args:
        lambda_param: 0-1, higher = more relevance, lower = more diversity

    Returns:
        [(original_index, mmr_score, text)]
    """
    # Encode everything
    query_emb = model.encode(query, normalize_embeddings=True)
    doc_embs = model.encode(documents, normalize_embeddings=True)

    # Query-document similarities
    query_sims = np.dot(doc_embs, query_emb)

    selected = []
    selected_indices = set()
    candidates = list(range(len(documents)))

    for _ in range(min(k, len(documents))):
        best_idx = -1
        best_score = -float('inf')

        for idx in candidates:
            if idx in selected_indices:
                continue

            # Relevance to query
            relevance = float(query_sims[idx])

            # Max similarity to already-selected docs
            if selected:
                selected_embs = doc_embs[list(selected_indices)]
                max_sim = float(np.max(np.dot(selected_embs, doc_embs[idx])))
            else:
                max_sim = 0.0

            # MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx == -1:
            break

        selected.append((best_idx, best_score, documents[best_idx]))
        selected_indices.add(best_idx)

    return selected


# Example
model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "RBAC uses ClusterRole and RoleBinding for access control",
    "Configure RBAC with kubectl create clusterrolebinding",
    "RBAC authorization for Kubernetes API requests",
    "Network policies control pod-to-pod traffic",
    "Pod security standards define three restriction levels",
    "Secrets can be stored in external vaults like HashiCorp Vault",
    "Container images should be scanned for vulnerabilities",
]

# Without diversity
print("=== Pure relevance (top-5 by similarity) ===")
query_emb = model.encode("Kubernetes security features", normalize_embeddings=True)
doc_embs = model.encode(documents, normalize_embeddings=True)
sims = np.dot(doc_embs, query_emb)
for idx in np.argsort(sims)[::-1][:5]:
    print(f"  [{sims[idx]:.3f}] {documents[idx]}")

# With MMR diversity
print("\n=== MMR (λ=0.5, balanced) ===")
results = mmr_rerank("Kubernetes security features", documents, model, k=5, lambda_param=0.5)
for idx, score, text in results:
    print(f"  [{score:.3f}] {text}")
```

---

## Part 2: Context Packing Strategies

### Breadth-First vs Depth-First

Once you have ranked chunks, you must decide **how** to fill the context window. This is the **packing strategy** — and it directly affects answer quality.

```
BREADTH-FIRST PACKING:
  Take ONE chunk from each source document.

  [DocA: chunk1] [DocB: chunk1] [DocC: chunk1] [DocD: chunk1] [DocE: chunk1]

  ✅ Diverse sources → comprehensive answers
  ✅ Good for: "Compare X and Y", "What are the options?"
  ❌ Shallow depth per document
  ❌ May miss important detail only in a follow-up chunk

DEPTH-FIRST PACKING:
  Take MULTIPLE chunks from the top-ranked document.

  [DocA: chunk1] [DocA: chunk2] [DocA: chunk3]

  ✅ Deep context from one coherent source
  ✅ Good for: "Explain step-by-step how X works"
  ❌ Misses other perspectives
  ❌ If DocA is wrong, everything is wrong

HYBRID PACKING (recommended default):
  Take top 2-3 chunks from top 2-3 documents.

  [DocA: chunk1, chunk2] [DocB: chunk1, chunk2] [DocC: chunk1]

  ✅ Balance of depth and breadth
  ✅ Works for most query types
  ❌ Needs careful token budget management
```

### Code — Context Packing

```python
"""
Context packing strategies: breadth-first, depth-first, and hybrid.

No external dependencies needed.
"""

from dataclasses import dataclass


@dataclass
class RankedChunk:
    text: str
    score: float
    doc_id: str
    chunk_index: int  # position within the source document


def pack_breadth_first(
    chunks: list[RankedChunk],
    max_chunks: int = 5,
) -> list[RankedChunk]:
    """
    Breadth-first: one chunk per document, highest-scoring first.
    Maximizes source diversity.
    """
    seen_docs = set()
    packed = []

    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        if chunk.doc_id not in seen_docs:
            packed.append(chunk)
            seen_docs.add(chunk.doc_id)
        if len(packed) >= max_chunks:
            break

    return packed


def pack_depth_first(
    chunks: list[RankedChunk],
    max_chunks: int = 5,
) -> list[RankedChunk]:
    """
    Depth-first: multiple chunks from the top-ranked document.
    Maximizes context coherence.
    """
    if not chunks:
        return []

    # Find the top-scoring document
    best_doc = max(chunks, key=lambda c: c.score).doc_id

    # Get all chunks from that document, in document order
    doc_chunks = sorted(
        [c for c in chunks if c.doc_id == best_doc],
        key=lambda c: c.chunk_index,
    )

    return doc_chunks[:max_chunks]


def pack_hybrid(
    chunks: list[RankedChunk],
    max_chunks: int = 5,
    chunks_per_doc: int = 2,
) -> list[RankedChunk]:
    """
    Hybrid: top N chunks from top M documents.
    Balances depth and breadth.
    """
    from collections import defaultdict

    # Group by document
    doc_chunks: dict[str, list[RankedChunk]] = defaultdict(list)
    for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
        doc_chunks[chunk.doc_id].append(chunk)

    # Rank documents by their best chunk score
    doc_order = sorted(
        doc_chunks.keys(),
        key=lambda d: doc_chunks[d][0].score,
        reverse=True,
    )

    packed = []
    for doc_id in doc_order:
        # Take top chunks_per_doc from this document, in document order
        top = sorted(
            doc_chunks[doc_id][:chunks_per_doc],
            key=lambda c: c.chunk_index,
        )
        packed.extend(top)
        if len(packed) >= max_chunks:
            break

    return packed[:max_chunks]


# Example
chunks = [
    RankedChunk("RBAC controls API access", 0.95, "security_guide", 0),
    RankedChunk("RBAC uses ClusterRole objects", 0.90, "security_guide", 1),
    RankedChunk("Network policies restrict traffic", 0.88, "networking_guide", 0),
    RankedChunk("Network policy selectors", 0.82, "networking_guide", 1),
    RankedChunk("Pod security standards", 0.80, "security_guide", 3),
    RankedChunk("Secrets management with Vault", 0.78, "secrets_guide", 0),
]

print("=== Breadth-first (diverse) ===")
for c in pack_breadth_first(chunks, max_chunks=3):
    print(f"  [{c.score:.2f}] {c.doc_id}: {c.text}")

print("\n=== Depth-first (coherent) ===")
for c in pack_depth_first(chunks, max_chunks=3):
    print(f"  [{c.score:.2f}] {c.doc_id}: {c.text}")

print("\n=== Hybrid (balanced) ===")
for c in pack_hybrid(chunks, max_chunks=4, chunks_per_doc=2):
    print(f"  [{c.score:.2f}] {c.doc_id}: {c.text}")
```

---

## Part 3: Diversity vs Relevance — Formalized

### When Redundancy Helps vs Hurts

```
WHEN REDUNDANCY HELPS:
  ✅ Same fact stated in two different ways
     → LLM synthesis is more confident and accurate
  ✅ One doc is formal, another explains simply
     → LLM blends both perspectives for a better answer
  ✅ Multiple sources confirm the same answer
     → Higher trust in the response

WHEN REDUNDANCY HURTS:
  ❌ Token budget is tight
     → Identical info from 2 slots wastes 1 slot
  ❌ Two "identical" docs have minor wording differences
     → LLM hedges or picks the wrong version
  ❌ Redundant chunks push out a unique relevant chunk
     → Coverage drops

DECISION FRAMEWORK:
  ┌──────────────────────────────┬───────────────┬──────────────────┐
  │ Query Type                   │ λ (MMR param) │ Packing Strategy │
  ├──────────────────────────────┼───────────────┼──────────────────┤
  │ Single-fact lookup           │ 0.8-0.9       │ Depth-first      │
  │ "What is the API rate limit?"│ (relevance)   │                  │
  ├──────────────────────────────┼───────────────┼──────────────────┤
  │ Multi-faceted question       │ 0.5-0.6       │ Breadth-first    │
  │ "What security features?"    │ (diversity)   │                  │
  ├──────────────────────────────┼───────────────┼──────────────────┤
  │ How-to / procedure           │ 0.7           │ Hybrid           │
  │ "How to deploy on K8s?"      │ (balanced)    │                  │
  └──────────────────────────────┴───────────────┴──────────────────┘
```

### Tuning λ Practically

```
HOW TO TUNE λ (not guesswork):

  Step 1: Build a golden eval set with 50+ queries
  Step 2: Run evaluation with λ = 0.5, 0.6, 0.7, 0.8, 0.9
  Step 3: Measure context_precision (does the LLM get the right info?)
  Step 4: Pick the λ that maximizes context_precision

  Example results:
    λ=0.5 → context_precision = 0.72  (too diverse, missed key info)
    λ=0.6 → context_precision = 0.78
    λ=0.7 → context_precision = 0.84  ← BEST for this dataset
    λ=0.8 → context_precision = 0.82
    λ=0.9 → context_precision = 0.79  (too redundant, wasted slots)

  IMPORTANT: λ is dataset-specific. Always tune on YOUR data.
  The "default 0.7" advice is a starting point, not a universal truth.
```

---

## Part 4: Citation Alignment

### Why It Matters

When an LLM generates an answer from retrieved context, users need to know **which chunk supported which claim**. Citation alignment maps answer sentences back to their source chunks, enabling verification and building trust.

```
ANSWER WITH CITATIONS:

"Kubernetes provides several security mechanisms.
RBAC controls API access through ClusterRole objects [1].
Network policies restrict pod-to-pod communication [2].
Pod security standards enforce container restrictions [3]."

[1] k8s-security-guide, section 4.2
[2] k8s-networking-guide, section 2.1
[3] k8s-security-guide, section 5.3
```

### Citation Implementation

```python
"""
Citation alignment: map LLM answer sentences to source chunks.

Requirements: pip install sentence-transformers numpy
"""

import re
import numpy as np
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


@dataclass
class Citation:
    sentence: str
    source_chunk_index: int
    source_text: str
    confidence: float  # how well the sentence matches the source


class CitationAligner:
    """Map answer sentences to their source chunks."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        min_confidence: float = 0.4,
    ):
        self.model = SentenceTransformer(model_name)
        self.min_confidence = min_confidence

    def align(
        self,
        answer: str,
        source_chunks: list[str],
    ) -> list[Citation]:
        """
        For each sentence in the answer, find the best matching source chunk.
        """
        sentences = self._split_sentences(answer)
        if not sentences or not source_chunks:
            return []

        # Encode sentences and chunks
        sent_embs = self.model.encode(sentences, normalize_embeddings=True)
        chunk_embs = self.model.encode(source_chunks, normalize_embeddings=True)

        # For each sentence, find best matching chunk
        citations = []
        for i, (sentence, sent_emb) in enumerate(zip(sentences, sent_embs)):
            sims = np.dot(chunk_embs, sent_emb)
            best_idx = int(np.argmax(sims))
            confidence = float(sims[best_idx])

            if confidence >= self.min_confidence:
                citations.append(Citation(
                    sentence=sentence,
                    source_chunk_index=best_idx,
                    source_text=source_chunks[best_idx][:100] + "...",
                    confidence=confidence,
                ))

        return citations

    def format_with_citations(
        self,
        answer: str,
        source_chunks: list[str],
        source_labels: list[str] | None = None,
    ) -> str:
        """
        Add inline citations to the answer text.
        """
        citations = self.align(answer, source_chunks)
        labels = source_labels or [f"Source {i+1}" for i in range(len(source_chunks))]

        # Map sentence → citation marker
        cited_answer = answer
        footnotes = []
        used_sources = set()

        for citation in citations:
            idx = citation.source_chunk_index
            if idx not in used_sources:
                used_sources.add(idx)
                footnote_num = len(footnotes) + 1
                footnotes.append(f"[{footnote_num}] {labels[idx]}")

            # Find the footnote number for this source
            footnote_num = [
                i + 1 for i, fn in enumerate(footnotes)
                if labels[idx] in fn
            ][0]

            # Add citation marker after the sentence
            cited_answer = cited_answer.replace(
                citation.sentence,
                f"{citation.sentence} [{footnote_num}]",
                1,
            )

        # Add footnotes
        if footnotes:
            cited_answer += "\n\n" + "\n".join(footnotes)

        return cited_answer

    def _split_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 10]


# ─── Usage ───
if __name__ == "__main__":
    aligner = CitationAligner()

    answer = (
        "Kubernetes provides multiple security layers. "
        "RBAC controls access to the Kubernetes API through roles and bindings. "
        "Network policies restrict traffic between pods at the network level. "
        "Secrets should be stored in external vaults for production use."
    )

    source_chunks = [
        "RBAC (Role-Based Access Control) uses ClusterRole and RoleBinding objects "
        "to control who can access the Kubernetes API and what actions they can perform.",
        "Network policies are Kubernetes resources that control network traffic between pods. "
        "They use selectors to define ingress and egress rules.",
        "For production environments, Kubernetes secrets should be stored in external "
        "secret management systems like HashiCorp Vault or AWS Secrets Manager.",
    ]

    source_labels = [
        "K8s Security Guide §4.2",
        "K8s Networking Guide §2.1",
        "K8s Production Checklist §7.3",
    ]

    cited = aligner.format_with_citations(answer, source_chunks, source_labels)
    print(cited)
```

---

## Part 3: Context Ordering for Answerability

```
THE "LOST IN THE MIDDLE" PROBLEM:

  Research (Liu et al., 2023) shows LLMs:
  ┌──────────────────────────────────────┐
  │  Position    │ Attention │ Quality   │
  ├──────────────┼───────────┼───────────┤
  │  Beginning   │ HIGH      │ Best      │
  │  End         │ MEDIUM    │ Good      │
  │  Middle      │ LOW       │ Worst     │
  └──────────────┴───────────┴───────────┘

ORDERING STRATEGIES:

  1. RELEVANCE-FIRST (default, simple):
     [highest_score, ..., lowest_score]

  2. BOOKEND (for many chunks):
     [highest, third, fifth, ..., sixth, fourth, second]
     Most relevant at beginning AND end, least relevant in middle.

  3. CHRONOLOGICAL (for time-sensitive docs):
     [oldest_first, ..., newest_last]
     When temporal order matters (changelogs, updates).
```

---

## Pitfalls & Common Mistakes

| Mistake                                      | Impact                           | Fix                                             |
| -------------------------------------------- | -------------------------------- | ----------------------------------------------- |
| **No diversity, all chunks from same topic** | Narrow, incomplete answers       | Use MMR with λ=0.5-0.7                          |
| **No citations**                             | Users can't verify claims        | Add source attribution                          |
| **Citations that don't match**               | Worse than no citations          | Use semantic matching, set confidence threshold |
| **Random context order**                     | LLM struggles with middle chunks | Put most relevant first or use bookend ordering |
| **Too much diversity**                       | Irrelevant chunks included       | Keep λ closer to 0.7 (bias toward relevance)    |

---

## Key Takeaways

1. **Pure relevance ranking causes topic clustering** — use MMR (λ=0.5-0.7) for breadth.
2. **Citations build trust** — map each answer sentence to its source chunk.
3. **Context order matters** — put most relevant chunks at the beginning.
4. **Diversity is most important for broad queries** — factual lookups don't need it.
5. **The full pipeline: retrieve → re-rank → deduplicate → diversify → pack → order → cite**.
