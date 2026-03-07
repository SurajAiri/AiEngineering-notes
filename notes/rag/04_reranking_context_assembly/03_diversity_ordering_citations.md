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

## Part 2: Citation Alignment

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
