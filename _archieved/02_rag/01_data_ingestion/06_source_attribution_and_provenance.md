# Source Attribution & Chunk-Level Provenance

## Why This Matters

Users need to know **where** an answer came from. Regulators may require it. Developers need it to debug. Without provenance, a RAG system is a black box that says "trust me" — and in production, nobody should.

---

## Source Attribution vs Provenance

```
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  SOURCE ATTRIBUTION                                          │
│  "Where did this answer come from?"                          │
│                                                              │
│  → Document name, page number, URL                           │
│  → Shown to the USER                                         │
│  → For trust and verification                                │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  CHUNK-LEVEL PROVENANCE                                      │
│  "What exact content influenced this generation?"            │
│                                                              │
│  → Which chunks were retrieved, in what order                │
│  → What scores they had                                      │
│  → Shown to DEVELOPERS                                       │
│  → For debugging and quality improvement                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## The Attribution Flow

```
Query: "What's the return policy?"
         │
         ▼
    ┌─────────┐
    │Retrieval│ → Chunk A (return_policy.pdf, p3, score=0.92)
    │         │ → Chunk B (faq.md, section 4, score=0.87)
    │         │ → Chunk C (terms.pdf, p12, score=0.81)
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │   LLM   │ → Generated answer + citations
    └────┬────┘
         │
         ▼
    Answer: "Returns are accepted within 7 days [1][2]"

    Sources:
    [1] return_policy.pdf, page 3
    [2] faq.md, section "Returns & Refunds"
```

---

## Simple Code — Track What Goes Into the LLM

```python
"""
Minimal source attribution: track which chunks the LLM sees.
"""

# Simulated retrieved chunks
retrieved_chunks = [
    {
        "content": "Returns are accepted within 7 days with a digital receipt.",
        "metadata": {
            "source": "return_policy_v3.pdf",
            "page": 3,
            "chunk_index": 2,
        },
        "score": 0.92,
    },
    {
        "content": "For exchanges, no receipt is required within 3 days.",
        "metadata": {
            "source": "faq.md",
            "section": "Returns & Refunds",
            "chunk_index": 0,
        },
        "score": 0.87,
    },
]

# Build context with source markers
context_parts = []
sources = []
for i, chunk in enumerate(retrieved_chunks, 1):
    context_parts.append(f"[Source {i}]: {chunk['content']}")
    sources.append({
        "citation": f"[{i}]",
        "source": chunk["metadata"]["source"],
        "page": chunk["metadata"].get("page"),
        "section": chunk["metadata"].get("section"),
        "relevance_score": chunk["score"],
    })

context = "\n\n".join(context_parts)

# Prompt instructs the LLM to cite sources
prompt = f"""Answer the question using ONLY the provided sources.
Cite sources using [1], [2], etc.

Sources:
{context}

Question: What's the return policy?
"""

print(prompt)
print("\n--- Source Index ---")
for s in sources:
    print(f"  {s['citation']} → {s['source']}, page={s.get('page')}, section={s.get('section')}")
```

---

## Production Code — Full Provenance Tracking

```python
"""
Production provenance tracking for RAG systems.
Records the complete trail from query to answer.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single chunk retrieved for a query."""
    chunk_id: str
    content: str
    source: str
    score: float
    rank: int                          # position in retrieval results
    metadata: dict[str, Any] = field(default_factory=dict)
    rerank_score: float | None = None  # score after re-ranking
    included_in_context: bool = True   # was this chunk actually sent to LLM?
    exclusion_reason: str = ""         # why it was excluded (if it was)


@dataclass
class ProvenanceRecord:
    """Complete provenance for a single RAG query-answer pair."""
    trace_id: str
    timestamp: str
    query: str

    # Retrieval details
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    retrieval_method: str = ""         # "vector", "hybrid", "bm25"
    retrieval_k: int = 0
    retrieval_latency_ms: float = 0

    # Re-ranking details
    reranker_model: str = ""
    rerank_latency_ms: float = 0

    # Context assembly
    context_token_count: int = 0
    chunks_in_context: int = 0
    chunks_excluded: int = 0

    # Generation
    model: str = ""
    answer: str = ""
    generation_latency_ms: float = 0

    # Citations extracted from the answer
    citations: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ProvenanceTracker:
    """
    Tracks provenance through the RAG pipeline.
    Use as a context manager or call methods manually.
    """

    def __init__(self):
        self._records: list[ProvenanceRecord] = []

    def start_trace(self, query: str) -> ProvenanceRecord:
        """Start tracking a new query."""
        record = ProvenanceRecord(
            trace_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            query=query,
        )
        self._records.append(record)
        return record

    def record_retrieval(
        self,
        record: ProvenanceRecord,
        chunks: list[dict],
        method: str = "vector",
        k: int = 10,
        latency_ms: float = 0,
    ):
        """Record retrieval results."""
        record.retrieval_method = method
        record.retrieval_k = k
        record.retrieval_latency_ms = latency_ms

        for rank, chunk in enumerate(chunks):
            record.retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.get("id", f"chunk_{rank}"),
                    content=chunk["content"][:500],  # truncate for storage
                    source=chunk["metadata"].get("source", "unknown"),
                    score=chunk.get("score", 0.0),
                    rank=rank,
                    metadata=chunk.get("metadata", {}),
                )
            )

    def record_reranking(
        self,
        record: ProvenanceRecord,
        reranked_scores: list[float],
        model: str = "",
        latency_ms: float = 0,
    ):
        """Record re-ranking results."""
        record.reranker_model = model
        record.rerank_latency_ms = latency_ms
        for chunk, score in zip(record.retrieved_chunks, reranked_scores):
            chunk.rerank_score = score

    def record_context_assembly(
        self,
        record: ProvenanceRecord,
        included_indices: list[int],
        excluded_indices: list[int],
        exclusion_reasons: list[str],
        token_count: int,
    ):
        """Record which chunks were included/excluded from context."""
        record.context_token_count = token_count
        record.chunks_in_context = len(included_indices)
        record.chunks_excluded = len(excluded_indices)

        for idx in excluded_indices:
            if idx < len(record.retrieved_chunks):
                record.retrieved_chunks[idx].included_in_context = False
        for idx, reason in zip(excluded_indices, exclusion_reasons):
            if idx < len(record.retrieved_chunks):
                record.retrieved_chunks[idx].exclusion_reason = reason

    def record_generation(
        self,
        record: ProvenanceRecord,
        answer: str,
        model: str = "",
        latency_ms: float = 0,
    ):
        """Record the generated answer."""
        record.answer = answer
        record.model = model
        record.generation_latency_ms = latency_ms

        # Extract citations from answer (e.g., [1], [2])
        import re
        citation_refs = re.findall(r'\[(\d+)\]', answer)
        for ref in set(citation_refs):
            idx = int(ref) - 1
            if idx < len(record.retrieved_chunks):
                chunk = record.retrieved_chunks[idx]
                record.citations.append({
                    "reference": f"[{ref}]",
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "score": chunk.score,
                })

    def get_trace(self, trace_id: str) -> ProvenanceRecord | None:
        for r in self._records:
            if r.trace_id == trace_id:
                return r
        return None

    def format_user_citations(self, record: ProvenanceRecord) -> str:
        """Format citations for display to end users."""
        if not record.citations:
            return "No sources cited."

        lines = ["**Sources:**"]
        for c in record.citations:
            source = c["source"]
            lines.append(f"- {c['reference']} {source}")
        return "\n".join(lines)

    def format_debug_trace(self, record: ProvenanceRecord) -> str:
        """Format full trace for developer debugging."""
        lines = [
            f"=== TRACE {record.trace_id} ===",
            f"Query: {record.query}",
            f"Timestamp: {record.timestamp}",
            f"",
            f"--- Retrieval ({record.retrieval_method}, k={record.retrieval_k}) ---",
            f"Latency: {record.retrieval_latency_ms:.0f}ms",
        ]
        for chunk in record.retrieved_chunks:
            status = "✓ INCLUDED" if chunk.included_in_context else f"✗ EXCLUDED ({chunk.exclusion_reason})"
            lines.append(
                f"  [{chunk.rank}] score={chunk.score:.3f} "
                f"rerank={chunk.rerank_score:.3f if chunk.rerank_score else 'N/A':>5} "
                f"| {status} | {chunk.source}"
            )
        lines.extend([
            f"",
            f"--- Context ---",
            f"Tokens: {record.context_token_count}",
            f"Chunks included: {record.chunks_in_context}",
            f"Chunks excluded: {record.chunks_excluded}",
            f"",
            f"--- Generation ({record.model}) ---",
            f"Latency: {record.generation_latency_ms:.0f}ms",
            f"Answer: {record.answer[:200]}...",
            f"",
            f"--- Citations ---",
        ])
        for c in record.citations:
            lines.append(f"  {c['reference']} → {c['source']} (score={c['score']:.3f})")

        return "\n".join(lines)


# ─── Usage Example ───

if __name__ == "__main__":
    tracker = ProvenanceTracker()

    # 1. Start trace
    record = tracker.start_trace("What is the return policy?")

    # 2. Record retrieval
    tracker.record_retrieval(
        record,
        chunks=[
            {"id": "c_001", "content": "Returns within 7 days...", "metadata": {"source": "policy.pdf"}, "score": 0.92},
            {"id": "c_002", "content": "Exchanges within 3 days...", "metadata": {"source": "faq.md"}, "score": 0.87},
            {"id": "c_003", "content": "Shipping costs are...", "metadata": {"source": "shipping.pdf"}, "score": 0.65},
        ],
        method="hybrid",
        k=10,
        latency_ms=45,
    )

    # 3. Record reranking
    tracker.record_reranking(record, reranked_scores=[0.95, 0.88, 0.30], model="cross-encoder/ms-marco", latency_ms=120)

    # 4. Record context assembly (chunk 3 excluded — low rerank score)
    tracker.record_context_assembly(
        record,
        included_indices=[0, 1],
        excluded_indices=[2],
        exclusion_reasons=["rerank_score below threshold"],
        token_count=340,
    )

    # 5. Record generation
    tracker.record_generation(
        record,
        answer="Returns are accepted within 7 days with a digital receipt [1]. Exchanges can be made within 3 days without a receipt [2].",
        model="gpt-4o",
        latency_ms=800,
    )

    # Output
    print(tracker.format_user_citations(record))
    print()
    print(tracker.format_debug_trace(record))
```

---

## Pitfalls & Common Mistakes

| Mistake                            | Impact                                          | Fix                                                               |
| ---------------------------------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| **No source tracking**             | Can't show users where answers came from        | Attach source to every chunk at ingestion                         |
| **Citations without verification** | LLM cites [1] but info came from [2]            | Cross-check cited sources against actual content                  |
| **Logging only the answer**        | Can't debug retrieval quality                   | Log the full pipeline: query → chunks → rerank → context → answer |
| **Provenance not queryable**       | Can't investigate patterns in failures          | Store traces in a searchable system (DB, log aggregator)          |
| **Source metadata too vague**      | "file.pdf" isn't helpful — which page? section? | Include page, section, heading, and chunk position                |

---

## Key Takeaways

1. **Attribution is for users** — show them where the answer came from.
2. **Provenance is for developers** — log the complete retrieval-to-answer trail.
3. **Every chunk needs source metadata** — file, page, section, position.
4. **Log generously** — you'll need the trace when debugging production issues.
5. **Verify LLM citations** — models may cite the wrong source; cross-check programmatically.
