# RAG Observability & Debugging

## Why It Matters

When a RAG system gives a wrong answer, you need to know **why**. Was it a retrieval problem (wrong chunks), a generation problem (LLM hallucinated), or a data problem (source was wrong)? Observability gives you the tools to trace, diagnose, and fix issues.

```
WITHOUT OBSERVABILITY:
  User: "The system gave me a wrong answer."
  You:  "Which query? What happened? I have no idea." 🤷

WITH OBSERVABILITY:
  User: "The system gave me a wrong answer."
  You:  "Let me check the trace..."

  Query: "What's the refund policy?"
  ├── Embedding: ✅ 45ms
  ├── Retrieval: ⚠️ Top chunk about shipping, not refunds
  │   ├── Score: 0.62 (below 0.7 threshold)
  │   └── Expected docs: refund-policy-2024 NOT in results
  ├── Re-rank: ✅ Correctly deprioritized shipping chunk
  ├── Generation: ❌ Hallucinated a 30-day refund policy
  └── Root cause: Refund policy doc not indexed after last update
```

---

## Retrieval Trace Logging

```python
"""
Structured logging for RAG pipeline traces.

Requirements: pip install uuid (stdlib)
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class ChunkTrace:
    """Trace info for a single retrieved chunk."""
    chunk_id: str
    text_preview: str  # first 100 chars
    score: float
    source_doc: str
    was_used: bool = True    # included in final context?
    was_reranked: bool = False


@dataclass
class RAGTrace:
    """Complete trace of a single RAG query."""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Input
    query: str = ""
    query_type: str = ""  # factual, how_to, comparison

    # Retrieval
    retrieval_strategy: str = ""  # vector, bm25, hybrid
    chunks_retrieved: int = 0
    chunks_used: int = 0
    top_score: float = 0.0
    lowest_used_score: float = 0.0
    chunk_traces: list[ChunkTrace] = field(default_factory=list)

    # Re-ranking
    reranker_used: bool = False
    reranker_model: str = ""
    rank_changes: int = 0  # how many chunks changed position

    # Generation
    llm_model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    # Latency
    embed_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # Quality signals
    answer: str = ""
    user_feedback: str = ""  # thumbs_up, thumbs_down, none

    def to_log_entry(self) -> dict:
        """Convert to a structured log entry."""
        return {
            "trace_id": self.trace_id,
            "timestamp": self.timestamp,
            "query": self.query,
            "retrieval": {
                "strategy": self.retrieval_strategy,
                "chunks_retrieved": self.chunks_retrieved,
                "chunks_used": self.chunks_used,
                "top_score": round(self.top_score, 4),
                "lowest_used_score": round(self.lowest_used_score, 4),
            },
            "latency": {
                "embed_ms": round(self.embed_latency_ms, 1),
                "retrieval_ms": round(self.retrieval_latency_ms, 1),
                "rerank_ms": round(self.rerank_latency_ms, 1),
                "generation_ms": round(self.generation_latency_ms, 1),
                "total_ms": round(self.total_latency_ms, 1),
            },
            "cost": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "quality": {
                "user_feedback": self.user_feedback,
            },
        }


class TraceLogger:
    """Persist and query RAG traces."""

    def __init__(self, log_dir: str = "rag_traces"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.traces: list[RAGTrace] = []

    def log(self, trace: RAGTrace):
        """Log a trace to file and memory."""
        self.traces.append(trace)

        # Append to daily log file
        date_str = datetime.now().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"traces_{date_str}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(trace.to_log_entry()) + "\n")

    def get_low_quality_traces(
        self,
        score_threshold: float = 0.5,
    ) -> list[RAGTrace]:
        """Find traces with low retrieval scores (potential quality issues)."""
        return [
            t for t in self.traces
            if t.top_score < score_threshold
        ]

    def get_slow_traces(self, latency_threshold_ms: float = 3000) -> list[RAGTrace]:
        """Find traces that exceeded latency budget."""
        return [
            t for t in self.traces
            if t.total_latency_ms > latency_threshold_ms
        ]

    def get_negative_feedback(self) -> list[RAGTrace]:
        """Find traces where users gave negative feedback."""
        return [
            t for t in self.traces
            if t.user_feedback == "thumbs_down"
        ]


# ─── Usage ───
logger = TraceLogger()

# Simulate a traced RAG query
trace = RAGTrace(
    query="What is the refund policy?",
    retrieval_strategy="hybrid",
    chunks_retrieved=10,
    chunks_used=5,
    top_score=0.82,
    lowest_used_score=0.55,
    reranker_used=True,
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    llm_model="gpt-4o-mini",
    input_tokens=1200,
    output_tokens=150,
    embed_latency_ms=35,
    retrieval_latency_ms=28,
    rerank_latency_ms=180,
    generation_latency_ms=820,
    total_latency_ms=1063,
    answer="The refund policy allows returns within 30 days...",
)
trace.chunk_traces = [
    ChunkTrace("chunk_42", "Refund policy: customers may return...", 0.82, "policy.pdf"),
    ChunkTrace("chunk_15", "Shipping takes 3-5 business days...", 0.55, "shipping.pdf", was_used=False),
]

logger.log(trace)
print(json.dumps(trace.to_log_entry(), indent=2))
```

---

## Chunk Contribution Analysis

```python
"""
Analyze which chunks actually contributed to the answer.
Finds chunks that were retrieved but not used (waste) or
chunks that were heavily relied on.
"""

from dataclasses import dataclass
from collections import Counter


@dataclass
class ChunkContribution:
    chunk_id: str
    source_doc: str
    times_retrieved: int = 0
    times_used: int = 0
    avg_score: float = 0.0
    positive_feedback_count: int = 0
    negative_feedback_count: int = 0

    @property
    def usage_rate(self) -> float:
        return self.times_used / self.times_retrieved if self.times_retrieved > 0 else 0

    @property
    def quality_signal(self) -> str:
        if self.negative_feedback_count > self.positive_feedback_count:
            return "problematic"
        if self.usage_rate < 0.3:
            return "rarely_used"
        return "healthy"


class ChunkAnalyzer:
    """Analyze chunk retrieval and usage patterns from traces."""

    def __init__(self):
        self.contributions: dict[str, ChunkContribution] = {}

    def process_traces(self, traces: list[RAGTrace]):
        """Aggregate chunk usage from traces."""
        for trace in traces:
            for ct in trace.chunk_traces:
                if ct.chunk_id not in self.contributions:
                    self.contributions[ct.chunk_id] = ChunkContribution(
                        chunk_id=ct.chunk_id,
                        source_doc=ct.source_doc,
                    )

                contrib = self.contributions[ct.chunk_id]
                contrib.times_retrieved += 1
                if ct.was_used:
                    contrib.times_used += 1

                # Running average score
                n = contrib.times_retrieved
                contrib.avg_score = (
                    (contrib.avg_score * (n - 1) + ct.score) / n
                )

                # Feedback
                if trace.user_feedback == "thumbs_up":
                    contrib.positive_feedback_count += 1
                elif trace.user_feedback == "thumbs_down":
                    contrib.negative_feedback_count += 1

    def get_problematic_chunks(self) -> list[ChunkContribution]:
        """Chunks frequently retrieved but associated with negative feedback."""
        return [
            c for c in self.contributions.values()
            if c.quality_signal == "problematic"
        ]

    def get_waste_chunks(self) -> list[ChunkContribution]:
        """Chunks frequently retrieved but rarely used."""
        return [
            c for c in self.contributions.values()
            if c.times_retrieved >= 5 and c.usage_rate < 0.2
        ]

    def source_doc_quality(self) -> dict[str, dict]:
        """Aggregate quality signals by source document."""
        doc_stats: dict[str, dict] = {}
        for contrib in self.contributions.values():
            doc = contrib.source_doc
            if doc not in doc_stats:
                doc_stats[doc] = {
                    "chunks": 0, "total_retrieved": 0,
                    "total_used": 0, "positive": 0, "negative": 0,
                }
            stats = doc_stats[doc]
            stats["chunks"] += 1
            stats["total_retrieved"] += contrib.times_retrieved
            stats["total_used"] += contrib.times_used
            stats["positive"] += contrib.positive_feedback_count
            stats["negative"] += contrib.negative_feedback_count

        return doc_stats
```

---

## A/B Testing for Retrieval Strategies

```python
"""
A/B test framework for comparing RAG configurations.
"""

import random
import json
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ABTestConfig:
    """Configuration for an A/B test."""
    test_name: str
    variant_a_name: str  # e.g., "hybrid_retrieval"
    variant_b_name: str  # e.g., "vector_only"
    traffic_split: float = 0.5  # fraction going to variant B
    start_date: str = field(default_factory=lambda: datetime.now().isoformat())

    def assign_variant(self, query_id: str) -> str:
        """Deterministically assign a query to a variant."""
        # Use hash for deterministic, reproducible assignment
        import hashlib
        hash_val = int(hashlib.md5(
            f"{self.test_name}:{query_id}".encode()
        ).hexdigest(), 16)
        if (hash_val % 100) / 100 < self.traffic_split:
            return self.variant_b_name
        return self.variant_a_name


@dataclass
class ABTestResults:
    variant_a_metrics: dict[str, list[float]] = field(default_factory=dict)
    variant_b_metrics: dict[str, list[float]] = field(default_factory=dict)

    def add_result(self, variant: str, metrics: dict[str, float]):
        target = (
            self.variant_a_metrics if variant == "a"
            else self.variant_b_metrics
        )
        for key, value in metrics.items():
            if key not in target:
                target[key] = []
            target[key].append(value)

    def compare(self) -> dict:
        """Compare variants across all tracked metrics."""
        import numpy as np

        comparison = {}
        all_metrics = set(self.variant_a_metrics) | set(self.variant_b_metrics)

        for metric in all_metrics:
            a_vals = self.variant_a_metrics.get(metric, [])
            b_vals = self.variant_b_metrics.get(metric, [])

            if a_vals and b_vals:
                a_mean = np.mean(a_vals)
                b_mean = np.mean(b_vals)
                comparison[metric] = {
                    "variant_a_mean": round(a_mean, 4),
                    "variant_b_mean": round(b_mean, 4),
                    "difference": round(b_mean - a_mean, 4),
                    "relative_change": round((b_mean - a_mean) / a_mean * 100, 2)
                        if a_mean != 0 else 0,
                    "variant_a_n": len(a_vals),
                    "variant_b_n": len(b_vals),
                }

        return comparison
```

---

## User Feedback Integration

```python
"""
Collect and use user feedback to improve RAG quality.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class FeedbackType(Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    WRONG_ANSWER = "wrong_answer"
    INCOMPLETE = "incomplete"
    IRRELEVANT = "irrelevant"
    OUTDATED = "outdated"


@dataclass
class UserFeedback:
    trace_id: str
    feedback_type: FeedbackType
    comment: str = ""
    correct_answer: str = ""  # if user provides the right answer
    timestamp: str = ""


class FeedbackAnalyzer:
    """Analyze user feedback patterns to find systemic issues."""

    def __init__(self):
        self.feedback: list[UserFeedback] = []

    def add(self, fb: UserFeedback):
        self.feedback.append(fb)

    def issue_breakdown(self) -> dict[str, int]:
        """Count issues by type."""
        from collections import Counter
        return dict(Counter(fb.feedback_type.value for fb in self.feedback))

    def common_failure_queries(
        self,
        traces: list[RAGTrace],
        min_negative: int = 2,
    ) -> list[dict]:
        """
        Find queries that repeatedly get negative feedback.
        These are candidates for adding to the eval set or fixing retrieval.
        """
        negative_trace_ids = {
            fb.trace_id for fb in self.feedback
            if fb.feedback_type in (
                FeedbackType.THUMBS_DOWN,
                FeedbackType.WRONG_ANSWER,
            )
        }

        # Group by similar queries
        from collections import defaultdict
        query_issues = defaultdict(int)
        for trace in traces:
            if trace.trace_id in negative_trace_ids:
                # Normalize query for grouping
                normalized = trace.query.lower().strip()
                query_issues[normalized] += 1

        return [
            {"query": q, "negative_count": c}
            for q, c in sorted(query_issues.items(), key=lambda x: -x[1])
            if c >= min_negative
        ]
```

---

## Retrieval Quality Dashboard Metrics

```
DASHBOARD LAYOUT:

  ┌────────────────────────────────────────────────────────┐
  │                    RAG Health Dashboard                 │
  ├────────────────────┬───────────────────────────────────┤
  │                    │                                   │
  │  🟢 RETRIEVAL      │  📊 QUALITY TRENDS               │
  │  Recall@5: 0.89    │  ┌─────────────────────┐        │
  │  MRR: 0.82         │  │ Faithfulness (30d)  │        │
  │  Avg top score:0.78│  │ 0.95 ████████████   │        │
  │                    │  │ 0.91 █████████████   │        │
  │  🟡 LATENCY        │  │ 0.88 ████████████   │        │
  │  p50: 850ms        │  └─────────────────────┘        │
  │  p95: 2100ms       │                                   │
  │  p99: 4500ms       │  📊 FEEDBACK                    │
  │                    │  👍 82%  👎 12%  😐 6%           │
  │  🟢 COST           │                                   │
  │  Avg: $0.008/query │  ⚠️  ALERTS                     │
  │  Daily: $84.00     │  • 3 docs stale (>30d)          │
  │                    │  • p99 above 5s threshold        │
  │  🟢 FRESHNESS      │  • Chunk "doc_42" has 5          │
  │  92% docs < 30d old│    negative feedbacks            │
  │                    │                                   │
  └────────────────────┴───────────────────────────────────┘
```

```python
"""
Generate dashboard metrics from traces and feedback.
"""

import numpy as np
from datetime import datetime, timedelta


def dashboard_metrics(
    traces: list[RAGTrace],
    feedback: list[UserFeedback],
    window_days: int = 7,
) -> dict:
    """Compute dashboard metrics for the last N days."""
    cutoff = datetime.now() - timedelta(days=window_days)
    recent_traces = [
        t for t in traces
        if datetime.fromisoformat(t.timestamp) > cutoff
    ]

    if not recent_traces:
        return {"error": "No traces in window"}

    # Retrieval metrics
    top_scores = [t.top_score for t in recent_traces]

    # Latency metrics
    latencies = [t.total_latency_ms for t in recent_traces]

    # Cost metrics
    total_input = sum(t.input_tokens for t in recent_traces)
    total_output = sum(t.output_tokens for t in recent_traces)

    # Feedback metrics
    recent_feedback = [
        f for f in feedback
        if f.trace_id in {t.trace_id for t in recent_traces}
    ]
    positive = sum(1 for f in recent_feedback if f.feedback_type == FeedbackType.THUMBS_UP)
    negative = sum(1 for f in recent_feedback if f.feedback_type == FeedbackType.THUMBS_DOWN)

    return {
        "window_days": window_days,
        "total_queries": len(recent_traces),
        "retrieval": {
            "avg_top_score": round(np.mean(top_scores), 3),
            "low_score_queries": sum(1 for s in top_scores if s < 0.5),
        },
        "latency": {
            "p50_ms": round(np.percentile(latencies, 50), 0),
            "p95_ms": round(np.percentile(latencies, 95), 0),
            "p99_ms": round(np.percentile(latencies, 99), 0),
        },
        "cost": {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        },
        "feedback": {
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": positive / (positive + negative)
                if (positive + negative) > 0 else None,
        },
    }
```

---

## Pitfalls & Common Mistakes

| Mistake                                        | Impact                                    | Fix                                                             |
| ---------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------- |
| **No trace logging**                           | Can't debug wrong answers                 | Log every query with full trace                                 |
| **Logging only errors**                        | Silent quality degradation goes unnoticed | Log ALL queries, not just failures                              |
| **No user feedback loop**                      | Can't learn from real-world usage         | Add thumbs up/down, analyze patterns                            |
| **Dashboard but no alerts**                    | Problems seen too late                    | Set alerts on p99 latency, low scores, negative feedback spikes |
| **Not tracking chunk contribution**            | Waste of tokens on unused chunks          | Analyze which chunks are retrieved but never used               |
| **A/B tests without statistical significance** | Drawing conclusions from noise            | Wait for sufficient sample size before deciding                 |

---

## Key Takeaways

1. **Log every query as a trace** — you need the full pipeline trace to debug issues.
2. **Chunk contribution analysis** reveals waste — chunks that are retrieved but never contribute to answers.
3. **User feedback is the most important quality signal** — make it easy to give.
4. **Dashboards need alerts** — metrics nobody looks at are useless.
5. **A/B test retrieval changes** — don't deploy blindly, measure the impact.
6. **The debugging flow: trace → identify stage → fix root cause** — traces tell you WHERE it broke.
