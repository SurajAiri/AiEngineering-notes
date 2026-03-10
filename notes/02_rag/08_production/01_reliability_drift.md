# Reliability & Drift

## Why It Matters

A RAG system that works great on day 1 can silently degrade over weeks and months. **Index drift** (documents change but the index doesn't), **data freshness decay** (stale context), and **silent degradation** (no errors, just worse answers) are the biggest production risks.

```
SILENT DEGRADATION TIMELINE:

  Day 1:    System deployed ✅
            Answers are accurate, Recall@5 = 0.91

  Week 4:   3 source documents updated, index not refreshed.
            2 queries now get stale answers. No alerts fired.

  Month 3:  15% of indexed content is outdated.
            Users reporting "system gives old info" in support tickets.
            Recall@5 still looks ok because eval set is also stale.

  Month 6:  Customer churn attributed to "unreliable AI assistant."
            Nobody connected it to index drift.

THE PROBLEM: Nothing crashes. Nothing throws errors.
The system just gradually returns worse answers.
```

---

## Index Drift

```
WHAT IS INDEX DRIFT?

  Source documents change, but the vector index still has the old version.

  Source (current):   "API v3 uses OAuth2 for authentication"
  Index (stale):      "API v2 uses API keys for authentication"

  User asks: "How do I authenticate?"
  System answers based on old index: "Use API keys" ← WRONG

TYPES OF DRIFT:

  ┌──────────────────┬──────────────────────────────────┐
  │ Drift Type       │ Impact                           │
  ├──────────────────┼──────────────────────────────────┤
  │ Content update   │ Answer is outdated               │
  │ Content deletion │ System references removed docs   │
  │ New content      │ System can't answer new topics   │
  │ Schema change    │ Metadata filters break           │
  └──────────────────┴──────────────────────────────────┘
```

### Detection & Prevention

```python
"""
Index drift detection: compare source documents against indexed versions.

Requirements: pip install hashlib
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class DriftReport:
    stale_docs: list[str]          # content changed since indexing
    deleted_docs: list[str]        # removed from source, still in index
    new_docs: list[str]            # in source, not in index
    unchanged_docs: list[str]
    check_time: datetime = field(default_factory=datetime.now)

    @property
    def drift_percentage(self) -> float:
        total = len(self.stale_docs) + len(self.deleted_docs) + \
                len(self.new_docs) + len(self.unchanged_docs)
        if total == 0:
            return 0.0
        drifted = len(self.stale_docs) + len(self.deleted_docs) + len(self.new_docs)
        return drifted / total * 100

    @property
    def needs_reindex(self) -> bool:
        return self.drift_percentage > 10  # >10% drift → reindex


class DriftDetector:
    """
    Detect index drift by comparing document hashes.

    Stores hash of each document at index time, then compares
    against current document state to find changes.
    """

    def __init__(self, manifest_path: str = "index_manifest.json"):
        self.manifest_path = Path(manifest_path)
        self.manifest: dict[str, dict] = {}  # doc_id → {hash, indexed_at}
        self._load_manifest()

    def record_indexed(self, doc_id: str, content: str):
        """Record that a document was indexed."""
        self.manifest[doc_id] = {
            "hash": hashlib.sha256(content.encode()).hexdigest(),
            "indexed_at": datetime.now().isoformat(),
        }
        self._save_manifest()

    def check_drift(
        self,
        current_docs: dict[str, str],  # doc_id → current content
    ) -> DriftReport:
        """
        Compare current documents against indexed versions.

        Args:
            current_docs: Map of doc_id to current document content.
        """
        stale = []
        deleted = []
        new = []
        unchanged = []

        # Check indexed docs against current state
        for doc_id, info in self.manifest.items():
            if doc_id not in current_docs:
                deleted.append(doc_id)
            else:
                current_hash = hashlib.sha256(
                    current_docs[doc_id].encode()
                ).hexdigest()
                if current_hash != info["hash"]:
                    stale.append(doc_id)
                else:
                    unchanged.append(doc_id)

        # Check for new docs not yet indexed
        for doc_id in current_docs:
            if doc_id not in self.manifest:
                new.append(doc_id)

        return DriftReport(
            stale_docs=stale,
            deleted_docs=deleted,
            new_docs=new,
            unchanged_docs=unchanged,
        )

    def _load_manifest(self):
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())

    def _save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))


# ─── Usage ───
if __name__ == "__main__":
    detector = DriftDetector("test_manifest.json")

    # Simulate: indexed 3 docs a week ago
    detector.record_indexed("doc_1", "API v2 uses API keys")
    detector.record_indexed("doc_2", "Default timeout is 30s")
    detector.record_indexed("doc_3", "Deployment guide for v2")

    # Now check against current state
    current = {
        "doc_1": "API v3 uses OAuth2",      # CHANGED
        "doc_2": "Default timeout is 30s",   # UNCHANGED
        # doc_3 deleted
        "doc_4": "New monitoring guide",     # NEW
    }

    report = detector.check_drift(current)
    print(f"Drift: {report.drift_percentage:.1f}%")
    print(f"Stale: {report.stale_docs}")
    print(f"Deleted: {report.deleted_docs}")
    print(f"New: {report.new_docs}")
    print(f"Needs reindex: {report.needs_reindex}")
```

---

## Data Freshness Monitoring

```python
"""
Track data freshness: how old is the indexed content?
"""

from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class FreshnessReport:
    total_docs: int
    fresh_docs: int          # indexed within threshold
    stale_docs: int          # older than threshold
    oldest_doc_age_days: int
    avg_age_days: float

    @property
    def freshness_percentage(self) -> float:
        if self.total_docs == 0:
            return 100.0
        return self.fresh_docs / self.total_docs * 100


def check_freshness(
    indexed_dates: dict[str, datetime],
    freshness_threshold_days: int = 30,
) -> FreshnessReport:
    """
    Check how fresh the indexed documents are.

    Args:
        indexed_dates: doc_id → last indexed datetime
        freshness_threshold_days: docs older than this are "stale"
    """
    now = datetime.now()
    threshold = now - timedelta(days=freshness_threshold_days)

    ages = []
    fresh = 0
    stale = 0

    for doc_id, indexed_at in indexed_dates.items():
        age = (now - indexed_at).days
        ages.append(age)
        if indexed_at >= threshold:
            fresh += 1
        else:
            stale += 1

    return FreshnessReport(
        total_docs=len(indexed_dates),
        fresh_docs=fresh,
        stale_docs=stale,
        oldest_doc_age_days=max(ages) if ages else 0,
        avg_age_days=sum(ages) / len(ages) if ages else 0,
    )


# Example
now = datetime.now()
indexed = {
    "doc_1": now - timedelta(days=5),   # fresh
    "doc_2": now - timedelta(days=15),  # fresh
    "doc_3": now - timedelta(days=45),  # stale
    "doc_4": now - timedelta(days=90),  # very stale
    "doc_5": now - timedelta(days=2),   # fresh
}

report = check_freshness(indexed, freshness_threshold_days=30)
print(f"Freshness: {report.freshness_percentage:.0f}%")
print(f"Fresh: {report.fresh_docs}, Stale: {report.stale_docs}")
print(f"Oldest doc: {report.oldest_doc_age_days} days")
```

---

## Regression Testing for RAG

```python
"""
RAG regression testing: run golden eval set after every change
and compare against baseline.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RegressionResult:
    metric_name: str
    baseline_value: float
    current_value: float

    @property
    def delta(self) -> float:
        return self.current_value - self.baseline_value

    @property
    def is_regression(self) -> bool:
        return self.delta < -0.02  # >2% drop = regression

    @property
    def is_improvement(self) -> bool:
        return self.delta > 0.02


class RegressionTracker:
    """Track RAG metrics over time and detect regressions."""

    def __init__(self, history_path: str = "rag_metrics_history.json"):
        self.history_path = Path(history_path)
        self.history: list[dict] = []
        self._load()

    def record(self, metrics: dict[str, float], label: str = ""):
        """Record a set of metrics."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "label": label,
            "metrics": metrics,
        }
        self.history.append(entry)
        self._save()

    def compare_latest(self) -> list[RegressionResult]:
        """Compare latest metrics against baseline (first entry)."""
        if len(self.history) < 2:
            return []

        baseline = self.history[0]["metrics"]
        current = self.history[-1]["metrics"]

        results = []
        for metric in baseline:
            if metric in current:
                results.append(RegressionResult(
                    metric_name=metric,
                    baseline_value=baseline[metric],
                    current_value=current[metric],
                ))

        return results

    def regression_report(self) -> str:
        """Generate a human-readable regression report."""
        results = self.compare_latest()
        if not results:
            return "Not enough data for regression comparison."

        lines = ["## RAG Regression Report", ""]
        regressions = [r for r in results if r.is_regression]
        improvements = [r for r in results if r.is_improvement]

        if regressions:
            lines.append("### ❌ Regressions")
            for r in regressions:
                lines.append(
                    f"  - **{r.metric_name}**: {r.baseline_value:.3f} → "
                    f"{r.current_value:.3f} ({r.delta:+.3f})"
                )

        if improvements:
            lines.append("### ✅ Improvements")
            for r in improvements:
                lines.append(
                    f"  - **{r.metric_name}**: {r.baseline_value:.3f} → "
                    f"{r.current_value:.3f} ({r.delta:+.3f})"
                )

        stable = [r for r in results if not r.is_regression and not r.is_improvement]
        if stable:
            lines.append(f"### — Stable ({len(stable)} metrics)")

        return "\n".join(lines)

    def _load(self):
        if self.history_path.exists():
            self.history = json.loads(self.history_path.read_text())

    def _save(self):
        self.history_path.write_text(json.dumps(self.history, indent=2))


# ─── Usage ───
tracker = RegressionTracker("test_metrics.json")

# Record baseline
tracker.record({
    "recall@5": 0.91,
    "mrr": 0.85,
    "ndcg@5": 0.88,
    "faithfulness": 0.92,
}, label="baseline - v1.0")

# After a change
tracker.record({
    "recall@5": 0.89,   # slight drop
    "mrr": 0.87,        # improved
    "ndcg@5": 0.86,     # slight drop
    "faithfulness": 0.94, # improved
}, label="after chunking change")

print(tracker.regression_report())
```

---

## Silent Degradation Detection

Silent degradation is the **most dangerous** production RAG failure: no errors, no crashes, just gradually worse answers. Users stop trusting the system before engineers notice anything is wrong.

```
WHAT "SILENT" MEANS:

  ❌ Not this: System throws errors, timeouts, crashes
  ✅ This:     System returns answers. Answers look fine.
               But week over week, answers get SUBTLY worse.

  SIGNALS YOU MUST TRACK:

  ┌──────────────────────────┬────────────────────────────┐
  │ Signal                   │ What It Tells You          │
  ├──────────────────────────┼────────────────────────────┤
  │ Avg retrieval score      │ Index quality degrading    │
  │ trending down            │ (HNSW graph decay, stale   │
  │                          │ embeddings, content drift) │
  ├──────────────────────────┼────────────────────────────┤
  │ Abstention rate rising   │ System saying "I don't     │
  │                          │ know" more often (could be │
  │                          │ good OR coverage gap)      │
  ├──────────────────────────┼────────────────────────────┤
  │ Negative feedback rate   │ Users noticing bad answers │
  │ increasing               │ before your metrics do     │
  ├──────────────────────────┼────────────────────────────┤
  │ Previously-green query   │ A category that was solid  │
  │ categories going yellow  │ is now flaky (targeted     │
  │                          │ content degradation)       │
  └──────────────────────────┴────────────────────────────┘
```

### Multi-Signal Degradation Monitor

```python
"""
Silent degradation detection: multi-signal monitoring for RAG quality.

No external dependencies needed.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class WeeklyMetrics:
    """Weekly aggregate metrics for one RAG system."""
    week_label: str  # e.g. "2025-W12"
    avg_retrieval_score: float
    abstention_rate: float  # fraction of queries where system refused
    negative_feedback_rate: float  # fraction of thumbs_down
    p95_latency_ms: float
    total_queries: int
    category_scores: dict[str, float] = field(default_factory=dict)
    # e.g. {"api_docs": 0.85, "billing": 0.72, "onboarding": 0.91}


@dataclass
class DegradationAlert:
    signal: str
    severity: str  # "warning" or "critical"
    message: str
    current_value: float
    previous_value: float


class SilentDegradationMonitor:
    """
    Multi-signal monitor for RAG quality degradation.
    Compares week-over-week metrics to detect slow decline.
    """

    def __init__(
        self,
        score_drop_warning: float = 0.03,     # 3% drop
        score_drop_critical: float = 0.07,     # 7% drop
        abstention_rise_warning: float = 0.05, # 5pp rise
        feedback_drop_warning: float = 0.10,   # 10pp rise in negative
        latency_rise_warning: float = 500,     # 500ms p95 increase
    ):
        self.thresholds = {
            "score_drop_warning": score_drop_warning,
            "score_drop_critical": score_drop_critical,
            "abstention_rise_warning": abstention_rise_warning,
            "feedback_drop_warning": feedback_drop_warning,
            "latency_rise_warning": latency_rise_warning,
        }

    def check(
        self,
        current: WeeklyMetrics,
        previous: WeeklyMetrics,
    ) -> list[DegradationAlert]:
        """
        Compare two weeks of metrics and generate alerts.
        """
        alerts = []

        # 1. Retrieval score drop
        score_delta = previous.avg_retrieval_score - current.avg_retrieval_score
        if score_delta > self.thresholds["score_drop_critical"]:
            alerts.append(DegradationAlert(
                signal="retrieval_score",
                severity="critical",
                message=f"Retrieval score dropped significantly: "
                        f"{previous.avg_retrieval_score:.3f} → {current.avg_retrieval_score:.3f}",
                current_value=current.avg_retrieval_score,
                previous_value=previous.avg_retrieval_score,
            ))
        elif score_delta > self.thresholds["score_drop_warning"]:
            alerts.append(DegradationAlert(
                signal="retrieval_score",
                severity="warning",
                message=f"Retrieval score trending down: "
                        f"{previous.avg_retrieval_score:.3f} → {current.avg_retrieval_score:.3f}",
                current_value=current.avg_retrieval_score,
                previous_value=previous.avg_retrieval_score,
            ))

        # 2. Abstention rate rising
        abstention_delta = current.abstention_rate - previous.abstention_rate
        if abstention_delta > self.thresholds["abstention_rise_warning"]:
            alerts.append(DegradationAlert(
                signal="abstention_rate",
                severity="warning",
                message=f"Abstention rate rising: "
                        f"{previous.abstention_rate:.0%} → {current.abstention_rate:.0%}",
                current_value=current.abstention_rate,
                previous_value=previous.abstention_rate,
            ))

        # 3. Negative feedback rising
        feedback_delta = current.negative_feedback_rate - previous.negative_feedback_rate
        if feedback_delta > self.thresholds["feedback_drop_warning"]:
            alerts.append(DegradationAlert(
                signal="negative_feedback",
                severity="critical",
                message=f"Negative feedback spike: "
                        f"{previous.negative_feedback_rate:.0%} → {current.negative_feedback_rate:.0%}",
                current_value=current.negative_feedback_rate,
                previous_value=previous.negative_feedback_rate,
            ))

        # 4. Latency creep
        latency_delta = current.p95_latency_ms - previous.p95_latency_ms
        if latency_delta > self.thresholds["latency_rise_warning"]:
            alerts.append(DegradationAlert(
                signal="p95_latency",
                severity="warning",
                message=f"p95 latency increasing: "
                        f"{previous.p95_latency_ms:.0f}ms → {current.p95_latency_ms:.0f}ms",
                current_value=current.p95_latency_ms,
                previous_value=previous.p95_latency_ms,
            ))

        # 5. Category-level regression
        for category in current.category_scores:
            if category in previous.category_scores:
                cat_delta = (
                    previous.category_scores[category]
                    - current.category_scores[category]
                )
                if cat_delta > self.thresholds["score_drop_warning"]:
                    alerts.append(DegradationAlert(
                        signal=f"category:{category}",
                        severity="warning",
                        message=f"Category '{category}' degrading: "
                                f"{previous.category_scores[category]:.3f} → "
                                f"{current.category_scores[category]:.3f}",
                        current_value=current.category_scores[category],
                        previous_value=previous.category_scores[category],
                    ))

        return alerts


# ─── Usage ───
monitor = SilentDegradationMonitor()

last_week = WeeklyMetrics(
    week_label="2025-W11",
    avg_retrieval_score=0.82,
    abstention_rate=0.05,
    negative_feedback_rate=0.08,
    p95_latency_ms=1800,
    total_queries=5000,
    category_scores={"api_docs": 0.88, "billing": 0.79, "onboarding": 0.91},
)

this_week = WeeklyMetrics(
    week_label="2025-W12",
    avg_retrieval_score=0.77,   # dropped
    abstention_rate=0.12,       # rose
    negative_feedback_rate=0.12,
    p95_latency_ms=2100,
    total_queries=4800,
    category_scores={"api_docs": 0.85, "billing": 0.65, "onboarding": 0.90},
    #                                     ↑ billing dropped significantly
)

alerts = monitor.check(this_week, last_week)
for alert in alerts:
    icon = "🔴" if alert.severity == "critical" else "🟡"
    print(f"  {icon} [{alert.signal}] {alert.message}")

# Output:
#   🔴 [retrieval_score] Retrieval score dropped significantly: 0.820 → 0.770
#   🟡 [abstention_rate] Abstention rate rising: 5% → 12%
#   🟡 [p95_latency] p95 latency increasing: 1800ms → 2100ms
#   🟡 [category:billing] Category 'billing' degrading: 0.790 → 0.650
```

---

## Pitfalls & Common Mistakes

| Mistake                          | Impact                                      | Fix                                             |
| -------------------------------- | ------------------------------------------- | ----------------------------------------------- |
| **No drift monitoring**          | Index silently becomes stale                | Hash-based change detection, scheduled checks   |
| **Full reindex on every change** | Expensive, slow, risky                      | Incremental updates: only re-embed changed docs |
| **Eval set also goes stale**     | Regression tests pass but answers are wrong | Version eval sets alongside source docs         |
| **No baseline metrics**          | Can't tell if a change helped or hurt       | Record metrics before every change              |
| **Manual freshness checks**      | Gets forgotten, humans are unreliable       | Automate with cron jobs / CI pipeline           |

---

## Key Takeaways

1. **Index drift is the #1 production RAG risk** — documents change, index doesn't.
2. **Hash-based change detection** is cheap and effective for catching drift.
3. **Track freshness** — know how old your indexed content is.
4. **Run regression tests after every change** — compare against baseline metrics.
5. **Automate everything** — drift checks, freshness reports, regression tests.
