# Versioned Documents

## Why This Matters

In production RAG systems, documents change over time. Policies get updated, API docs get new versions, contracts get amended. If your system doesn't handle versioning, you'll serve stale answers, mix old and new information, or lose the ability to answer questions about historical states.

---

## The Versioning Problem in RAG

```
                    Time →

    v1 (Jan 2024)         v2 (Jun 2024)         v3 (Dec 2024)
    ┌──────────┐          ┌──────────┐          ┌──────────┐
    │ Policy:  │          │ Policy:  │          │ Policy:  │
    │ 30-day   │   ───►   │ 14-day   │   ───►   │ 7-day    │
    │ returns  │          │ returns  │          │ returns  │
    └──────────┘          └──────────┘          └──────────┘

    User asks: "What's the return policy?"

    WRONG: "30-day returns" (stale v1 answer)
    RIGHT: "7-day returns"  (current v3 answer)

    But sometimes:
    User asks: "What was the return policy last year?"
    RIGHT: "14-day returns" (historical v2 answer)
```

---

## Versioning Strategies

### Strategy 1: Replace-and-Reindex (Simplest)

```
On update:
1. Delete old document chunks from vector store
2. Ingest new version
3. Re-embed and re-index

Pros: Simple, always current
Cons: No history, potential downtime during reindex
```

### Strategy 2: Version-Tagged Metadata

```
On update:
1. Mark old version as superseded
2. Ingest new version with version metadata
3. Filter at query time

Pros: Preserves history, no reindex needed
Cons: Index grows, need filter logic
```

### Strategy 3: Separate Indexes Per Version

```
On update:
1. Keep old index intact
2. Create new index for new version
3. Route queries to appropriate index

Pros: Clean separation, easy rollback
Cons: Resource-heavy, complex routing
```

---

## Simple Code — Version-Tagged Documents

```python
"""
Minimal example: versioned documents with metadata filtering.
"""

from datetime import datetime

# Each chunk stores version metadata
chunks = [
    {
        "text": "Our return policy allows 30-day returns with receipt.",
        "metadata": {
            "doc_id": "return_policy",
            "version": 1,
            "effective_date": "2024-01-01",
            "superseded_date": "2024-06-01",
            "is_current": False,
        }
    },
    {
        "text": "Our return policy allows 14-day returns with receipt.",
        "metadata": {
            "doc_id": "return_policy",
            "version": 2,
            "effective_date": "2024-06-01",
            "superseded_date": "2024-12-01",
            "is_current": False,
        }
    },
    {
        "text": "Our return policy allows 7-day returns with receipt.",
        "metadata": {
            "doc_id": "return_policy",
            "version": 3,
            "effective_date": "2024-12-01",
            "superseded_date": None,
            "is_current": True,
        }
    },
]


def retrieve_current(query: str, all_chunks: list[dict]) -> list[dict]:
    """Retrieve only current versions."""
    return [c for c in all_chunks if c["metadata"]["is_current"]]


def retrieve_as_of(query: str, all_chunks: list[dict], date_str: str) -> list[dict]:
    """Retrieve the version that was effective on a specific date."""
    target = datetime.strptime(date_str, "%Y-%m-%d")
    results = []
    for c in all_chunks:
        effective = datetime.strptime(c["metadata"]["effective_date"], "%Y-%m-%d")
        superseded = c["metadata"]["superseded_date"]
        if superseded:
            superseded = datetime.strptime(superseded, "%Y-%m-%d")
            if effective <= target < superseded:
                results.append(c)
        else:
            if effective <= target:
                results.append(c)
    return results


# Current query
current = retrieve_current("return policy", chunks)
print(f"Current: {current[0]['text']}")
# → "Our return policy allows 7-day returns with receipt."

# Historical query
historical = retrieve_as_of("return policy", chunks, "2024-08-15")
print(f"As of Aug 2024: {historical[0]['text']}")
# → "Our return policy allows 14-day returns with receipt."
```

---

## Production Code — Version Manager

```python
"""
Production versioned document manager for RAG systems.
Handles versioning, supersession, and point-in-time retrieval.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    CURRENT = "current"
    SUPERSEDED = "superseded"
    DRAFT = "draft"
    ARCHIVED = "archived"


@dataclass
class DocumentVersion:
    doc_id: str                    # stable ID across versions (e.g., "return_policy")
    version: int
    content: str
    content_hash: str
    status: VersionStatus
    effective_date: datetime
    superseded_date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    change_summary: str = ""       # what changed from previous version

    @property
    def version_id(self) -> str:
        """Unique ID for this specific version."""
        return f"{self.doc_id}::v{self.version}"


class DocumentVersionManager:
    """
    Manages document versions for RAG ingestion.
    Ensures only one current version per doc_id at any time.
    """

    def __init__(self):
        # doc_id → list of versions (ordered by version number)
        self._versions: dict[str, list[DocumentVersion]] = {}

    def _content_hash(self, content: str) -> str:
        normalized = ' '.join(content.split()).lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def add_version(
        self,
        doc_id: str,
        content: str,
        effective_date: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        change_summary: str = "",
    ) -> DocumentVersion | None:
        """
        Add a new version of a document.
        Automatically supersedes the current version.
        Returns None if content is identical to current version (no-op).
        """
        content_hash = self._content_hash(content)
        effective = effective_date or datetime.now(timezone.utc)

        # Check if content actually changed
        if doc_id in self._versions:
            current = self.get_current(doc_id)
            if current and current.content_hash == content_hash:
                logger.info(f"No content change for {doc_id}, skipping")
                return None

        # Supersede current version
        if doc_id in self._versions:
            for v in self._versions[doc_id]:
                if v.status == VersionStatus.CURRENT:
                    v.status = VersionStatus.SUPERSEDED
                    v.superseded_date = effective
                    logger.info(f"Superseded {v.version_id}")

        # Create new version
        version_num = len(self._versions.get(doc_id, [])) + 1
        new_version = DocumentVersion(
            doc_id=doc_id,
            version=version_num,
            content=content,
            content_hash=content_hash,
            status=VersionStatus.CURRENT,
            effective_date=effective,
            metadata=metadata or {},
            change_summary=change_summary,
        )

        self._versions.setdefault(doc_id, []).append(new_version)
        logger.info(f"Created {new_version.version_id}")
        return new_version

    def get_current(self, doc_id: str) -> DocumentVersion | None:
        """Get the current version of a document."""
        for v in reversed(self._versions.get(doc_id, [])):
            if v.status == VersionStatus.CURRENT:
                return v
        return None

    def get_all_current(self) -> list[DocumentVersion]:
        """Get all current document versions (for full reindex)."""
        results = []
        for versions in self._versions.values():
            for v in versions:
                if v.status == VersionStatus.CURRENT:
                    results.append(v)
        return results

    def get_as_of(self, doc_id: str, point_in_time: datetime) -> DocumentVersion | None:
        """Get the version that was effective at a specific point in time."""
        candidates = []
        for v in self._versions.get(doc_id, []):
            if v.effective_date <= point_in_time:
                if v.superseded_date is None or v.superseded_date > point_in_time:
                    candidates.append(v)

        if not candidates:
            return None
        # Return the latest effective version
        return max(candidates, key=lambda v: v.effective_date)

    def get_version_history(self, doc_id: str) -> list[dict]:
        """Get the full version history for a document."""
        return [
            {
                "version_id": v.version_id,
                "version": v.version,
                "status": v.status.value,
                "effective_date": v.effective_date.isoformat(),
                "superseded_date": v.superseded_date.isoformat() if v.superseded_date else None,
                "change_summary": v.change_summary,
                "content_hash": v.content_hash[:12],
            }
            for v in self._versions.get(doc_id, [])
        ]

    def prepare_for_indexing(self) -> list[dict]:
        """
        Prepare all current documents for vector store ingestion.
        Returns chunks with version metadata attached.
        """
        index_items = []
        for doc_ver in self.get_all_current():
            index_items.append({
                "content": doc_ver.content,
                "metadata": {
                    "doc_id": doc_ver.doc_id,
                    "version": doc_ver.version,
                    "version_id": doc_ver.version_id,
                    "effective_date": doc_ver.effective_date.isoformat(),
                    "is_current": True,
                    **doc_ver.metadata,
                }
            })
        return index_items


# ─── Usage ───
if __name__ == "__main__":
    vm = DocumentVersionManager()

    # Initial version
    vm.add_version(
        doc_id="return_policy",
        content="Returns accepted within 30 days with original receipt.",
        effective_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        metadata={"category": "policy", "department": "customer_service"},
        change_summary="Initial policy",
    )

    # Policy update
    vm.add_version(
        doc_id="return_policy",
        content="Returns accepted within 14 days. No receipt needed for exchanges.",
        effective_date=datetime(2024, 6, 1, tzinfo=timezone.utc),
        change_summary="Shortened return window, added no-receipt exchanges",
    )

    # Another update
    vm.add_version(
        doc_id="return_policy",
        content="Returns accepted within 7 days. Digital receipt required.",
        effective_date=datetime(2024, 12, 1, tzinfo=timezone.utc),
        change_summary="Shortened to 7 days, digital receipts only",
    )

    # No-op update (same content)
    result = vm.add_version(
        doc_id="return_policy",
        content="Returns accepted within 7 days. Digital receipt required.",
    )
    print(f"No-op result: {result}")  # None — content unchanged

    # Query current
    current = vm.get_current("return_policy")
    print(f"\nCurrent: {current.content}")

    # Query historical
    historical = vm.get_as_of(
        "return_policy",
        datetime(2024, 8, 15, tzinfo=timezone.utc)
    )
    print(f"Aug 2024: {historical.content}")

    # Version history
    print("\nVersion History:")
    for entry in vm.get_version_history("return_policy"):
        print(f"  v{entry['version']} ({entry['status']}): {entry['change_summary']}")

    # Prepare for indexing
    print(f"\nDocuments for indexing: {len(vm.prepare_for_indexing())}")
```

---

## Handling Version Conflicts in Retrieval

When retrieval returns chunks from multiple versions:

```python
def resolve_version_conflicts(chunks: list[dict]) -> list[dict]:
    """
    If retrieval returns chunks from multiple versions of the same doc,
    keep only the most recent version's chunks.
    """
    # Group by doc_id
    by_doc: dict[str, list[dict]] = {}
    for chunk in chunks:
        doc_id = chunk["metadata"]["doc_id"]
        by_doc.setdefault(doc_id, []).append(chunk)

    resolved = []
    for doc_id, doc_chunks in by_doc.items():
        # Keep only chunks from the latest version
        max_version = max(c["metadata"]["version"] for c in doc_chunks)
        for c in doc_chunks:
            if c["metadata"]["version"] == max_version:
                resolved.append(c)
            else:
                print(f"Dropped stale chunk from {doc_id} v{c['metadata']['version']}")

    return resolved
```

---

## Pitfalls & Common Mistakes

| Mistake                                   | Impact                                        | Fix                                                  |
| ----------------------------------------- | --------------------------------------------- | ---------------------------------------------------- |
| **No version metadata on chunks**         | Can't filter stale content at query time      | Always attach version + effective_date metadata      |
| **Not superseding old versions**          | Retrieval returns mixed old/new content       | Mark old versions as superseded; filter in retrieval |
| **Reindexing everything on every change** | Slow, expensive, causes downtime              | Use incremental updates: delete old chunks, add new  |
| **No content change detection**           | Re-embedding identical content wastes compute | Hash content and skip if unchanged                   |
| **Losing version history**                | Can't audit what changed or debug regressions | Keep all versions, mark status appropriately         |
| **Time-zone ignorance**                   | Effective dates compared incorrectly          | Always use UTC with timezone-aware datetimes         |

---

## Trade-offs

| Strategy                  | Complexity | Storage | Query Speed          | History |
| ------------------------- | ---------- | ------- | -------------------- | ------- |
| Replace & Reindex         | Low        | Low     | Fast                 | None    |
| Version Metadata + Filter | Medium     | Medium  | Medium (filter cost) | Full    |
| Separate Indexes          | High       | High    | Fast (per index)     | Full    |

**Recommendation**: Start with **version metadata + filter**. It balances simplicity with capability. Move to separate indexes only if you have strict latency requirements and many historical queries.

---

## Key Takeaways

1. **Every chunk needs version metadata** — doc_id, version number, effective date, is_current flag.
2. **Detect content changes before re-embedding** — hash comparison saves significant compute.
3. **Supersede, don't delete** — keep old versions for history and debugging.
4. **Filter at query time** — default to current versions; support historical queries when needed.
5. **Handle version conflicts post-retrieval** — if mixed versions come back, resolve to the latest.
