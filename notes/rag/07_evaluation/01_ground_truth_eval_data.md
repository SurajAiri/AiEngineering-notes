# Ground Truth & Evaluation Data

## Why It Matters

You can't evaluate a RAG system without knowing what the **right answer** is. Ground truth evaluation data is the foundation of every RAG metric — Recall@k, MRR, nDCG, faithfulness, and correctness all require a reference to compare against.

```
WITHOUT GROUND TRUTH:
  "The system returned 5 chunks. Are they good?"
  → You have NO WAY to know.

WITH GROUND TRUTH:
  Query: "What's the API rate limit?"
  Expected answer: "1000 requests/minute per API key"
  Relevant doc IDs: [doc_42, doc_87]

  Retrieved: [doc_42 ✅, doc_15 ❌, doc_87 ✅, doc_33 ❌, doc_9 ❌]
  → Recall@5 = 2/2 = 100% (found both relevant docs)
  → MRR = 1/1 = 1.0 (first relevant doc at position 1)
```

---

## Components of a Golden Evaluation Set

```
A COMPLETE EVALUATION EXAMPLE:

  {
    "query": "How do I configure SSL certificates?",
    "query_type": "how-to",
    "difficulty": "medium",
    "expected_answer": "Use certbot to generate Let's Encrypt certificates.
                        Configure nginx with ssl_certificate and
                        ssl_certificate_key directives.",
    "relevant_doc_ids": ["nginx-ssl-guide-section-3", "certbot-docs-quickstart"],
    "relevant_passages": [
        "Run certbot --nginx to automatically configure SSL...",
        "Add ssl_certificate /path/to/cert.pem to nginx.conf..."
    ],
    "irrelevant_doc_ids": ["nginx-basic-setup", "apache-ssl-guide"],
    "metadata": {
        "created_at": "2024-06-15",
        "domain": "devops",
        "requires_multi_hop": false
    }
  }
```

---

## Building Golden Query Sets

```python
"""
Create and manage golden evaluation datasets for RAG.

Requirements: pip install openai
"""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from openai import OpenAI


class QueryDifficulty(Enum):
    EASY = "easy"          # Single chunk, exact match possible
    MEDIUM = "medium"      # Requires understanding, may need 2 chunks
    HARD = "hard"          # Multi-hop, requires synthesis across chunks
    ADVERSARIAL = "adversarial"  # Designed to trick the system


class QueryType(Enum):
    FACTUAL = "factual"        # "What is X?"
    HOW_TO = "how_to"          # "How do I do X?"
    COMPARISON = "comparison"  # "X vs Y?"
    MULTI_HOP = "multi_hop"    # Requires connecting multiple facts
    UNANSWERABLE = "unanswerable"  # Not in corpus — should abstain


@dataclass
class GoldenExample:
    query: str
    expected_answer: str
    relevant_doc_ids: list[str]
    query_type: QueryType
    difficulty: QueryDifficulty
    relevant_passages: list[str] = field(default_factory=list)
    irrelevant_doc_ids: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def id(self) -> str:
        return hashlib.md5(self.query.encode()).hexdigest()[:12]


class GoldenDatasetBuilder:
    """Build and manage golden evaluation datasets."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm = OpenAI()
        self.llm_model = llm_model
        self.examples: list[GoldenExample] = []

    def add_manual(self, example: GoldenExample):
        """Add a manually created evaluation example."""
        self.examples.append(example)

    def generate_from_chunk(
        self,
        chunk: str,
        doc_id: str,
        num_queries: int = 3,
    ) -> list[GoldenExample]:
        """
        Auto-generate evaluation queries from a document chunk.

        This gives you a starting point — ALWAYS review and edit
        the generated examples.
        """
        prompt = f"""Given this document chunk, generate {num_queries} diverse
evaluation queries that can be answered from this text.

For each query, provide:
1. A natural question a user might ask
2. The expected answer (from the text only)
3. The difficulty level (easy/medium/hard)
4. The query type (factual/how_to/comparison)

Document chunk:
{chunk}

Respond with JSON:
{{
  "queries": [
    {{
      "query": "...",
      "expected_answer": "...",
      "difficulty": "easy|medium|hard",
      "query_type": "factual|how_to|comparison"
    }}
  ]
}}"""

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        result = json.loads(response.choices[0].message.content)
        examples = []

        for q in result.get("queries", []):
            example = GoldenExample(
                query=q["query"],
                expected_answer=q["expected_answer"],
                relevant_doc_ids=[doc_id],
                relevant_passages=[chunk],
                query_type=QueryType(q.get("query_type", "factual")),
                difficulty=QueryDifficulty(q.get("difficulty", "medium")),
                tags=["auto-generated"],
            )
            examples.append(example)
            self.examples.append(example)

        return examples

    def generate_unanswerable(self, domain: str, num: int = 3) -> list[GoldenExample]:
        """Generate queries that SHOULD NOT be answerable from the corpus."""
        prompt = f"""Generate {num} questions about {domain} that a user might reasonably
ask, but that are unlikely to be in a typical documentation corpus.

These should be the kind of questions where the system should say
"I don't have enough information" rather than hallucinate.

Respond with JSON: {{"queries": ["q1", "q2", "q3"]}}"""

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
        )

        result = json.loads(response.choices[0].message.content)
        examples = []

        for q in result.get("queries", []):
            example = GoldenExample(
                query=q,
                expected_answer="[UNANSWERABLE - system should abstain]",
                relevant_doc_ids=[],
                query_type=QueryType.UNANSWERABLE,
                difficulty=QueryDifficulty.ADVERSARIAL,
                tags=["unanswerable", "auto-generated"],
            )
            examples.append(example)
            self.examples.append(example)

        return examples

    def save(self, path: str):
        """Save dataset to JSON."""
        data = []
        for ex in self.examples:
            d = asdict(ex)
            d["query_type"] = ex.query_type.value
            d["difficulty"] = ex.difficulty.value
            data.append(d)

        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str):
        """Load dataset from JSON."""
        data = json.loads(Path(path).read_text())
        for d in data:
            d["query_type"] = QueryType(d["query_type"])
            d["difficulty"] = QueryDifficulty(d["difficulty"])
            self.examples.append(GoldenExample(**d))

    def stats(self) -> dict:
        """Dataset statistics."""
        from collections import Counter
        return {
            "total": len(self.examples),
            "by_difficulty": dict(Counter(e.difficulty.value for e in self.examples)),
            "by_type": dict(Counter(e.query_type.value for e in self.examples)),
            "auto_generated": sum(1 for e in self.examples if "auto-generated" in e.tags),
            "manual": sum(1 for e in self.examples if "auto-generated" not in e.tags),
        }


# ─── Usage ───
if __name__ == "__main__":
    builder = GoldenDatasetBuilder()

    # Manual: always include the most important queries
    builder.add_manual(GoldenExample(
        query="What is the API rate limit?",
        expected_answer="1000 requests per minute per API key",
        relevant_doc_ids=["api-reference-rate-limits"],
        query_type=QueryType.FACTUAL,
        difficulty=QueryDifficulty.EASY,
    ))

    builder.add_manual(GoldenExample(
        query="How do I migrate from v2 to v3 of the API?",
        expected_answer="Update the base URL, change auth from API key to OAuth2, update response parsing for new JSON structure",
        relevant_doc_ids=["migration-guide-v3", "api-v3-changelog"],
        query_type=QueryType.HOW_TO,
        difficulty=QueryDifficulty.HARD,
    ))

    # Auto-generate from chunks (review these!)
    chunk = "The default timeout for API calls is 30 seconds. You can customize it by setting the TIMEOUT_MS environment variable. Maximum allowed timeout is 120 seconds."
    auto_examples = builder.generate_from_chunk(chunk, doc_id="api-config-timeouts")

    # Generate unanswerable queries
    unanswerable = builder.generate_unanswerable("REST API documentation")

    print(f"Dataset stats: {builder.stats()}")
    builder.save("golden_eval_set.json")
```

---

## Query Difficulty Stratification

```
WHY STRATIFY BY DIFFICULTY:

  A system that scores 90% on easy queries but 20% on hard
  queries has a very different profile than one scoring 60%
  on everything.

  ┌────────────────┬─────────────────────────────────────┐
  │ Difficulty     │ Characteristics                     │
  ├────────────────┼─────────────────────────────────────┤
  │ EASY           │ Single chunk, keyword overlap,      │
  │                │ factual lookup                      │
  │                │ "What is the default port?"          │
  ├────────────────┼─────────────────────────────────────┤
  │ MEDIUM         │ May need 2 chunks, some reasoning,  │
  │                │ paraphrased queries                  │
  │                │ "How do I change the port?"           │
  ├────────────────┼─────────────────────────────────────┤
  │ HARD           │ Multi-hop, synthesis across docs,   │
  │                │ comparison queries                   │
  │                │ "Which port config affects both      │
  │                │  the API and the worker service?"    │
  ├────────────────┼─────────────────────────────────────┤
  │ ADVERSARIAL    │ Unanswerable, misleading, edge      │
  │                │ cases, negation                      │
  │                │ "What port does the service NOT      │
  │                │  support?"                           │
  └────────────────┴─────────────────────────────────────┘

  RECOMMENDED DISTRIBUTION:
    Easy:        30%
    Medium:      40%
    Hard:        20%
    Adversarial: 10%
```

---

## Temporal Evaluation Sets

```
WHY TEMPORAL SETS MATTER:

  Scenario: API docs updated quarterly.

  Q1 Eval Set: Tests against Q1 docs (version 2.1)
  Q2 Eval Set: Tests against Q2 docs (version 2.2)

  If you use Q1 eval set against Q2 index:
  - Some answers changed → detection of "correct regression"
  - Some queries now have new relevant docs → changed recall

  Actions:
  1. Keep eval sets versioned alongside doc versions
  2. Run both old and new eval sets against new index
  3. Old set catches regressions
  4. New set validates new content is retrievable
```

```python
"""Simple temporal eval set management."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class TemporalEvalSet:
    name: str
    version: str
    created_at: datetime
    valid_until: datetime | None  # None = still current
    examples: list[GoldenExample]

    @property
    def is_current(self) -> bool:
        if self.valid_until is None:
            return True
        return datetime.now() < self.valid_until

    def compare_with(self, newer_set: "TemporalEvalSet") -> dict:
        """Compare two eval sets to identify changed queries."""
        old_queries = {e.query for e in self.examples}
        new_queries = {e.query for e in newer_set.examples}

        return {
            "added_queries": new_queries - old_queries,
            "removed_queries": old_queries - new_queries,
            "unchanged_queries": old_queries & new_queries,
            "old_count": len(self.examples),
            "new_count": len(newer_set.examples),
        }
```

---

## Pitfalls & Common Mistakes

| Mistake                        | Impact                                                 | Fix                                                       |
| ------------------------------ | ------------------------------------------------------ | --------------------------------------------------------- |
| **No eval set at all**         | Can't measure improvements or regressions              | Create minimum 50 golden examples before optimizing       |
| **Only easy queries**          | High scores that don't reflect real-world performance  | Stratify: 30% easy, 40% medium, 20% hard, 10% adversarial |
| **Only auto-generated**        | LLM-generated queries may not match real user patterns | Mix auto-generated with real user queries from logs       |
| **No unanswerable queries**    | Can't test abstention capability                       | Include 10% unanswerable queries                          |
| **Eval set never updated**     | Becomes stale as documents change                      | Version eval sets; update alongside doc changes           |
| **Judging by a single metric** | Recall@5 looks great but precision is terrible         | Track multiple metrics across difficulty levels           |

---

## Key Takeaways

1. **Ground truth comes first** — you literally cannot evaluate without it.
2. **50+ examples minimum** — fewer gives noisy, unreliable metrics.
3. **Stratify by difficulty** — a system that aces easy queries but fails hard ones is brittle.
4. **Include unanswerable queries** — test that the system knows when to say "I don't know."
5. **Auto-generate, then review** — LLMs can bootstrap eval sets, but humans must validate.
6. **Version your eval sets** — tie them to document versions for tracking regressions.

---

## Popular Libraries

| Library   | Purpose                                | Install                |
| --------- | -------------------------------------- | ---------------------- |
| RAGAS     | Auto-generate eval test sets from docs | `pip install ragas`    |
| DeepEval  | Test-case generation + evaluation      | `pip install deepeval` |
| LangSmith | Manual annotation + dataset management | Via LangChain Cloud    |

### Quick Example — Generate Test Set with RAGAS

```python
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))

# Generate eval questions from your documents
generator = TestsetGenerator(llm=generator_llm)
testset = generator.generate_with_langchain_docs(
    documents=your_langchain_docs,  # Your loaded documents
    testset_size=50,
)

# Each test case has: question, ground_truth, contexts, metadata
df = testset.to_pandas()
print(df[["question", "ground_truth"]].head())
df.to_csv("eval_set.csv", index=False)  # Save for reuse
```

---

## Common Questions

### Q: How many eval examples do I need?

**A:** **50 minimum** for rough signal, **200+** for reliable metrics. Stratify: ~30% easy (direct lookup), ~40% medium (synthesis across 2-3 chunks), ~20% hard (reasoning required), ~10% unanswerable. Quality matters more than quantity — 50 well-crafted examples > 500 auto-generated ones reviewed by nobody.

### Q: Can I fully automate eval set creation?

**A:** You can **bootstrap** with LLMs (RAGAS or GPT-4 to generate question/answer pairs from your docs), but you MUST have a human review pass. LLM-generated questions tend to be simpler than real user questions and may miss edge cases. Use auto-generation for the initial set, then supplement with real user queries from production logs.
