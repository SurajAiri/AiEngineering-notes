# Corrective RAG (CRAG) — Retrieval Quality Assessment & Fallback

## Why It Matters

Standard RAG blindly trusts whatever the retriever returns — even if the chunks are irrelevant or contradictory. CRAG adds a **self-assessment layer** that evaluates retrieval quality and triggers corrective actions when the context is insufficient.

```
STANDARD RAG (blind trust):

  Query ──▶ Retrieve ──▶ Generate ──▶ Answer
                │
                └─ Even if chunks are garbage,
                   the LLM still tries to answer.
                   → Hallucination or nonsense.

CORRECTIVE RAG (CRAG):

  Query ──▶ Retrieve ──▶ EVALUATE QUALITY
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              CORRECT              INCORRECT / AMBIGUOUS
                    │                   │
                    ▼         ┌─────────┴─────────┐
             Use as-is        ▼                   ▼
                         Strip noise          Web search
                         Refine chunks        External KB
                              │                   │
                              └─────────┬─────────┘
                                        ▼
                                   Generate with
                                   corrected context
```

---

## CRAG Classification

```
CRAG evaluates each retrieved document into three categories:

  ┌────────────┬──────────────────────────────────────┐
  │ Category   │ Action Taken                         │
  ├────────────┼──────────────────────────────────────┤
  │ CORRECT    │ Use the document as context.          │
  │            │ Apply knowledge refinement to strip   │
  │            │ irrelevant parts.                     │
  ├────────────┼──────────────────────────────────────┤
  │ INCORRECT  │ Discard the document entirely.        │
  │            │ Trigger web search or fallback.       │
  ├────────────┼──────────────────────────────────────┤
  │ AMBIGUOUS  │ Partially relevant. Keep but also     │
  │            │ trigger supplementary search.         │
  └────────────┴──────────────────────────────────────┘

The key insight: Don't use ALL retrieved documents.
Grade each one and only use what's genuinely helpful.
```

---

## Implementation — CRAG System

```python
"""
Corrective RAG: evaluate retrieval quality and apply corrections.

Requirements: pip install openai sentence-transformers faiss-cpu numpy
"""

import json
import numpy as np
import faiss
from enum import Enum
from dataclasses import dataclass, field
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class RelevanceGrade(Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    AMBIGUOUS = "ambiguous"


@dataclass
class GradedChunk:
    text: str
    grade: RelevanceGrade
    confidence: float
    reason: str
    refined_text: str = ""  # stripped of irrelevant parts


@dataclass
class CRAGResult:
    answer: str
    graded_chunks: list[GradedChunk]
    used_web_search: bool
    web_results: list[str] = field(default_factory=list)


class CorrectiveRAG:
    """
    CRAG: Corrective Retrieval-Augmented Generation.

    1. Retrieve chunks
    2. Grade each chunk (correct / incorrect / ambiguous)
    3. Refine correct chunks (strip noise)
    4. If too many incorrect → fall back to web search
    5. Generate answer from cleaned context
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
        confidence_threshold: float = 0.5,
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.llm = OpenAI()
        self.llm_model = llm_model
        self.confidence_threshold = confidence_threshold
        self.chunks: list[str] = []
        self.index = None

    def ingest(self, chunks: list[str]):
        """Build vector index."""
        self.chunks = chunks
        embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def query(self, question: str, k: int = 5) -> CRAGResult:
        """Full CRAG pipeline."""
        # Step 1: Retrieve
        retrieved = self._retrieve(question, k=k)

        # Step 2: Grade each chunk
        graded = self._grade_chunks(question, retrieved)

        # Step 3: Decide action based on grades
        correct_chunks = [g for g in graded if g.grade == RelevanceGrade.CORRECT]
        ambiguous_chunks = [g for g in graded if g.grade == RelevanceGrade.AMBIGUOUS]
        incorrect_count = sum(1 for g in graded if g.grade == RelevanceGrade.INCORRECT)

        # Step 4: Refine correct chunks
        for chunk in correct_chunks:
            chunk.refined_text = self._refine_chunk(question, chunk.text)

        web_results = []
        used_web = False

        # If majority is incorrect or no correct chunks, fall back
        if not correct_chunks and not ambiguous_chunks:
            web_results = self._web_search_fallback(question)
            used_web = True
        elif not correct_chunks and ambiguous_chunks:
            # Use ambiguous chunks but also supplement
            web_results = self._web_search_fallback(question)
            used_web = True

        # Step 5: Build context from refined chunks + web results
        context_parts = []
        for chunk in correct_chunks:
            context_parts.append(chunk.refined_text or chunk.text)
        for chunk in ambiguous_chunks:
            context_parts.append(f"[partially relevant] {chunk.text}")
        for wr in web_results:
            context_parts.append(f"[web source] {wr}")

        # Step 6: Generate answer
        answer = self._generate(question, context_parts)

        return CRAGResult(
            answer=answer,
            graded_chunks=graded,
            used_web_search=used_web,
            web_results=web_results,
        )

    def _retrieve(self, query: str, k: int) -> list[str]:
        """Standard vector retrieval."""
        if self.index is None:
            return []
        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        scores, indices = self.index.search(
            query_emb.reshape(1, -1).astype(np.float32), min(k, len(self.chunks))
        )
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    def _grade_chunks(self, query: str, chunks: list[str]) -> list[GradedChunk]:
        """Grade each chunk's relevance to the query using LLM."""
        graded = []
        for chunk in chunks:
            prompt = f"""Grade whether this document is relevant to the query.

Query: {query}

Document: {chunk}

Respond with JSON:
{{
    "grade": "correct" | "incorrect" | "ambiguous",
    "confidence": 0.0 to 1.0,
    "reason": "one sentence explanation"
}}

- "correct": Document directly answers or supports answering the query.
- "incorrect": Document is unrelated to the query.
- "ambiguous": Document is tangentially related but not directly helpful."""

            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )

            result = json.loads(response.choices[0].message.content)
            try:
                grade = RelevanceGrade(result.get("grade", "ambiguous"))
            except ValueError:
                grade = RelevanceGrade.AMBIGUOUS

            graded.append(GradedChunk(
                text=chunk,
                grade=grade,
                confidence=result.get("confidence", 0.5),
                reason=result.get("reason", ""),
            ))

        return graded

    def _refine_chunk(self, query: str, chunk: str) -> str:
        """
        Knowledge refinement: strip irrelevant parts of a correct chunk.
        Keep only the sentences that help answer the query.
        """
        prompt = f"""Extract ONLY the sentences from the document that are
directly relevant to answering this query. Remove filler, boilerplate,
and tangential content.

Query: {query}
Document: {chunk}

Return only the relevant sentences, nothing else."""

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()

    def _web_search_fallback(self, query: str) -> list[str]:
        """
        Fallback when internal retrieval fails.
        In production, this would call a search API (Bing, Google, Tavily).
        """
        # Placeholder — in production, integrate with a search API:
        # from tavily import TavilyClient
        # client = TavilyClient(api_key="...")
        # results = client.search(query, max_results=3)
        # return [r["content"] for r in results["results"]]

        return [f"[Web search placeholder for: {query}]"]

    def _generate(self, query: str, context_parts: list[str]) -> str:
        """Generate final answer from cleaned context."""
        if not context_parts:
            return "I don't have enough information to answer this question."

        context = "\n---\n".join(context_parts)
        prompt = f"""Answer the question using ONLY the provided context.
If the context is insufficient, say so.

Context:
{context}

Question: {query}"""

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# ─── Usage ───
if __name__ == "__main__":
    crag = CorrectiveRAG()

    chunks = [
        "FastAPI supports OAuth2 authentication with JWT tokens for API security.",
        "Python was created by Guido van Rossum in 1991.",
        "To configure rate limiting in FastAPI, use the slowapi library.",
        "The weather in Paris is typically mild in spring.",
        "FastAPI uses Pydantic models for request and response validation.",
    ]
    crag.ingest(chunks)

    result = crag.query("How does authentication work in FastAPI?")

    print(f"Answer: {result.answer}")
    print(f"Web search used: {result.used_web_search}")
    print(f"\nChunk grades:")
    for gc in result.graded_chunks:
        print(f"  [{gc.grade.value}] ({gc.confidence:.2f}) {gc.text[:60]}...")
        print(f"    Reason: {gc.reason}")
```

---

## Lighter Version: Score-Based CRAG (No LLM Grading)

```python
"""
Lightweight CRAG using retrieval scores instead of LLM grading.
Much cheaper — no extra LLM calls for evaluation.
"""


def score_based_crag(
    chunks: list[str],
    scores: list[float],
    correct_threshold: float = 0.6,
    ambiguous_threshold: float = 0.4,
) -> dict:
    """
    Grade chunks by their retrieval score.

    Returns:
        {"correct": [...], "ambiguous": [...], "incorrect": [...],
         "action": "use_internal" | "supplement" | "web_search"}
    """
    correct = []
    ambiguous = []
    incorrect = []

    for chunk, score in zip(chunks, scores):
        if score >= correct_threshold:
            correct.append((chunk, score))
        elif score >= ambiguous_threshold:
            ambiguous.append((chunk, score))
        else:
            incorrect.append((chunk, score))

    # Decide action
    if correct:
        action = "use_internal"  # have good results
    elif ambiguous:
        action = "supplement"    # partially relevant, need more
    else:
        action = "web_search"   # nothing good, fall back

    return {
        "correct": correct,
        "ambiguous": ambiguous,
        "incorrect": incorrect,
        "action": action,
    }


# Example
result = score_based_crag(
    chunks=["OAuth2 in FastAPI...", "Python was created...", "Rate limiting..."],
    scores=[0.82, 0.25, 0.55],
)
print(f"Action: {result['action']}")
print(f"Correct chunks: {len(result['correct'])}")
print(f"Incorrect chunks: {len(result['incorrect'])}")
# Action: use_internal
# Correct chunks: 1
# Incorrect chunks: 1
```

---

## CRAG vs Agentic RAG

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│ Feature             │ CRAG                 │ Agentic RAG          │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Focus               │ Quality of retrieved │ Strategy selection    │
│                     │ documents            │ for retrieval         │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Evaluation target   │ Each document        │ Overall retrieval     │
│                     │ individually         │ sufficiency           │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Correction action   │ Grade + filter +     │ Rewrite query,        │
│                     │ refine + fallback    │ decompose, expand     │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Web search          │ Core feature         │ Optional              │
│                     │ (when internal fails)│                       │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Iteration           │ Single pass          │ Multi-iteration loop  │
│                     │ (grade once)         │                       │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Best for            │ Noisy corpus with    │ Complex queries that  │
│                     │ irrelevant docs      │ need query refinement │
└─────────────────────┴──────────────────────┴──────────────────────┘

They complement each other: Use CRAG to filter bad chunks,
use Agentic RAG to improve query strategy. Can be combined.
```

---

## Pitfalls & Common Mistakes

| Mistake                                    | Impact                       | Fix                                                                    |
| ------------------------------------------ | ---------------------------- | ---------------------------------------------------------------------- |
| **LLM grading every chunk is expensive**   | 5 extra LLM calls per query  | Use score-based grading for most; LLM grading for low-confidence cases |
| **Web search returns noisy results**       | Introduces external noise    | Filter web results with same grading pipeline                          |
| **Refining too aggressively**              | Strips useful context        | Only refine chunks that are partially relevant                         |
| **No fallback when web search also fails** | System still hallucinates    | Implement proper abstention ("I don't know")                           |
| **Grading prompt is too strict**           | Most chunks marked incorrect | Tune grading criteria; allow partially relevant chunks                 |

---

## Key Takeaways

1. **CRAG adds a quality gate** — don't blindly trust retrieval results.
2. **Grade each chunk individually** — not all retrieved chunks are equally useful.
3. **Knowledge refinement** strips noise from partially relevant chunks.
4. **Web search is the fallback** when internal retrieval fails — integrate with a search API.
5. **Score-based grading is 10x cheaper** than LLM grading — use it for the common case.
6. **CRAG and Agentic RAG are complementary** — CRAG filters chunks, Agentic RAG improves queries.
