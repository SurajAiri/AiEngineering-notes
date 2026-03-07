# Agentic RAG — Self-Reflective & Iterative Retrieval

## Why It Matters

Standard RAG is a **single-pass pipeline**: retrieve → rank → generate. But what if:

- The first retrieval misses critical information?
- The query is ambiguous and needs clarification?
- The answer requires combining results from multiple retrieval strategies?

Agentic RAG adds a **reasoning loop** where the system evaluates its own retrieval quality and decides what to do next — retry with a different query, switch retrieval strategy, or acknowledge insufficient information.

```
STANDARD RAG (one-shot):

  Query ──▶ Retrieve(k=5) ──▶ Generate ──▶ Answer

  Problem: If retrieval is bad, the answer is bad.
  No chance to recover.

AGENTIC RAG (iterative):

  Query ──▶ Retrieve ──▶ Evaluate Quality
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              SUFFICIENT           INSUFFICIENT
                    │                   │
                    ▼         ┌─────────┴─────────┐
               Generate       │    Decide Action   │
                    │         │                     │
                    ▼         ▼         ▼           ▼
               Answer    Rewrite    Change       Expand
                         Query     Strategy      Search
                           │         │             │
                           └─────────┴─────────────┘
                                     │
                                     ▼
                              Retrieve Again
                              (back to evaluate)
```

---

## Core Concept: The Agent Loop

```
AGENTIC RAG DECISION FLOW:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  1. RETRIEVE: Get initial chunks                 │
  │       ↓                                          │
  │  2. REFLECT: Are these chunks sufficient?        │
  │       ↓                                          │
  │  3. DECIDE:                                      │
  │     ┌──────────────────────────────────────┐     │
  │     │ If HIGH confidence → Generate answer │     │
  │     │ If LOW confidence  → Try another     │     │
  │     │   strategy:                          │     │
  │     │   • Rewrite query (more specific)    │     │
  │     │   • Decompose into sub-queries       │     │
  │     │   • Switch from vector to keyword    │     │
  │     │   • Expand k (retrieve more)         │     │
  │     │   • Search a different index         │     │
  │     │ If MAX RETRIES hit → Abstain / say   │     │
  │     │   "I don't have enough information"  │     │
  │     └──────────────────────────────────────┘     │
  │       ↓                                          │
  │  4. LOOP back to step 1 (with new strategy)      │
  │                                                  │
  └──────────────────────────────────────────────────┘

  Safety: MAX_ITERATIONS prevents infinite loops (typically 3).
```

---

## Implementation — Agentic RAG System

```python
"""
Agentic RAG: self-reflective retrieval with iterative refinement.

Requirements: pip install openai sentence-transformers faiss-cpu numpy
"""

import json
import numpy as np
import faiss
from enum import Enum
from dataclasses import dataclass, field
from openai import OpenAI
from sentence_transformers import SentenceTransformer


class Action(Enum):
    ANSWER = "answer"            # Sufficient context, generate answer
    REWRITE = "rewrite"          # Rewrite query to be more specific
    DECOMPOSE = "decompose"      # Break into sub-queries
    EXPAND = "expand"            # Retrieve more chunks (increase k)
    ABSTAIN = "abstain"          # Can't answer with available data


@dataclass
class RetrievalState:
    """Tracks the state across retrieval iterations."""
    query: str
    original_query: str
    chunks: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    iteration: int = 0
    action_history: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)


class AgenticRAG:
    """
    RAG system with self-reflective retrieval loop.

    The agent:
    1. Retrieves chunks
    2. Evaluates if they're sufficient
    3. Decides next action (answer, rewrite, decompose, expand, abstain)
    4. Loops until confident or max iterations reached
    """

    MAX_ITERATIONS = 3

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_model: str = "gpt-4o-mini",
    ):
        self.embedder = SentenceTransformer(model_name)
        self.llm = OpenAI()
        self.llm_model = llm_model
        self.chunks: list[str] = []
        self.index = None

    def ingest(self, chunks: list[str]):
        """Index chunks for vector retrieval."""
        self.chunks = chunks
        embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def query(self, question: str) -> dict:
        """
        Main entry point. Runs the agentic retrieval loop.

        Returns:
            {"answer": str, "sources": list[str], "iterations": int,
             "actions": list[str]}
        """
        state = RetrievalState(
            query=question,
            original_query=question,
        )

        while state.iteration < self.MAX_ITERATIONS:
            state.iteration += 1

            # Step 1: Retrieve
            new_chunks, new_scores = self._retrieve(state.query, k=5)

            # Merge with existing chunks (dedup)
            for chunk, score in zip(new_chunks, new_scores):
                if chunk not in state.chunks:
                    state.chunks.append(chunk)
                    state.scores.append(score)

            # Step 2: Reflect — evaluate retrieval quality
            action = self._reflect(state)
            state.action_history.append(action.value)

            # Step 3: Act on decision
            if action == Action.ANSWER:
                answer = self._generate_answer(state)
                return {
                    "answer": answer,
                    "sources": state.chunks[:5],
                    "iterations": state.iteration,
                    "actions": state.action_history,
                }
            elif action == Action.REWRITE:
                state.query = self._rewrite_query(state)
            elif action == Action.DECOMPOSE:
                state = self._decompose_and_retrieve(state)
            elif action == Action.EXPAND:
                # Retrieve more with broader k
                extra_chunks, extra_scores = self._retrieve(state.query, k=10)
                for chunk, score in zip(extra_chunks, extra_scores):
                    if chunk not in state.chunks:
                        state.chunks.append(chunk)
                        state.scores.append(score)
            elif action == Action.ABSTAIN:
                return {
                    "answer": "I don't have enough information to answer this question accurately.",
                    "sources": state.chunks,
                    "iterations": state.iteration,
                    "actions": state.action_history,
                }

        # Max iterations reached — try to answer with what we have
        answer = self._generate_answer(state)
        return {
            "answer": answer,
            "sources": state.chunks[:5],
            "iterations": state.iteration,
            "actions": state.action_history,
        }

    def _retrieve(self, query: str, k: int = 5) -> tuple[list[str], list[float]]:
        """Vector similarity search."""
        if self.index is None:
            return [], []
        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        scores, indices = self.index.search(
            query_emb.reshape(1, -1).astype(np.float32), min(k, len(self.chunks))
        )
        chunks = [self.chunks[i] for i in indices[0] if i < len(self.chunks)]
        return chunks, [float(s) for s in scores[0]]

    def _reflect(self, state: RetrievalState) -> Action:
        """
        LLM evaluates retrieval quality and decides next action.

        This is the core "agent" decision — the system reflects on whether
        the retrieved context is sufficient to answer the query.
        """
        context_preview = "\n---\n".join(state.chunks[:5])

        prompt = f"""You are evaluating whether retrieved context is sufficient to answer a query.

ORIGINAL QUERY: {state.original_query}
CURRENT QUERY: {state.query}
ITERATION: {state.iteration}/{self.MAX_ITERATIONS}
PREVIOUS ACTIONS: {state.action_history}

RETRIEVED CONTEXT (top chunks):
{context_preview}

Evaluate and choose ONE action:
- "answer": Context is sufficient to answer the query well.
- "rewrite": Query is too vague; rewrite it to be more specific.
- "decompose": Query is complex; break into sub-queries.
- "expand": Context is partially relevant; retrieve more.
- "abstain": Context is clearly irrelevant; cannot answer from this data.

Respond with JSON: {{"action": "...", "reason": "one sentence explanation"}}"""

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)
        action_str = result.get("action", "answer")

        try:
            return Action(action_str)
        except ValueError:
            return Action.ANSWER

    def _rewrite_query(self, state: RetrievalState) -> str:
        """LLM rewrites the query to improve retrieval."""
        prompt = f"""The following query didn't retrieve good results.
Rewrite it to be more specific and likely to match relevant documents.

Original query: {state.original_query}
Current query: {state.query}

Respond with just the rewritten query, nothing else."""

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _decompose_and_retrieve(self, state: RetrievalState) -> RetrievalState:
        """Break query into sub-queries and retrieve for each."""
        prompt = f"""Break this complex query into 2-3 simpler sub-queries that can each
be answered independently. Return as JSON: {{"sub_queries": ["q1", "q2", "q3"]}}

Query: {state.original_query}"""

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)
        sub_queries = result.get("sub_queries", [state.query])
        state.sub_queries = sub_queries

        for sq in sub_queries:
            sq_chunks, sq_scores = self._retrieve(sq, k=3)
            for chunk, score in zip(sq_chunks, sq_scores):
                if chunk not in state.chunks:
                    state.chunks.append(chunk)
                    state.scores.append(score)

        return state

    def _generate_answer(self, state: RetrievalState) -> str:
        """Generate final answer from collected context."""
        context = "\n---\n".join(state.chunks[:8])

        prompt = f"""Answer the question using ONLY the context provided.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {state.original_query}

Answer:"""

        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# ─── Usage ───
if __name__ == "__main__":
    agent = AgenticRAG()

    chunks = [
        "FastAPI supports async request handling with Python's asyncio.",
        "Authentication in FastAPI uses OAuth2 with JWT tokens.",
        "Rate limiting can be implemented with slowapi middleware.",
        "FastAPI generates OpenAPI docs automatically at /docs.",
        "Database connections should use connection pooling with SQLAlchemy.",
        "CORS is configured via CORSMiddleware in FastAPI.",
        "Deployment options include Docker, Kubernetes, and serverless.",
        "Environment variables should be managed with pydantic-settings.",
    ]
    agent.ingest(chunks)

    result = agent.query("How do I secure and deploy a FastAPI application?")
    print(f"Answer: {result['answer']}")
    print(f"Iterations: {result['iterations']}")
    print(f"Actions taken: {result['actions']}")
    print(f"Sources used: {len(result['sources'])}")
```

---

## Simpler Version: Self-Reflective Retrieval Without Full Agent

```python
"""
Lightweight self-reflective retrieval: check if retrieval is good enough
before generating an answer. No LLM calls for reflection — uses heuristics.
"""

import numpy as np


class SelfReflectiveRetriever:
    """
    Uses score-based heuristics instead of LLM to evaluate retrieval quality.
    Much cheaper and faster than full agentic approach.
    """

    def __init__(
        self,
        min_score: float = 0.4,
        min_chunks: int = 2,
        score_gap_threshold: float = 0.15,
    ):
        self.min_score = min_score
        self.min_chunks = min_chunks
        self.score_gap_threshold = score_gap_threshold

    def evaluate_retrieval(
        self,
        scores: list[float],
    ) -> dict:
        """
        Heuristic evaluation of retrieval quality.

        Returns:
            {"quality": "good"|"marginal"|"poor", "reason": str,
             "suggestion": str}
        """
        if not scores:
            return {
                "quality": "poor",
                "reason": "No results returned",
                "suggestion": "rewrite",
            }

        # Check top score
        top_score = scores[0]
        if top_score < self.min_score:
            return {
                "quality": "poor",
                "reason": f"Top score {top_score:.3f} below threshold {self.min_score}",
                "suggestion": "rewrite",
            }

        # Check how many chunks are above threshold
        good_chunks = sum(1 for s in scores if s >= self.min_score)
        if good_chunks < self.min_chunks:
            return {
                "quality": "marginal",
                "reason": f"Only {good_chunks} chunks above threshold",
                "suggestion": "expand",
            }

        # Check score cliff (big gap between scores)
        if len(scores) >= 2:
            gap = scores[0] - scores[1]
            if gap > self.score_gap_threshold:
                return {
                    "quality": "marginal",
                    "reason": f"Large score gap ({gap:.3f}) suggests topic mismatch",
                    "suggestion": "decompose",
                }

        return {
            "quality": "good",
            "reason": "Sufficient relevant chunks found",
            "suggestion": "answer",
        }


# Example
evaluator = SelfReflectiveRetriever()

# Case 1: Good retrieval
print(evaluator.evaluate_retrieval([0.85, 0.78, 0.72, 0.45, 0.30]))
# {'quality': 'good', 'reason': 'Sufficient relevant chunks found', 'suggestion': 'answer'}

# Case 2: Poor retrieval
print(evaluator.evaluate_retrieval([0.25, 0.20, 0.18]))
# {'quality': 'poor', 'reason': 'Top score 0.250 below threshold 0.4', 'suggestion': 'rewrite'}

# Case 3: One good result, rest garbage
print(evaluator.evaluate_retrieval([0.90, 0.30, 0.25, 0.20]))
# {'quality': 'marginal', 'reason': 'Large score gap...', 'suggestion': 'decompose'}
```

---

## Pitfalls & Common Mistakes

| Mistake                         | Impact                                        | Fix                                                                       |
| ------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------- |
| **No max iteration limit**      | Infinite loop, runaway costs                  | Always cap at 2-3 iterations                                              |
| **LLM reflection is expensive** | 2-3x latency and cost per query               | Use heuristic evaluator for most queries, LLM only for complex ones       |
| **Agent always rewrites**       | Never actually answers, loops forever         | Bias toward answering — require strong evidence for rewrites              |
| **No dedup across iterations**  | Same chunks retrieved multiple times          | Track seen chunks across iterations                                       |
| **Over-engineering the agent**  | Complex state machines that are hard to debug | Start with simple heuristic evaluation, add LLM reflection only if needed |

---

## Key Takeaways

1. **Agentic RAG adds a feedback loop** — retrieve, evaluate, adapt, retry.
2. **Start with heuristics** (score thresholds), not LLM reflection — cheaper and faster.
3. **Always cap iterations** (2-3 max) — unbounded loops are expensive.
4. **Three key actions**: rewrite query, decompose query, expand retrieval depth.
5. **Most queries don't need iteration** — use agentic flow only when standard retrieval is insufficient.
