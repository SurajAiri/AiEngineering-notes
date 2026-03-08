# Faithfulness & Correctness

## Why It Matters

Retrieval metrics (Recall, MRR, nDCG) tell you if the right chunks were found. But that's not enough — the LLM must **faithfully use those chunks** to generate a correct answer. Faithfulness evaluation checks whether the answer is grounded in the provided context, not hallucinated.

```
THE GAP BETWEEN RETRIEVAL AND ANSWER QUALITY:

  Great Retrieval + Unfaithful LLM = Wrong Answer

  ┌─────────────────────────────────────────────┐
  │ Context (retrieved):                        │
  │   "The API rate limit is 1000 req/min."     │
  │                                             │
  │ LLM Answer:                                 │
  │   "The API rate limit is 5000 req/minute,   │
  │    which can be increased via premium plan." │
  │                                             │
  │ Problems:                                   │
  │   ❌ 5000 ≠ 1000 (contradiction)            │
  │   ❌ "premium plan" not in context           │
  │      (hallucinated detail)                  │
  └─────────────────────────────────────────────┘

FAITHFULNESS DIMENSIONS:

  ┌──────────────────┬──────────────────────────────────┐
  │ Dimension        │ What It Checks                   │
  ├──────────────────┼──────────────────────────────────┤
  │ Faithfulness     │ Is every claim in the answer     │
  │                  │ supported by the context?        │
  ├──────────────────┼──────────────────────────────────┤
  │ Correctness      │ Is the answer factually right    │
  │                  │ (compared to ground truth)?      │
  ├──────────────────┼──────────────────────────────────┤
  │ Citation Accuracy│ Do the cited sources actually     │
  │                  │ support the claims made?         │
  ├──────────────────┼──────────────────────────────────┤
  │ Completeness     │ Does the answer cover all        │
  │                  │ relevant info from the context?  │
  ├──────────────────┼──────────────────────────────────┤
  │ Refusal          │ Does the system correctly refuse  │
  │                  │ when it can't answer?            │
  └──────────────────┴──────────────────────────────────┘
```

---

## Faithfulness Evaluation (Context → Answer)

### Concept: Claim Decomposition

```
APPROACH: Break the answer into individual claims,
then verify each claim against the context.

ANSWER:
  "FastAPI supports async handlers and uses Pydantic for validation.
   It can serve 50,000 requests per second on a single core."

DECOMPOSED CLAIMS:
  1. "FastAPI supports async handlers"           → Check context
  2. "FastAPI uses Pydantic for validation"      → Check context
  3. "it can serve 50,000 req/s on single core"  → Check context

CONTEXT:
  "FastAPI is built on Starlette for async support and Pydantic
   for data validation."

VERIFICATION:
  1. ✅ Supported (async via Starlette)
  2. ✅ Supported (Pydantic mentioned)
  3. ❌ NOT in context (hallucinated performance claim)

FAITHFULNESS SCORE: 2/3 = 0.67
```

### Implementation

```python
"""
Faithfulness evaluation: verify LLM answers against retrieved context.

Requirements: pip install openai
"""

import json
from dataclasses import dataclass, field
from openai import OpenAI


@dataclass
class FaithfulnessResult:
    claims: list[str]
    supported: list[bool]
    explanations: list[str]
    score: float  # fraction of claims supported

    def summary(self) -> str:
        lines = [f"Faithfulness: {self.score:.0%} ({sum(self.supported)}/{len(self.claims)} claims supported)"]
        for claim, supported, explanation in zip(self.claims, self.supported, self.explanations):
            status = "✅" if supported else "❌"
            lines.append(f"  {status} {claim}")
            lines.append(f"     → {explanation}")
        return "\n".join(lines)


class FaithfulnessEvaluator:
    """Evaluate whether LLM answers are grounded in the provided context."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = OpenAI()
        self.model = model

    def evaluate(self, answer: str, context: str) -> FaithfulnessResult:
        """
        1. Decompose answer into individual claims
        2. Verify each claim against the context
        """
        claims = self._decompose_claims(answer)
        supported = []
        explanations = []

        for claim in claims:
            result = self._verify_claim(claim, context)
            supported.append(result["supported"])
            explanations.append(result["explanation"])

        score = sum(supported) / len(claims) if claims else 1.0

        return FaithfulnessResult(
            claims=claims,
            supported=supported,
            explanations=explanations,
            score=score,
        )

    def _decompose_claims(self, answer: str) -> list[str]:
        """Break an answer into individual factual claims."""
        prompt = f"""Break the following answer into individual factual claims.
Each claim should be a single, verifiable statement.
Ignore filler phrases, transitions, and opinions.

Answer: {answer}

Respond with JSON: {{"claims": ["claim1", "claim2", ...]}}"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("claims", [])

    def _verify_claim(self, claim: str, context: str) -> dict:
        """Check if a claim is supported by the context."""
        prompt = f"""Determine if the following claim is supported by the context.

Claim: {claim}

Context:
{context}

Rules:
- "supported": The context directly states or clearly implies this claim.
- "not_supported": The context does not contain information to verify this claim,
  OR the claim contradicts the context.

Respond with JSON:
{{"supported": true/false, "explanation": "one sentence reason"}}"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return json.loads(response.choices[0].message.content)


# ─── Usage ───
if __name__ == "__main__":
    evaluator = FaithfulnessEvaluator()

    context = """FastAPI is a modern Python web framework built on Starlette and Pydantic.
It supports async request handling through Python's asyncio. Pydantic handles
data validation and serialization. FastAPI automatically generates OpenAPI
documentation at /docs endpoint."""

    # Good answer (faithful)
    good_answer = "FastAPI uses Pydantic for data validation and generates API docs automatically."
    result = evaluator.evaluate(good_answer, context)
    print("=== Good Answer ===")
    print(result.summary())

    # Bad answer (contains hallucination)
    bad_answer = "FastAPI uses Pydantic for validation and can handle 100,000 concurrent connections out of the box."
    result = evaluator.evaluate(bad_answer, context)
    print("\n=== Bad Answer ===")
    print(result.summary())
```

---

## Correctness Evaluation (Answer vs Ground Truth)

```python
"""
Correctness evaluation: compare LLM answer against ground truth.
"""

import json
from dataclasses import dataclass
from openai import OpenAI


@dataclass
class CorrectnessResult:
    is_correct: bool
    score: float       # 0-1 (partial credit)
    explanation: str
    missing_info: list[str]   # info in ground truth but not in answer
    extra_info: list[str]     # info in answer but not in ground truth


class CorrectnessEvaluator:
    """Compare LLM answers against ground truth for factual correctness."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm = OpenAI()
        self.model = model

    def evaluate(
        self,
        answer: str,
        ground_truth: str,
        query: str,
    ) -> CorrectnessResult:
        """
        Compare answer to ground truth.
        Awards partial credit for partially correct answers.
        """
        prompt = f"""Compare the ANSWER to the GROUND TRUTH for the given QUERY.

QUERY: {query}
GROUND TRUTH: {ground_truth}
ANSWER: {answer}

Evaluate:
1. Is the answer factually correct compared to the ground truth?
2. What information from the ground truth is MISSING from the answer?
3. What information in the answer is NOT in the ground truth (could be hallucinated)?
4. Give a score from 0.0 to 1.0:
   - 1.0: Fully correct, covers all key points
   - 0.5: Partially correct, some key info missing or wrong
   - 0.0: Completely wrong or unrelated

Respond with JSON:
{{
    "is_correct": true/false,
    "score": 0.0-1.0,
    "explanation": "assessment",
    "missing_info": ["info1", "info2"],
    "extra_info": ["info1"]
}}"""

        response = self.llm.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )

        result = json.loads(response.choices[0].message.content)
        return CorrectnessResult(
            is_correct=result.get("is_correct", False),
            score=result.get("score", 0.0),
            explanation=result.get("explanation", ""),
            missing_info=result.get("missing_info", []),
            extra_info=result.get("extra_info", []),
        )


# Example
evaluator = CorrectnessEvaluator()
result = evaluator.evaluate(
    answer="The API rate limit is 1000 requests per minute.",
    ground_truth="The API rate limit is 1000 requests per minute per API key. Premium plans get 5000 req/min.",
    query="What is the API rate limit?",
)
print(f"Correct: {result.is_correct}, Score: {result.score}")
print(f"Missing: {result.missing_info}")
print(f"Extra: {result.extra_info}")
```

---

## Refusal Evaluation

```python
"""
Test whether the system correctly refuses to answer unanswerable queries.
"""

from dataclasses import dataclass


@dataclass
class RefusalTestCase:
    query: str
    should_refuse: bool  # True = system should say "I don't know"
    context: str         # Retrieved context (may be irrelevant)


def evaluate_refusal(
    test_case: RefusalTestCase,
    system_answer: str,
) -> dict:
    """
    Check if the system correctly refused or answered.

    Four outcomes:
    - True Positive:  Should refuse, did refuse  (good)
    - True Negative:  Should answer, did answer  (good)
    - False Positive: Should answer, did refuse  (over-cautious)
    - False Negative: Should refuse, did answer  (hallucination risk!)
    """
    # Simple heuristic: check if answer contains refusal signals
    refusal_phrases = [
        "i don't have enough information",
        "i cannot find",
        "the provided context doesn't contain",
        "i'm unable to answer",
        "not enough information",
        "cannot determine",
        "no information available",
    ]

    did_refuse = any(
        phrase in system_answer.lower()
        for phrase in refusal_phrases
    )

    if test_case.should_refuse and did_refuse:
        return {"outcome": "true_positive", "correct": True}
    elif not test_case.should_refuse and not did_refuse:
        return {"outcome": "true_negative", "correct": True}
    elif not test_case.should_refuse and did_refuse:
        return {"outcome": "false_positive", "correct": False,
                "issue": "Over-cautious: refused to answer when it should have"}
    else:  # should_refuse and not did_refuse
        return {"outcome": "false_negative", "correct": False,
                "issue": "DANGEROUS: answered when it should have refused (hallucination risk)"}


def refusal_metrics(outcomes: list[dict]) -> dict:
    """Compute precision and recall for refusal detection."""
    tp = sum(1 for o in outcomes if o["outcome"] == "true_positive")
    tn = sum(1 for o in outcomes if o["outcome"] == "true_negative")
    fp = sum(1 for o in outcomes if o["outcome"] == "false_positive")
    fn = sum(1 for o in outcomes if o["outcome"] == "false_negative")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        "accuracy": (tp + tn) / len(outcomes) if outcomes else 0,
        "refusal_precision": precision,
        "refusal_recall": recall,
        "false_negatives (hallucination risk)": fn,
        "false_positives (over-cautious)": fp,
    }


# Example
test_cases = [
    RefusalTestCase(
        query="What is the API rate limit?",
        should_refuse=False,
        context="The API rate limit is 1000 req/min.",
    ),
    RefusalTestCase(
        query="What is the CEO's favorite color?",
        should_refuse=True,
        context="The company was founded in 2015.",
    ),
]

outcomes = [
    evaluate_refusal(test_cases[0], "The API rate limit is 1000 req/min."),
    evaluate_refusal(test_cases[1], "I don't have enough information to answer that."),
]

print(refusal_metrics(outcomes))
# {'accuracy': 1.0, 'refusal_precision': 1.0, 'refusal_recall': 1.0, ...}
```

---

## Partial Answer Detection

```
NOT ALL ANSWERS ARE FULLY RIGHT OR FULLY WRONG:

  Query: "What are the authentication methods?"

  Ground truth: "OAuth2, API keys, and SAML"

  Answer A: "OAuth2 and API keys"                → PARTIAL (missed SAML)
  Answer B: "OAuth2, API keys, and SAML"          → COMPLETE ✅
  Answer C: "Bearer tokens and basic auth"         → WRONG ❌
  Answer D: "OAuth2, API keys, and certificates"   → PARTIAL + HALLUCINATED

  SCORING PARTIAL ANSWERS:

  Answer A: 2/3 of ground truth covered           → Score: 0.67
  Answer B: 3/3 of ground truth covered           → Score: 1.0
  Answer C: 0/3 of ground truth covered           → Score: 0.0
  Answer D: 2/3 correct + 1 hallucinated claim    → Score: 0.5
                                                    (penalize for extras)
```

---

## Full RAG Evaluation Pipeline

```python
"""
End-to-end RAG evaluation: retrieval + faithfulness + correctness.
"""

from dataclasses import dataclass


@dataclass
class RAGEvalResult:
    """Complete evaluation of a single query through the RAG pipeline."""
    query: str
    # Retrieval metrics
    recall_at_5: float
    mrr: float
    # Generation metrics
    faithfulness_score: float
    correctness_score: float
    did_refuse: bool
    should_refuse: bool

    @property
    def retrieval_ok(self) -> bool:
        return self.recall_at_5 >= 0.5

    @property
    def generation_ok(self) -> bool:
        return self.faithfulness_score >= 0.8 and self.correctness_score >= 0.7

    @property
    def refusal_ok(self) -> bool:
        return self.did_refuse == self.should_refuse


def aggregate_rag_eval(results: list[RAGEvalResult]) -> dict:
    """Aggregate RAG evaluation results."""
    n = len(results)
    return {
        "queries_evaluated": n,
        # Retrieval
        "avg_recall@5": sum(r.recall_at_5 for r in results) / n,
        "avg_mrr": sum(r.mrr for r in results) / n,
        "retrieval_pass_rate": sum(r.retrieval_ok for r in results) / n,
        # Generation
        "avg_faithfulness": sum(r.faithfulness_score for r in results) / n,
        "avg_correctness": sum(r.correctness_score for r in results) / n,
        "generation_pass_rate": sum(r.generation_ok for r in results) / n,
        # Refusal
        "refusal_accuracy": sum(r.refusal_ok for r in results) / n,
        # Overall
        "full_pass_rate": sum(
            r.retrieval_ok and r.generation_ok and r.refusal_ok
            for r in results
        ) / n,
    }
```

---

## Pitfalls & Common Mistakes

| Mistake                                   | Impact                                          | Fix                                                  |
| ----------------------------------------- | ----------------------------------------------- | ---------------------------------------------------- |
| **Only evaluating retrieval**             | Good retrieval ≠ good answers                   | Always evaluate faithfulness and correctness too     |
| **Using the same LLM to evaluate itself** | Bias — models tend to approve their own outputs | Use a stronger model for evaluation, or human eval   |
| **Binary correct/incorrect**              | Misses partially correct answers                | Use graded scoring (0-1)                             |
| **No refusal testing**                    | System hallucinates on unanswerable queries     | Include 10%+ unanswerable queries in eval set        |
| **Evaluating only on easy queries**       | Hides poor performance on hard queries          | Stratify evaluation by difficulty                    |
| **Ignoring citation accuracy**            | Answers cite wrong sources                      | Verify that cited chunks actually support the claims |

---

## Key Takeaways

1. **Faithfulness** = is every claim in the answer supported by the context?
2. **Correctness** = does the answer match the ground truth?
3. **Both are needed** — an answer can be faithful-to-context but wrong (stale context), or correct but unfaithful (hallucinated the right answer).
4. **Claim decomposition** is the best approach for faithfulness — break answer into claims, verify each.
5. **Test refusal** — a system that always answers is dangerous; it should say "I don't know" when appropriate.
6. **Partial credit matters** — most answers aren't fully right or fully wrong.

---

## Popular Libraries

### Quick Example — RAGAS Faithfulness + Answer Correctness

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_correctness
from ragas import EvaluationDataset

eval_data = EvaluationDataset.from_list([
    {
        "user_input": "When was Python created?",
        "retrieved_contexts": ["Python was created by Guido van Rossum and first released in 1991."],
        "response": "Python was created in 1991 by Guido van Rossum.",
        "reference": "Python was first released in 1991.",
    },
])

results = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_correctness],
)
print(results)
# faithfulness: 1.0 (every claim grounded in context)
# answer_correctness: 0.95 (matches ground truth)
```

---

## Common Questions

### Q: How do I detect hallucinations in RAG answers?

**A:** Measure **faithfulness** — decompose the answer into individual claims, then check if each claim is supported by the retrieved context. If a claim has no supporting evidence in the context, it's likely hallucinated. RAGAS automates this with LLM-as-judge.

### Q: Should I use GPT-4 to evaluate GPT-4 outputs?

**A:** Ideally use a **stronger or different** model for evaluation. If you're generating answers with GPT-4o-mini, evaluate faithfulness with GPT-4o. Same-model evaluation has known bias (models tend to rate their own outputs higher). For high-stakes applications, include human evaluation on a sample.
