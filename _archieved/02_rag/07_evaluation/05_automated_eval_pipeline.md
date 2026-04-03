# Automated RAG Evaluation Pipeline

## 🟢 How to Approach This Topic

> **Why this matters for your job:** In production, you can't manually check RAG quality after every change. You need automated pipelines that run evaluations on every code push, catch regressions before deployment, and track quality over time. This is what separates a prototype from a production system.

**Prerequisites:** Understand RAGAS and DeepEval from `04_evaluation_frameworks.md`. Have a golden evaluation dataset from `01_ground_truth_eval_data.md`.

**Reading order:**

1. Pipeline architecture — 10 min
2. Implement the evaluation runner — 15 min
3. CI/CD integration — 15 min
4. Regression detection — 10 min

**⏱️ Core concept: 30 min | Full exploration: 1.5 hours**

---

## Pipeline Architecture

```
┌────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│ Code Push  │────▶│ CI Triggered │────▶│ Run RAG on    │────▶│ Compute      │
│ (PR/merge) │     │ (GitHub      │     │ eval dataset  │     │ RAGAS metrics│
│            │     │  Actions)    │     │ (50-200 Q&A)  │     │              │
└────────────┘     └──────────────┘     └───────────────┘     └──────┬───────┘
                                                                      │
                          ┌───────────────┐     ┌──────────────┐      │
                          │ Block deploy  │◀────│ Compare to   │◀─────┘
                          │ if regression │     │ baseline     │
                          │ detected      │     │ scores       │
                          └───────────────┘     └──────────────┘
```

---

## Step 1: Golden Test Set

```python
"""
Store your golden test set as a versioned JSON file.
Check it into your repo so it's always available in CI.
"""
import json

# tests/eval_data/golden_v1.json
golden_test_set = [
    {
        "id": "q001",
        "question": "What is the return policy?",
        "ground_truth": "Items can be returned within 30 days of purchase with receipt.",
        "difficulty": "easy",
        "category": "policy",
    },
    {
        "id": "q002",
        "question": "Can I return an item bought on sale?",
        "ground_truth": "Sale items can be returned within 14 days for store credit only.",
        "difficulty": "medium",
        "category": "policy",
    },
    {
        "id": "q003",
        "question": "What happens if I lost my receipt but have the credit card statement?",
        "ground_truth": "Returns without receipt are accepted with credit card statement, limited to store credit.",
        "difficulty": "hard",
        "category": "policy",
    },
    # ... 50-200 questions covering your domain
]

with open("tests/eval_data/golden_v1.json", "w") as f:
    json.dump(golden_test_set, f, indent=2)
```

---

## Step 2: Evaluation Runner

```python
"""
tests/eval_runner.py
Core evaluation script that runs your RAG pipeline against the golden set.
"""
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset


@dataclass
class EvalResult:
    timestamp: str
    commit_sha: str
    dataset_version: str
    num_samples: int
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    duration_seconds: float
    per_sample: list  # detailed per-question results


def run_evaluation(
    pipeline_fn,
    golden_path: str = "tests/eval_data/golden_v1.json",
    commit_sha: str = "local",
) -> EvalResult:
    """
    Run full RAGAS evaluation against golden test set.

    Args:
        pipeline_fn: Function that takes a question string and returns
                     {"answer": str, "contexts": list[str]}
        golden_path: Path to golden test set JSON
        commit_sha: Git commit SHA for tracking
    """
    # Load golden set
    with open(golden_path) as f:
        golden = json.load(f)

    # Run pipeline on all questions
    start = time.time()
    questions, answers, contexts, truths = [], [], [], []

    for item in golden:
        result = pipeline_fn(item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        truths.append(item["ground_truth"])

    # RAGAS evaluation
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": truths,
    })

    ragas_result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    duration = time.time() - start

    # Build result
    df = ragas_result.to_pandas()
    per_sample = []
    for i, row in df.iterrows():
        per_sample.append({
            "id": golden[i]["id"],
            "question": golden[i]["question"],
            "difficulty": golden[i].get("difficulty", "unknown"),
            "faithfulness": row["faithfulness"],
            "answer_relevancy": row["answer_relevancy"],
            "context_precision": row["context_precision"],
            "context_recall": row["context_recall"],
        })

    return EvalResult(
        timestamp=datetime.utcnow().isoformat(),
        commit_sha=commit_sha,
        dataset_version=os.path.basename(golden_path),
        num_samples=len(golden),
        avg_faithfulness=float(df["faithfulness"].mean()),
        avg_answer_relevancy=float(df["answer_relevancy"].mean()),
        avg_context_precision=float(df["context_precision"].mean()),
        avg_context_recall=float(df["context_recall"].mean()),
        duration_seconds=round(duration, 2),
        per_sample=per_sample,
    )


def save_result(result: EvalResult, output_dir: str = "tests/eval_results"):
    """Save evaluation results for historical tracking."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{result.timestamp.replace(':', '-')}_{result.commit_sha[:8]}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)
    return filepath
```

---

## Step 3: Regression Detector

```python
"""
tests/regression_check.py
Compare current eval results to baseline and detect regressions.
"""
import json
import sys
from pathlib import Path


# Thresholds — tune these based on your experience
THRESHOLDS = {
    "faithfulness": 0.80,       # minimum acceptable score
    "answer_relevancy": 0.75,
    "context_precision": 0.70,
    "context_recall": 0.75,
}

# Maximum allowed regression from baseline
MAX_REGRESSION = 0.05  # 5% drop triggers failure


def load_baseline(baseline_path: str = "tests/eval_data/baseline.json") -> dict:
    """Load the baseline scores from last accepted evaluation."""
    if not Path(baseline_path).exists():
        return None
    with open(baseline_path) as f:
        return json.load(f)


def check_regression(current: dict, baseline: dict = None) -> tuple[bool, list[str]]:
    """
    Check if current results meet thresholds and haven't regressed.
    Returns: (passed: bool, issues: list[str])
    """
    issues = []

    # Check absolute thresholds
    for metric, threshold in THRESHOLDS.items():
        key = f"avg_{metric}"
        score = current.get(key, 0)
        if score < threshold:
            issues.append(
                f"FAIL: {metric} = {score:.3f} (below threshold {threshold})"
            )

    # Check regression from baseline
    if baseline:
        for metric in THRESHOLDS:
            key = f"avg_{metric}"
            current_score = current.get(key, 0)
            baseline_score = baseline.get(key, 0)
            drop = baseline_score - current_score
            if drop > MAX_REGRESSION:
                issues.append(
                    f"REGRESSION: {metric} dropped {drop:.3f} "
                    f"({baseline_score:.3f} → {current_score:.3f})"
                )

    passed = len(issues) == 0
    return passed, issues


def main():
    """CLI entry point for CI/CD."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", help="Path to current eval result JSON")
    parser.add_argument("--baseline", default="tests/eval_data/baseline.json")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Update baseline with current results if passing")
    args = parser.parse_args()

    with open(args.result_file) as f:
        current = json.load(f)

    baseline = load_baseline(args.baseline)
    passed, issues = check_regression(current, baseline)

    # Print report
    print("\n" + "=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)
    print(f"Commit:     {current.get('commit_sha', 'unknown')}")
    print(f"Samples:    {current.get('num_samples', 0)}")
    print(f"Duration:   {current.get('duration_seconds', 0):.1f}s")
    print("-" * 60)
    for metric in THRESHOLDS:
        key = f"avg_{metric}"
        score = current.get(key, 0)
        threshold = THRESHOLDS[metric]
        status = "✅" if score >= threshold else "❌"
        baseline_str = ""
        if baseline:
            baseline_score = baseline.get(key, 0)
            diff = score - baseline_score
            baseline_str = f" (Δ {diff:+.3f})"
        print(f"  {status} {metric}: {score:.3f} (threshold: {threshold}){baseline_str}")
    print("-" * 60)

    if issues:
        print("\n⚠️  ISSUES:")
        for issue in issues:
            print(f"  • {issue}")
        print()

    if passed and args.update_baseline:
        import shutil
        shutil.copy(args.result_file, args.baseline)
        print(f"✅ Baseline updated: {args.baseline}")

    print(f"\nResult: {'PASSED ✅' if passed else 'FAILED ❌'}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
```

---

## Step 4: GitHub Actions Integration

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation

on:
  pull_request:
    paths:
      - "src/rag/**" # RAG pipeline code
      - "prompts/**" # Prompt templates
      - "config/**" # Configuration changes
  push:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run RAG evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python -m tests.eval_runner \
            --golden tests/eval_data/golden_v1.json \
            --commit ${{ github.sha }} \
            --output tests/eval_results/current.json

      - name: Check for regressions
        run: |
          python tests/regression_check.py \
            tests/eval_results/current.json \
            --baseline tests/eval_data/baseline.json

      - name: Upload results as artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: eval-results
          path: tests/eval_results/

      - name: Comment PR with results
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const result = JSON.parse(fs.readFileSync('tests/eval_results/current.json'));
            const body = `## RAG Evaluation Results
            | Metric | Score | Threshold |
            |--------|-------|-----------|
            | Faithfulness | ${result.avg_faithfulness.toFixed(3)} | 0.80 |
            | Answer Relevancy | ${result.avg_answer_relevancy.toFixed(3)} | 0.75 |
            | Context Precision | ${result.avg_context_precision.toFixed(3)} | 0.70 |
            | Context Recall | ${result.avg_context_recall.toFixed(3)} | 0.75 |

            Duration: ${result.duration_seconds}s | Samples: ${result.num_samples}`;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });
```

---

## With DeepEval (Alternative CI Approach)

```python
"""
tests/test_rag_regression.py
DeepEval's pytest-native approach for CI/CD.
Run with: deepeval test run tests/test_rag_regression.py
"""
import json
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric


def load_golden_set():
    with open("tests/eval_data/golden_v1.json") as f:
        return json.load(f)


def run_pipeline(question: str) -> dict:
    """Your RAG pipeline. Replace with actual implementation."""
    # from src.rag.pipeline import RAGPipeline
    # pipeline = RAGPipeline()
    # return pipeline.query(question)
    return {"answer": "placeholder", "contexts": ["placeholder"]}


GOLDEN = load_golden_set()


@pytest.mark.parametrize("item", GOLDEN, ids=[g["id"] for g in GOLDEN])
def test_rag_quality(item):
    result = run_pipeline(item["question"])

    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["ground_truth"],
        retrieval_context=result["contexts"],
    )

    faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")
    relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")

    assert_test(test_case, [faithfulness, relevancy])
```

```bash
# In CI
deepeval test run tests/test_rag_regression.py --verbose -x  # -x stops on first failure
```

---

## Tracking Quality Over Time

```python
"""
Simple quality dashboard — load historical eval results and plot trends.
"""
import json
from pathlib import Path


def load_history(results_dir: str = "tests/eval_results") -> list[dict]:
    """Load all historical evaluation results."""
    results = []
    for f in sorted(Path(results_dir).glob("*.json")):
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def print_trend_report(history: list[dict]):
    """Print quality trend summary."""
    if len(history) < 2:
        print("Need at least 2 evaluation runs for trend analysis.")
        return

    latest = history[-1]
    previous = history[-2]

    print("\n📈 Quality Trend Report")
    print("=" * 50)
    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        key = f"avg_{metric}"
        curr = latest.get(key, 0)
        prev = previous.get(key, 0)
        diff = curr - prev
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "→"
        print(f"  {metric}: {curr:.3f} {arrow} ({diff:+.3f})")


# ─── Usage ───
# history = load_history()
# print_trend_report(history)
```

---

## Common Pitfalls

| Pitfall                     | Impact                        | Fix                                            |
| --------------------------- | ----------------------------- | ---------------------------------------------- |
| No golden test set          | Can't evaluate anything       | Start with 20 questions, grow to 100+          |
| Testing only in development | Prod quality unknown          | Add CI/CD evaluation pipeline                  |
| No baseline comparison      | Don't know if quality dropped | Store and compare baselines                    |
| Too few test cases          | Results are noisy             | Need 50+ for reliable metrics                  |
| Same eval set forever       | Doesn't cover new content     | Add new questions quarterly                    |
| Expensive eval in every PR  | Slow PRs, high cost           | Run lightweight eval in PRs, full eval nightly |

---

## Syllabus Mapping

Maps to `p2_rag_depth.md` 2.7 (Evaluation) and 2.8 (Production — continuous monitoring).
