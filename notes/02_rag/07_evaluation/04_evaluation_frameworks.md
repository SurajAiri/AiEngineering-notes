# Evaluation Frameworks for RAG

## 🟢 How to Approach This Topic

> **Why this matters for your job:** You'll be asked "how do you know your RAG is working?" in every interview. Knowing RAGAS, DeepEval, and tracing platforms (LangSmith, LangFuse, Phoenix) separates you from engineers who only build but can't measure. These frameworks turn manual "does it look right?" into automated, repeatable evaluation.

**Prerequisites:** Understand retrieval metrics (recall@k, MRR, nDCG) from `02_retrieval_evaluation.md` and faithfulness/correctness from `03_faithfulness_correctness.md`.

**Reading order:**

1. RAGAS (primary framework, most popular) — 25 min
2. DeepEval (alternative, good for CI/CD) — 15 min
3. LangSmith (tracing + evaluation, LangChain ecosystem) — 15 min
4. LangFuse (open-source alternative to LangSmith) — 10 min
5. Arize Phoenix (observability + evaluation) — 10 min

**⏱️ Core concept (RAGAS): 30 min | Full exploration: 2–3 hours**

---

## Framework Landscape

```
┌──────────────────────────────────────────────────────────────────┐
│                  RAG Evaluation Frameworks                       │
├──────────────────────┬───────────────────────────────────────────┤
│   EVAL LIBRARIES     │        OBSERVABILITY PLATFORMS            │
│   (compute metrics)  │     (trace, log, evaluate in prod)       │
│                      │                                           │
│   ┌──────────┐       │    ┌───────────┐   ┌──────────┐          │
│   │  RAGAS   │       │    │ LangSmith │   │ LangFuse │          │
│   │ (most    │       │    │ (closed   │   │ (open    │          │
│   │ popular) │       │    │ source,   │   │ source,  │          │
│   └──────────┘       │    │ LangChain)│   │ self-    │          │
│   ┌──────────┐       │    └───────────┘   │ hosted)  │          │
│   │ DeepEval │       │    ┌───────────┐   └──────────┘          │
│   │ (CI/CD   │       │    │  Arize    │                         │
│   │ focused) │       │    │  Phoenix  │                         │
│   └──────────┘       │    │ (open     │                         │
│                      │    │ source)   │                         │
│                      │    └───────────┘                         │
└──────────────────────┴───────────────────────────────────────────┘

Use eval libraries for: computing metrics, batch evaluation, regression tests
Use platforms for: production tracing, debugging, continuous monitoring
Best practice: Use BOTH together (RAGAS for metrics + LangFuse for tracing)
```

---

## Quick Decision Framework

| Need                                     | Best Tool         | Why                                       |
| ---------------------------------------- | ----------------- | ----------------------------------------- |
| Quick evaluation of RAG quality          | **RAGAS**         | Most metrics out of the box, simple API   |
| CI/CD integration + unit test style      | **DeepEval**      | pytest-like, good assertions, CI friendly |
| LangChain ecosystem + commercial support | **LangSmith**     | Native integration, best debugger         |
| Open-source tracing + self-hosted        | **LangFuse**      | Deploy your own, privacy-friendly         |
| Deep observability + embeddings analysis | **Arize Phoenix** | Embedding drift, cluster visualization    |
| Just getting started                     | **RAGAS**         | Easiest to learn, most tutorials          |

---

## 1. RAGAS (Retrieval Augmented Generation Assessment)

> **The most popular RAG evaluation framework.** If you learn only one, learn this.

### Installation

```bash
pip install ragas langchain-openai
```

### Core RAGAS Metrics

| Metric                  | What It Measures                               | Range | Uses LLM? |
| ----------------------- | ---------------------------------------------- | ----- | --------- |
| `faithfulness`          | Is the answer supported by context?            | 0–1   | Yes       |
| `answer_relevancy`      | Does the answer address the question?          | 0–1   | Yes       |
| `context_precision`     | Are relevant chunks ranked higher?             | 0–1   | Yes       |
| `context_recall`        | Are all relevant facts retrieved?              | 0–1   | Yes       |
| `answer_correctness`    | Does the answer match ground truth?            | 0–1   | Yes       |
| `context_entity_recall` | Are key entities from ground truth in context? | 0–1   | No        |

```
                    RAGAS Metric Map

                 ┌─────────────────┐
                 │    Question      │
                 └────────┬────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
       ┌──────────┐ ┌──────────┐ ┌──────────┐
       │ Context  │ │ Answer   │ │ Ground   │
       │ (chunks) │ │ (LLM)   │ │ Truth    │
       └────┬─────┘ └────┬─────┘ └────┬─────┘
            │             │            │
     ┌──────┴──────┐      │     ┌──────┴──────┐
     │ context_    │      │     │ context_    │
     │ precision   │      │     │ recall      │
     └─────────────┘      │     └─────────────┘
                          │
              ┌───────────┼───────────┐
              │           │           │
       ┌──────┴──────┐  ┌┴──────────┐│
       │faithfulness │  │answer_    ││
       │(answer vs   │  │relevancy  ││
       │ context)    │  │(answer vs ││
       └─────────────┘  │ question) ││
                        └───────────┘│
                    ┌─────────────────┘
                    │
             ┌──────┴──────┐
             │answer_      │
             │correctness  │
             │(answer vs   │
             │ ground truth)│
             └─────────────┘
```

### Basic RAGAS Evaluation

```python
"""
Evaluate your RAG pipeline with RAGAS.
pip install ragas langchain-openai datasets
"""
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from datasets import Dataset

# Prepare evaluation data
# Each sample needs: question, answer, contexts, ground_truth
eval_data = {
    "question": [
        "What is the capital of France?",
        "How does photosynthesis work?",
        "What is the speed of light?",
    ],
    "answer": [
        "The capital of France is Paris.",
        "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen.",
        "The speed of light is approximately 3 × 10^8 meters per second.",
    ],
    "contexts": [
        ["Paris is the capital and largest city of France."],
        ["Photosynthesis is the process by which plants convert light energy into chemical energy, producing glucose from CO2 and water."],
        ["Light travels at approximately 299,792,458 meters per second in a vacuum."],
    ],
    "ground_truth": [
        "Paris is the capital of France.",
        "Photosynthesis is the process where plants use sunlight, water, and carbon dioxide to produce glucose and oxygen.",
        "The speed of light in vacuum is 299,792,458 m/s.",
    ],
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ],
)

# View overall scores
print(results)
# {'faithfulness': 0.95, 'answer_relevancy': 0.92, ...}

# View per-sample scores
df = results.to_pandas()
print(df[["question", "faithfulness", "answer_relevancy"]].to_string())
```

### Evaluate Your Actual RAG Pipeline with RAGAS

```python
"""
Plug RAGAS into your existing RAG pipeline.
This evaluates the entire pipeline end-to-end.
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset


def evaluate_rag_pipeline(pipeline, eval_questions, ground_truths=None):
    """
    Run your RAG pipeline on evaluation questions and score with RAGAS.

    Args:
        pipeline: Your RAG pipeline with a .query(question) -> {answer, contexts} method
        eval_questions: List of test questions
        ground_truths: Optional list of expected answers
    """
    questions = []
    answers = []
    contexts = []
    truths = []

    for i, question in enumerate(eval_questions):
        result = pipeline.query(question)

        questions.append(question)
        answers.append(result["answer"])
        contexts.append(result["contexts"])  # list of retrieved chunk texts
        if ground_truths:
            truths.append(ground_truths[i])

    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = truths

    dataset = Dataset.from_dict(data)

    # Choose metrics based on what you have
    metrics = [faithfulness, answer_relevancy]
    if ground_truths:
        metrics.append(context_precision)

    return evaluate(dataset, metrics=metrics)


# ─── Usage ───
# results = evaluate_rag_pipeline(
#     pipeline=my_rag,
#     eval_questions=["What is X?", "How does Y work?"],
#     ground_truths=["X is ...", "Y works by ..."],
# )
# print(results.to_pandas())
```

### RAGAS Synthetic Test Data Generation

```python
"""
Generate evaluation data automatically from your documents.
Huge time saver — creates question/answer/context triples from your corpus.
"""
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader

# Load your documents
loader = DirectoryLoader("./docs/", glob="**/*.md")
documents = loader.load()

# Create test set generator
generator = TestsetGenerator.from_langchain(
    generator_llm=ChatOpenAI(model="gpt-4o-mini"),
    critic_llm=ChatOpenAI(model="gpt-4o"),  # use a stronger model for critique
    embeddings=OpenAIEmbeddings(),
)

# Generate test set with distribution of difficulty levels
testset = generator.generate_with_langchain_docs(
    documents,
    test_size=20,  # number of test samples
    distributions={
        simple: 0.5,        # straightforward factual questions
        reasoning: 0.25,    # requires combining information
        multi_context: 0.25, # needs multiple retrieved chunks
    },
)

# Convert to pandas for inspection
test_df = testset.to_pandas()
print(test_df[["question", "ground_truth", "evolution_type"]].head())

# Save for reuse
test_df.to_csv("eval_testset.csv", index=False)
```

### Common Pitfalls with RAGAS

| Pitfall                             | Why It's a Problem         | Fix                               |
| ----------------------------------- | -------------------------- | --------------------------------- |
| Using GPT-3.5 as evaluator          | Weak at claim verification | Use GPT-4o or GPT-4o-mini minimum |
| Not checking per-sample scores      | Average hides outliers     | Always inspect the DataFrame      |
| Only using auto-generated test sets | May not cover edge cases   | Mix auto + manual test cases      |
| Running RAGAS once and done         | Quality drifts over time   | Automate in CI/CD (see §5)        |
| Ignoring context_recall metric      | Retrieval issues invisible | Always include retrieval metrics  |

---

## 2. DeepEval

> **Best for CI/CD integration.** Works like pytest for LLM evaluation.

### Installation

```bash
pip install deepeval
```

### Basic DeepEval Evaluation

```python
"""
DeepEval uses a pytest-like interface with assertions.
pip install deepeval
"""
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    HallucinationMetric,
)


def test_rag_response():
    """Run as: deepeval test run test_rag.py"""

    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris, a major European city.",
        expected_output="Paris is the capital of France.",
        retrieval_context=[
            "Paris is the capital and most populous city of France.",
            "France is a country in Western Europe.",
        ],
    )

    # Define metrics with thresholds
    faithfulness = FaithfulnessMetric(threshold=0.7)
    relevancy = AnswerRelevancyMetric(threshold=0.7)
    precision = ContextualPrecisionMetric(threshold=0.7)
    recall = ContextualRecallMetric(threshold=0.7)

    # Assert — fails test if below threshold
    assert_test(test_case, [faithfulness, relevancy, precision, recall])
```

### DeepEval in CI/CD

```python
"""
file: tests/test_rag_evaluation.py
Run with: deepeval test run tests/test_rag_evaluation.py
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.dataset import EvaluationDataset


# Load test cases from file (generated from your golden dataset)
def load_test_cases():
    import json
    with open("tests/golden_dataset.json") as f:
        data = json.load(f)

    return [
        LLMTestCase(
            input=item["question"],
            actual_output=run_rag_pipeline(item["question"]),  # your pipeline
            expected_output=item["ground_truth"],
            retrieval_context=item.get("contexts", []),
        )
        for item in data
    ]


def run_rag_pipeline(question: str) -> str:
    """Replace with your actual RAG pipeline."""
    # from my_rag import pipeline
    # return pipeline.query(question)["answer"]
    return ""


@pytest.mark.parametrize("test_case", load_test_cases())
def test_rag_quality(test_case):
    faithfulness = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini")
    relevancy = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini")
    assert_test(test_case, [faithfulness, relevancy])
```

```bash
# Run in CI pipeline
deepeval test run tests/test_rag_evaluation.py --verbose

# Generate report
deepeval test run tests/test_rag_evaluation.py --report
```

### RAGAS vs DeepEval

| Feature              | RAGAS                       | DeepEval                    |
| -------------------- | --------------------------- | --------------------------- |
| API style            | Batch evaluate              | pytest assertions           |
| CI/CD integration    | Manual (wrap in script)     | Native (pytest runner)      |
| Metrics              | RAG-specific (7+)           | RAG + general LLM (15+)     |
| Test data generation | Built-in                    | Built-in                    |
| Dashboard            | No (use pandas)             | Yes (Confident AI platform) |
| Open source          | Yes                         | Yes (core), paid dashboard  |
| Best for             | Quick evaluation, exploring | Regression testing, CI/CD   |

---

## 3. LangSmith (Tracing + Evaluation)

> **Best if you're in the LangChain ecosystem.** Traces every step of your pipeline, lets you annotate and evaluate.

### Setup

```bash
pip install langsmith langchain-openai

# Set environment variables
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-api-key"
export LANGCHAIN_PROJECT="my-rag-project"
```

### Automatic Tracing (Zero Code Change)

```python
"""
With env vars set, all LangChain calls are automatically traced.
No code changes needed — just import and use LangChain normally.
"""
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-key"
os.environ["LANGCHAIN_PROJECT"] = "rag-evaluation"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# This call is automatically traced in LangSmith dashboard
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is RAG?")
# Go to smith.langchain.com to see the trace
```

### Evaluation with LangSmith Datasets

```python
"""
Create evaluation datasets and run evaluations in LangSmith.
"""
from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Create a dataset
dataset = client.create_dataset("rag-eval-v1", description="RAG evaluation questions")

# Add examples
examples = [
    {
        "input": {"question": "What is the capital of France?"},
        "output": {"answer": "Paris is the capital of France."},
    },
    {
        "input": {"question": "How does photosynthesis work?"},
        "output": {"answer": "Plants convert sunlight, water, and CO2 into glucose and oxygen."},
    },
]
for example in examples:
    client.create_example(
        inputs=example["input"],
        outputs=example["output"],
        dataset_id=dataset.id,
    )


# Define your RAG pipeline as a function
def my_rag_pipeline(inputs: dict) -> dict:
    question = inputs["question"]
    # ... your RAG pipeline here ...
    answer = "Paris is the capital of France."  # placeholder
    return {"answer": answer}


# Define evaluator
def correct_answer(run, example) -> dict:
    """Check if the answer is correct."""
    predicted = run.outputs["answer"]
    expected = example.outputs["answer"]
    # Simple string comparison (use LLM for semantic comparison)
    score = 1.0 if expected.lower() in predicted.lower() else 0.0
    return {"key": "correctness", "score": score}


# Run evaluation
results = evaluate(
    my_rag_pipeline,
    data="rag-eval-v1",
    evaluators=[correct_answer],
    experiment_prefix="rag-v1",
)
```

### When to Use LangSmith

- You're using LangChain and want zero-config tracing
- You need a visual debugger for complex chains
- Your team needs a shared evaluation dashboard
- You want to collect human feedback on production responses
- Cost: Free tier (5K traces/month), paid for more

---

## 4. LangFuse (Open-Source Alternative)

> **Best open-source tracing platform.** Self-hostable, privacy-friendly.

### Setup

```bash
pip install langfuse

# Set environment variables
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL
```

### Tracing RAG Pipeline

```python
"""
Trace your RAG pipeline with LangFuse.
Works with any framework (not just LangChain).
"""
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()


@observe()
def rag_pipeline(question: str) -> dict:
    """Full RAG pipeline with automatic tracing."""

    # Step 1: Retrieve
    chunks = retrieve(question)

    # Step 2: Rerank
    reranked = rerank(question, chunks)

    # Step 3: Generate
    answer = generate(question, reranked)

    # Add metadata to current trace
    langfuse_context.update_current_observation(
        metadata={"num_chunks": len(chunks), "num_reranked": len(reranked)},
    )

    return {"answer": answer, "chunks": reranked}


@observe()
def retrieve(question: str) -> list:
    """Retrieval step — automatically creates a span."""
    # ... your retrieval code ...
    return ["chunk1", "chunk2", "chunk3"]


@observe()
def rerank(question: str, chunks: list) -> list:
    """Reranking step — automatically creates a span."""
    # ... your reranking code ...
    return chunks[:2]


@observe()
def generate(question: str, context: list) -> str:
    """Generation step — automatically creates a span."""
    # ... your LLM call ...
    return "Generated answer"


# Use it
result = rag_pipeline("What is RAG?")
langfuse.flush()  # ensure traces are sent
```

### LangFuse with LangChain (CallbackHandler)

```python
"""
If you use LangChain, LangFuse integrates via callback handler.
"""
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    secret_key="sk-lf-...",
    public_key="pk-lf-...",
    host="https://cloud.langfuse.com",
)

# Add to any LangChain call
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke(
    "What is RAG?",
    config={"callbacks": [langfuse_handler]},
)
```

### Scoring in LangFuse

```python
"""
Add evaluation scores to traces in LangFuse.
"""
from langfuse import Langfuse

langfuse = Langfuse()

# Score a specific trace
langfuse.score(
    trace_id="trace-id-from-pipeline",
    name="faithfulness",
    value=0.95,
    comment="All claims supported by context",
)

# Or score within @observe decorator
@observe()
def scored_rag(question: str):
    answer = "..."
    # Auto-score within the trace
    langfuse_context.score_current_trace(
        name="user_feedback",
        value=1,  # thumbs up
    )
    return answer
```

---

## 5. Arize Phoenix (Observability + Evaluation)

> **Best for embedding analysis and drift detection.** Visualize your embedding space.

### Setup

```bash
pip install arize-phoenix opentelemetry-exporter-otlp
```

### Launch Phoenix and Trace

```python
"""
Phoenix runs locally and provides a UI for trace inspection.
"""
import phoenix as px

# Launch Phoenix (opens at http://localhost:6006)
session = px.launch_app()

# Instrument your app
from phoenix.otel import register
tracer_provider = register(project_name="my-rag")

# Now instrument LangChain, LlamaIndex, or OpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

# All LangChain calls are now traced in Phoenix UI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("What is RAG?")
# Check http://localhost:6006 to see the trace
```

### Embedding Analysis (Phoenix's Strength)

```python
"""
Phoenix excels at visualizing and analyzing embeddings.
Use this to debug retrieval quality.
"""
import phoenix as px
import pandas as pd
import numpy as np

# Create dataset of queries and their embeddings
query_df = pd.DataFrame({
    "text": ["What is RAG?", "How does indexing work?", ...],
    "embedding": [np.array([0.1, 0.2, ...]), ...],  # your embeddings
    "category": ["concept", "implementation", ...],
})

# Create dataset of documents and their embeddings
doc_df = pd.DataFrame({
    "text": ["RAG is a technique...", "Indexing involves...", ...],
    "embedding": [np.array([0.1, 0.2, ...]), ...],
})

# Launch with embedding analysis
session = px.launch_app(
    primary=px.Dataset(query_df, "queries"),
    reference=px.Dataset(doc_df, "documents"),
)
# Phoenix UI shows: UMAP clusters, drift detection, nearest neighbor analysis
```

---

## Combining Frameworks (Recommended)

```python
"""
Best practice: use RAGAS for metrics + LangFuse for tracing.
This gives you both quantitative evaluation AND production observability.
"""
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langfuse.decorators import observe, langfuse_context
from langfuse import Langfuse

langfuse = Langfuse()


@observe(name="rag-pipeline")
def traced_rag_pipeline(question: str) -> dict:
    """RAG pipeline traced with LangFuse."""
    # ... your pipeline ...
    return {
        "answer": "...",
        "contexts": ["chunk1", "chunk2"],
    }


def run_evaluation(eval_set: list[dict]):
    """
    Run RAGAS evaluation and push scores to LangFuse.
    """
    questions, answers, contexts, truths = [], [], [], []

    for item in eval_set:
        result = traced_rag_pipeline(item["question"])
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

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    # Push RAGAS scores to LangFuse
    df = results.to_pandas()
    for _, row in df.iterrows():
        langfuse.score(
            name="ragas_faithfulness",
            value=row["faithfulness"],
            trace_id=row.get("trace_id"),
        )

    langfuse.flush()
    return results
```

---

## Summary: Which Framework When?

```
START HERE
    │
    ▼
"Do I need tracing/monitoring?"──── NO ──── Use RAGAS alone
    │
   YES
    │
    ▼
"Am I on LangChain?" ─── YES ─── LangSmith (easiest integration)
    │
   NO / want open-source
    │
    ▼
"Need self-hosting?"──── YES ──── LangFuse (self-host, OSS)
    │
   NO
    │
    ▼
"Need embedding analysis?" ─── YES ─── Arize Phoenix
    │
   NO
    │
    ▼
LangFuse Cloud (free tier, good default)

For CI/CD regression tests: Add DeepEval regardless of choice above
```

---

## 📚 Additional Reading

- [RAGAS docs](https://docs.ragas.io/) — comprehensive guide with all metrics
- [DeepEval docs](https://docs.confident-ai.com/) — CI/CD focused evaluation
- [LangSmith docs](https://docs.smith.langchain.com/) — LangChain's tracing platform
- [LangFuse docs](https://langfuse.com/docs) — open-source tracing
- [Arize Phoenix](https://docs.arize.com/phoenix) — observability platform

---

## Syllabus Mapping

Maps to `p2_rag_depth.md` §2.7 (Evaluation) and `01_ai_engineering_checklist.md` (observability + evaluation).
