# 🎨 Prompt Engineering Patterns

## 📚 Overview

Prompt engineering is not "writing good prompts" — it's systematic design of input transformations that reliably produce desired outputs. This note goes beyond basic zero-shot/few-shot/CoT (covered in [LLM Fundamentals](./00_llm_fundamentals.md)) into advanced patterns: meta-prompting, automatic prompt optimization (DSPy), prompt chaining architectures, few-shot selection strategies, prompt versioning, and evaluation-driven prompt development.

> 📌 _Amateur prompt engineering: tweaking words. Professional prompt engineering: version-controlled templates with automated evaluation on regression sets._

---

## 🎯 Learning Objectives

- **Design** advanced prompt patterns (meta-prompting, self-consistency, tree-of-thought)
- **Implement** prompt chaining for complex multi-step tasks
- **Optimize** prompts programmatically (DSPy, automatic few-shot selection)
- **Version** and manage prompts as first-class software artifacts
- **Evaluate** prompt changes against regression test sets
- **Select** optimal few-shot examples dynamically (embedding-based retrieval)

---

## 🧠 Sections (To Be Written)

### 1. Advanced Prompting Patterns

- Self-consistency (multiple CoT paths → majority vote)
- Tree-of-thought (branching reasoning)
- Meta-prompting (prompt that writes prompts)
- Reflexion (self-critique and retry)
- Role-based prompting (persona engineering)
- Directional stimulus prompting

### 2. Prompt Chaining

- Sequential chains (output → input)
- Parallel chains (fan-out → aggregate)
- Conditional chains (branching based on output)
- Error recovery in chains (retry vs fallback)
- Token budget allocation across chain steps

### 3. Programmatic Prompt Optimization

- DSPy: signatures, modules, and optimizers
- Automatic few-shot selection (embedding similarity)
- Bayesian prompt optimization
- Gradient-free optimization of prompt templates
- When optimization beats manual tuning

### 4. Few-Shot Selection Strategies

- Random vs curated vs dynamic selection
- Embedding-based retrieval of similar examples
- Diversity-aware selection (cover edge cases)
- Label-balanced selection
- Maximum marginal relevance for few-shot

### 5. Prompt Versioning & Management

- Prompts as code (version control, PR reviews)
- Prompt registries (database-backed templates)
- A/B testing prompts in production
- Prompt diff tools
- Rollback strategies

### 6. Evaluation-Driven Development

- Regression test sets for prompts
- Automated scoring (exact match, LLM-as-judge, rubric)
- Statistical significance in prompt comparisons
- Cost-aware evaluation (better prompt vs cheaper model)
- Continuous evaluation in production

### 7. Common Pitfalls

| Symptom                                   | Cause                             | Fix                        |
| ----------------------------------------- | --------------------------------- | -------------------------- |
| Prompt works in playground, fails in prod | Different context/history in prod | Test with realistic inputs |
| Small change breaks unrelated outputs     | Prompt is fragile/entangled       | Modular prompt design      |
| Can't reproduce results                   | No version control                | Version prompts as code    |
| "Best prompt" varies by model             | Model-specific optimization       | Per-model prompt variants  |

---

## 📖 Resources

- DSPy documentation and tutorials
- OpenAI Prompt Engineering guide
- Anthropic Prompt Engineering guide
- promptfoo: prompt evaluation framework

---

## ➡️ Next Steps

Continue to **Phase 2: RAG** → [RAG Overview](../phase_2_rag/00_rag_overview.md)
