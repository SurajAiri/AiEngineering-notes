# Prompt Design for RAG Systems

> ⚠️ **Good to Read.** This is not about general prompt engineering — it's specifically about how to construct the prompt that feeds retrieved context to the LLM.

## Why This Matters

You retrieved great chunks. Now what? How you present those chunks to the LLM in the prompt determines whether you get a grounded, cited answer or a hallucinated mess. RAG prompt design is a production concern, not a creative writing exercise.

---

## The RAG Prompt Structure

```
┌──────────────────────────────────────────────────────┐
│                    RAG PROMPT                         │
│                                                      │
│  1. SYSTEM INSTRUCTION                               │
│     "Answer based ONLY on the provided context.      │
│      If the context doesn't contain the answer,      │
│      say 'I don't know.' Cite sources with [1]."     │
│                                                      │
│  2. RETRIEVED CONTEXT                                │
│     [1] chunk_text_1                                 │
│     [2] chunk_text_2                                 │
│     [3] chunk_text_3                                 │
│                                                      │
│  3. USER QUERY                                       │
│     "What is the refund policy for enterprise?"      │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Key Principles

### 1. Ground the LLM to Context Only

```python
# BAD: No grounding instruction
prompt = f"Answer this: {query}\n\nContext: {context}"
# LLM may use its training data instead of your context

# GOOD: Explicit grounding
prompt = f"""Answer the question based ONLY on the provided context.
If the context does not contain enough information to answer,
say "I don't have enough information to answer this question."
Do not use any knowledge outside of the provided context.

Context:
{context}

Question: {query}

Answer:"""
```

### 2. Number Your Sources for Citations

```python
# Format context with numbered sources
def format_context_for_prompt(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        parts.append(f"[{i}] (Source: {source})\n{chunk['text']}")
    return "\n\n".join(parts)
```

### 3. Handle "I Don't Know" Properly

```python
system_prompt = """You are a helpful assistant that answers questions using
ONLY the provided context.

Rules:
1. If the context contains the answer, provide it with citations [1], [2], etc.
2. If the context partially answers the question, provide what you can and
   note what's missing.
3. If the context does NOT contain the answer, respond with:
   "I don't have information about that in the available documents."
4. NEVER make up information that isn't in the context.
"""
```

### 4. Conversation History in RAG

```python
# For multi-turn RAG, include relevant conversation history
# but keep it concise to save token budget

def build_rag_messages(query, context, chat_history=None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add recent chat history (last 2-3 turns only)
    if chat_history:
        for turn in chat_history[-3:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

    # Current query with context
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    return messages
```

---

## Production Prompt Template

```python
"""
Production-ready RAG prompt template.
"""

RAG_SYSTEM_PROMPT = """You are a precise assistant that answers questions
using ONLY the provided context documents.

RULES:
1. Base your answer EXCLUSIVELY on the provided context.
2. Cite sources using [1], [2], etc. matching the source numbers.
3. If the context doesn't contain the answer, say:
   "I don't have information about that in the provided documents."
4. If the context partially answers the question, provide what you can
   and clearly state what information is missing.
5. Do not speculate or use external knowledge.
6. Keep answers concise and directly address the question.
"""

def build_rag_prompt(query: str, chunks: list[dict], max_context_tokens: int = 3000) -> str:
    """Build the user message with formatted context."""
    context_parts = []
    total_tokens = 0

    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        section = chunk.get("metadata", {}).get("section", "")
        source_label = f"{source}"
        if section:
            source_label += f" > {section}"

        entry = f"[{i}] (Source: {source_label})\n{chunk['text']}"
        # Rough token estimate (1 token ≈ 4 chars)
        entry_tokens = len(entry) // 4
        if total_tokens + entry_tokens > max_context_tokens:
            break
        context_parts.append(entry)
        total_tokens += entry_tokens

    context = "\n\n---\n\n".join(context_parts)

    return f"""Context Documents:

{context}

---

Question: {query}

Answer (cite sources with [1], [2], etc.):"""
```

---

## LangChain / LlamaIndex Integration

```python
# LangChain prompt template for RAG
from langchain.prompts import ChatPromptTemplate

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
])

# LlamaIndex custom prompt
from llama_index.core import PromptTemplate

qa_prompt = PromptTemplate(
    "Context:\n{context_str}\n\n"
    "Answer the following question based only on the context above.\n"
    "Question: {query_str}\n"
    "Answer: "
)
query_engine = index.as_query_engine(text_qa_template=qa_prompt)
```

---

## Common Questions

### Q: Should the context go before or after the question?

**A:** Context before the question generally works better. The LLM reads the context first, then encounters the question and can attend to relevant parts of the context. Some models work well either way — test with your specific model.

### Q: How do I prevent the LLM from ignoring the context and using its own knowledge?

**A:** Use strong grounding instructions ("ONLY use the provided context"), set temperature to 0, and include refusal instructions ("If the context doesn't contain the answer, say 'I don't know'"). No method is 100% reliable — always verify with faithfulness evaluation.

### Q: What temperature should I use for RAG?

**A:** **0 or very low (0.1)**. RAG is about factual answers from retrieved content, not creativity. Higher temperatures increase the chance of hallucination.
