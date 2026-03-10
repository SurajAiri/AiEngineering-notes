# RAG Debugging Checklist — "My RAG Gives Bad Answers"

## 🟢 How to Approach This Topic

> **Why this matters for your job:** RAG systems break silently. The LLM will confidently give wrong answers, and you need a systematic way to find out WHERE in the pipeline things went wrong. This checklist is your go-to diagnostic tool.

**When to use this:** Anytime your RAG system produces a bad answer — wrong, incomplete, hallucinated, or irrelevant. Walk through this checklist top-to-bottom.

**⏱️ Time to diagnose: 15-60 min depending on the issue**

---

## The Diagnostic Pipeline

Bad RAG answers come from one (or more) of these stages failing. Debug in this order — from data to output:

```
                  ┌────────────────────────────────────────┐
                  │        BAD ANSWER OBSERVED             │
                  └───────────────┬────────────────────────┘
                                  │
                    Debug in this order:
                                  │
               ┌──────────────────▼──────────────────────┐
          1.   │  DATA QUALITY                            │
               │  Is the answer even in the source data?  │
               └──────────────────┬──────────────────────┘
                                  │
               ┌──────────────────▼──────────────────────┐
          2.   │  CHUNKING                                │
               │  Is the answer preserved in a chunk?     │
               └──────────────────┬──────────────────────┘
                                  │
               ┌──────────────────▼──────────────────────┐
          3.   │  RETRIEVAL                               │
               │  Is the right chunk retrieved?           │
               └──────────────────┬──────────────────────┘
                                  │
               ┌──────────────────▼──────────────────────┐
          4.   │  RE-RANKING                              │
               │  Is the right chunk ranked high enough?  │
               └──────────────────┬──────────────────────┘
                                  │
               ┌──────────────────▼──────────────────────┐
          5.   │  CONTEXT ASSEMBLY                        │
               │  Is the context well-formatted for LLM? │
               └──────────────────┬──────────────────────┘
                                  │
               ┌──────────────────▼──────────────────────┐
          6.   │  GENERATION                              │
               │  Is the LLM using the context correctly? │
               └──────────────────────────────────────────┘
```

---

## Step 1: Check Data Quality

**Question:** Does the source data actually contain the answer?

```python
# Quick check: search raw documents for the expected answer
import os

def search_raw_docs(directory: str, search_term: str):
    """Search source documents for a term. If not found, data problem."""
    found_in = []
    for root, _, files in os.walk(directory):
        for f in files:
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read()
                    if search_term.lower() in content.lower():
                        # Find surrounding context
                        idx = content.lower().index(search_term.lower())
                        snippet = content[max(0, idx-100):idx+200]
                        found_in.append({"file": path, "snippet": snippet})
            except Exception:
                pass
    return found_in

results = search_raw_docs("documents/", "timeout configuration")
if not results:
    print("❌ FOUND THE PROBLEM: Answer not in source data!")
    print("   Fix: Add the missing document or update existing ones.")
else:
    print(f"✅ Found in {len(results)} documents. Move to Step 2.")
    for r in results:
        print(f"  📄 {r['file']}: ...{r['snippet']}...")
```

### Common Data Problems

| Symptom                        | Root Cause                    | Fix                                        |
| ------------------------------ | ----------------------------- | ------------------------------------------ |
| Answer not in any document     | Missing data                  | Add the document to your corpus            |
| Answer in document but garbled | Parser failure                | Try different parser (Docling, LlamaParse) |
| Answer in old version only     | Stale data, no versioning     | Implement document version tracking        |
| Answer in image/table          | Parser can't extract non-text | Use OCR-capable parser                     |

---

## Step 2: Check Chunking

**Question:** Is the answer preserved within a single chunk (or retrievable combination of chunks)?

```python
def check_chunking(chunks: list[str], expected_content: str) -> dict:
    """Check if expected content exists in any chunk."""
    expected_lower = expected_content.lower()

    exact_matches = []
    partial_matches = []

    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        if expected_lower in chunk_lower:
            exact_matches.append(i)
        elif any(word in chunk_lower for word in expected_lower.split()[:5]):
            partial_matches.append(i)

    if exact_matches:
        print(f"✅ Full answer found in chunk(s): {exact_matches}")
        for idx in exact_matches:
            print(f"  Chunk {idx} ({len(chunks[idx])} chars): {chunks[idx][:200]}...")
    elif partial_matches:
        print(f"⚠️ Partial match in chunk(s): {partial_matches}")
        print("  The answer might be SPLIT across chunk boundaries.")
        print("  Fix: Increase chunk_overlap or use hierarchical chunking.")
    else:
        print("❌ FOUND THE PROBLEM: Answer not in any chunk!")
        print("  Fix: Check if cleaning/parsing removed it. Re-chunk with larger size.")

    return {"exact": exact_matches, "partial": partial_matches}
```

### Common Chunking Problems

| Symptom                        | Root Cause                    | Fix                                     |
| ------------------------------ | ----------------------------- | --------------------------------------- |
| Answer split across 2 chunks   | Chunk boundary in wrong place | Increase overlap, use semantic chunking |
| Answer in tiny fragment        | Chunk too small, lost context | Increase chunk_size (try 512 tokens)    |
| Table/code split across chunks | Structure-unaware chunking    | Use structure-aware chunking (Docling)  |
| Answer buried in giant chunk   | Chunk too large               | Decrease chunk_size, add hierarchical   |

---

## Step 3: Check Retrieval

**Question:** Is the right chunk being retrieved? Where does it rank?

```python
def diagnose_retrieval(query: str, retriever, expected_chunk_text: str, k: int = 20):
    """Check if retrieval returns the expected chunk and at what rank."""
    results = retriever.invoke(query)

    found_rank = None
    for i, doc in enumerate(results[:k]):
        content = doc.page_content if hasattr(doc, "page_content") else doc.text
        if expected_chunk_text[:50].lower() in content.lower():
            found_rank = i + 1
            break

    if found_rank is None:
        print(f"❌ FOUND THE PROBLEM: Expected chunk NOT in top {k} results!")
        print("  Possible causes:")
        print("  - Vocabulary mismatch (query uses different words than document)")
        print("  - Embedding model not suitable for your domain")
        print("  - Need hybrid search (vector + BM25)")
        print("  - Query too vague or ambiguous")
    elif found_rank <= 5:
        print(f"✅ Found at rank {found_rank}. Retrieval is fine. Move to Step 4.")
    else:
        print(f"⚠️ Found at rank {found_rank} (outside typical top-5 cutoff)")
        print("  Fix: Add re-ranker, increase k, or try query rewriting.")

    # Show what WAS retrieved (for comparison)
    print(f"\nTop 5 actually retrieved:")
    for i, doc in enumerate(results[:5]):
        content = doc.page_content if hasattr(doc, "page_content") else doc.text
        score = getattr(doc, "score", "N/A")
        print(f"  {i+1}. [Score: {score}] {content[:150]}...")

    return found_rank


# Usage (LangChain):
# rank = diagnose_retrieval(
#     query="How does timeout handling work?",
#     retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
#     expected_chunk_text="Payment timeouts occur when the gateway",
# )
```

### Common Retrieval Problems

| Symptom                                    | Root Cause                       | Fix                                           |
| ------------------------------------------ | -------------------------------- | --------------------------------------------- |
| Right chunk at rank 15+                    | Embedding mismatch               | Add re-ranker, try hybrid search              |
| Right chunk not retrieved at all           | Vocabulary mismatch              | Add BM25, try query rewriting                 |
| All retrieved chunks are similar but wrong | Query too broad                  | Add metadata filtering, be more specific      |
| Low similarity scores across the board     | Embedding model mismatch         | Try different embedding model for your domain |
| BM25 finds it but vector doesn't           | Semantic gap                     | Use hybrid (ensemble) retrieval               |
| Vector finds it but BM25 doesn't           | Concept-level match, no keywords | Use hybrid retrieval (keep both)              |

---

## Step 4: Check Re-ranking

**Question:** Does the re-ranker change the order correctly, or does it push good results down?

```python
def diagnose_reranking(query: str, base_retriever, reranking_retriever,
                       expected_text: str):
    """Compare retrieval with and without re-ranking."""
    base_results = base_retriever.invoke(query)
    reranked_results = reranking_retriever.invoke(query)

    base_rank = None
    reranked_rank = None

    for i, doc in enumerate(base_results[:20]):
        if expected_text[:50].lower() in doc.page_content.lower():
            base_rank = i + 1

    for i, doc in enumerate(reranked_results[:20]):
        if expected_text[:50].lower() in doc.page_content.lower():
            reranked_rank = i + 1

    print(f"Base retrieval rank: {base_rank or 'Not found'}")
    print(f"After reranking:     {reranked_rank or 'Not found'}")

    if base_rank and reranked_rank:
        if reranked_rank < base_rank:
            print("✅ Reranker improved ranking!")
        elif reranked_rank > base_rank:
            print("⚠️ Reranker HURT ranking — consider adjusting or removing it")
        else:
            print("→ No change from reranking")
```

---

## Step 5: Check Context Assembly

**Question:** Is the context passed to the LLM well-formatted and complete?

```python
def diagnose_context(context_text: str, query: str):
    """Check the assembled context for issues."""
    issues = []

    # Check: is it too long?
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    token_count = len(enc.encode(context_text))
    if token_count > 6000:
        issues.append(f"⚠️ Context is {token_count} tokens — might be too long. LLM may miss info in the middle (lost-in-the-middle effect).")

    # Check: is it too short?
    if token_count < 50:
        issues.append("⚠️ Context is very short — might not contain enough info.")

    # Check: duplicate content?
    chunks = context_text.split("---")
    seen = set()
    for chunk in chunks:
        normalized = " ".join(chunk.split()[:20])
        if normalized in seen:
            issues.append("⚠️ Duplicate chunks detected in context.")
            break
        seen.add(normalized)

    # Check: does context contain the answer?
    # (You need to know what the expected answer should be)

    if not issues:
        print("✅ Context looks OK. Move to Step 6.")
    else:
        for issue in issues:
            print(issue)

    print(f"\nContext stats: {token_count} tokens, {len(chunks)} chunks")
    print(f"First 500 chars: {context_text[:500]}...")
```

### Common Context Problems

| Symptom                                | Root Cause                      | Fix                                        |
| -------------------------------------- | ------------------------------- | ------------------------------------------ |
| LLM ignores relevant info              | Lost-in-the-middle effect       | Put best chunks first, limit to 5-8 chunks |
| LLM hallucinates connections           | Too many loosely related chunks | Reduce k, increase relevance threshold     |
| Answer is in context but LLM misses it | Context formatting is poor      | Use clear separators, source labels        |
| Duplicate info in context              | No deduplication                | Deduplicate chunks before assembling       |

---

## Step 6: Check Generation

**Question:** Given good context, is the LLM producing the right answer?

```python
def diagnose_generation(query: str, context: str, answer: str, llm):
    """Test if the LLM can answer correctly given known-good context."""
    # Test with explicit instructions
    test_prompt = f"""Based ONLY on the context below, answer the question.
If the context doesn't contain the answer, say "NOT IN CONTEXT".

Context:
{context}

Question: {query}

Answer:"""

    test_response = llm.invoke(test_prompt)
    print(f"Test answer: {test_response.content}")

    if "NOT IN CONTEXT" in test_response.content:
        print("⚠️ LLM says answer not in context — go back to Steps 1-5")
    elif test_response.content.strip() != answer.strip():
        print("⚠️ LLM gives different answer with explicit prompt")
        print("  This might be a prompt engineering issue.")
        print("  Fix: Adjust system prompt, add examples, or try a different model.")
    else:
        print("✅ LLM generates correct answer when context is right.")
```

### Common Generation Problems

| Symptom                                 | Root Cause                          | Fix                                                       |
| --------------------------------------- | ----------------------------------- | --------------------------------------------------------- |
| LLM ignores context, uses own knowledge | Prompt doesn't enforce context-only | Add "Answer ONLY from the context. If not found, say so." |
| LLM copies context verbatim             | Prompt too restrictive              | Adjust to "Answer in your own words based on the context" |
| LLM hallucinates details not in context | Temperature too high or weak model  | Lower temperature to 0, try stronger model                |
| LLM gives partial answer                | Context incomplete or too long      | Improve retrieval or reduce context size                  |

---

## Quick Reference: One-Page Checklist

```
□ STEP 1: DATA — Is the answer in the source documents?
    □ Search raw docs for key terms
    □ Check parser output quality
    □ Verify data is up to date

□ STEP 2: CHUNKING — Is the answer in a retrievable chunk?
    □ Search chunks for expected content
    □ Check for chunk boundary splits
    □ Verify chunk size is appropriate

□ STEP 3: RETRIEVAL — Is the right chunk retrieved?
    □ Check rank of expected chunk (should be top 5)
    □ Compare vector vs BM25 vs hybrid
    □ Check similarity scores
    □ Try query rewriting

□ STEP 4: RE-RANKING — Does reranking help or hurt?
    □ Compare rank before/after reranking
    □ Check if reranker model is appropriate

□ STEP 5: CONTEXT — Is context well-assembled?
    □ Check token count (not too long, not too short)
    □ Check for duplicates
    □ Verify formatting and separators

□ STEP 6: GENERATION — Is the LLM using context correctly?
    □ Test with explicit context-only prompt
    □ Check temperature and model settings
    □ Verify prompt template
```

---

## Automated Diagnostic Script

```python
"""
Run all diagnostic steps in sequence.
Paste this into a notebook for quick debugging.
"""


def full_rag_diagnosis(
    query: str,
    expected_answer_keywords: list[str],
    raw_docs_dir: str,
    chunks: list[str],
    retriever,
    context_builder,
    llm,
):
    """Run complete RAG diagnostic pipeline."""
    print("=" * 60)
    print(f"DIAGNOSING: {query}")
    print("=" * 60)

    # Step 1: Data
    print("\n📋 STEP 1: Data Quality")
    for keyword in expected_answer_keywords:
        results = search_raw_docs(raw_docs_dir, keyword)
        status = "✅" if results else "❌"
        print(f"  {status} '{keyword}' found in {len(results)} docs")

    # Step 2: Chunking
    print("\n📋 STEP 2: Chunking")
    for keyword in expected_answer_keywords:
        result = check_chunking(chunks, keyword)

    # Step 3: Retrieval
    print("\n📋 STEP 3: Retrieval")
    retrieved = retriever.invoke(query)
    print(f"  Retrieved {len(retrieved)} chunks")
    for i, doc in enumerate(retrieved[:5]):
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        print(f"  {i+1}. {content[:150]}...")

    # Step 4-5: Context
    print("\n📋 STEP 4-5: Context Assembly")
    context = context_builder(retrieved)
    diagnose_context(context, query)

    # Step 6: Generation
    print("\n📋 STEP 6: Generation")
    diagnose_generation(query, context, "", llm)

    print("\n" + "=" * 60)
    print("DIAGNOSIS COMPLETE")
    print("=" * 60)
```

---

## 📌 Key Insight

> Most RAG failures are retrieval failures, not generation failures. If you put the right context in front of a modern LLM, it will almost always generate the right answer. Debug retrieval first.

The most common root causes in practice:

1. **Data not in corpus** (40%) — check data first!
2. **Chunk boundary split** (20%) — answer spans two chunks
3. **Vocabulary mismatch** (20%) — query uses different words
4. **Too many noisy chunks** (15%) — relevant chunk drowned by irrelevant ones
5. **LLM ignoring context** (5%) — usually a prompt issue

---

## Syllabus Mapping

Maps to **§2.5 Failure Modes** in `p2_rag_depth.md`. This checklist is the practical companion to the failure modes theory — when something goes wrong, use this to find and fix it.
