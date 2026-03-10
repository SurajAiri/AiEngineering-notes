# Chunk-Query Alignment & Structure-Aware Chunking

## Part 1: Chunk-Query Alignment Failures

### Why This Matters

Even with perfect embeddings and retrieval, if your **chunks don't align with the types of questions users ask**, retrieval will fail. This is one of the most common yet hardest-to-diagnose RAG failure modes.

---

### The Alignment Problem

```
Query: "What are the side effects of metformin?"

Chunk A (bad alignment — info split across chunks):
┌─────────────────────────────────────────┐
│ "...metformin is a first-line treatment │
│  for type 2 diabetes. It works by       │
│  reducing hepatic glucose production.   │  ← Chunk boundary
│  Common side effects include nausea,    │
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│  diarrhea, and abdominal pain. Rare     │
│  but serious: lactic acidosis. The      │
│  risk increases with renal impairment." │
└─────────────────────────────────────────┘

Neither chunk fully answers the question!

Chunk B (good alignment):
┌─────────────────────────────────────────┐
│ "Side effects of metformin:             │
│  Common: nausea, diarrhea, abdominal    │
│  pain, metallic taste.                  │
│  Rare but serious: lactic acidosis.     │
│  Risk increases with renal impairment." │
└─────────────────────────────────────────┘

Complete answer in one chunk ✅
```

---

### Common Misalignment Patterns

```python
"""
Demonstrate common chunk-query alignment failure patterns.
"""

# Pattern 1: ANSWER SPLIT ACROSS CHUNKS
# The answer spans a chunk boundary
problem_1 = {
    "query": "What are the input and output of the encoder?",
    "chunk_1": "The encoder takes a sequence of tokens as input and produces",
    "chunk_2": "a sequence of hidden states of dimension d_model=512.",
    "issue": "Input described in chunk 1, output in chunk 2",
}

# Pattern 2: QUESTION LEVEL ≠ CHUNK LEVEL
# Question asks for a summary but chunks are too detailed
problem_2 = {
    "query": "Give me an overview of the authentication system",
    "chunks": [
        "JWT tokens use RS256 signing algorithm...",         # too specific
        "The refresh token rotation period is 7 days...",    # too specific
        "OAuth2 PKCE flow starts with a code verifier...",   # too specific
    ],
    "issue": "User wants overview, chunks are implementation details",
}

# Pattern 3: MULTI-HOP ANSWER
# Answer requires combining info from multiple unconnected chunks
problem_3 = {
    "query": "Is the system's latency within SLA?",
    "chunk_a": "The p95 latency is 230ms.",           # from metrics page
    "chunk_b": "The SLA requires p95 < 200ms.",       # from SLA document
    "issue": "Need both chunks to answer; neither alone works",
}

# Pattern 4: VOCABULARY MISMATCH
# Query uses different words than the document
problem_4 = {
    "query": "How to fix memory issues?",
    "chunk": "OOM errors can be resolved by increasing the pod resource limits.",
    "issue": "'memory issues' vs 'OOM errors' — same concept, different words",
}
```

---

### Diagnosing Alignment Issues

```python
"""
Tool to diagnose chunk-query alignment problems.
Run this on your retrieval results to understand failure modes.

Requirements: pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer


class AlignmentDiagnostic:
    """Diagnose why retrieval is failing for specific queries."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def diagnose(
        self,
        query: str,
        retrieved_chunks: list[str],
        expected_answer: str,
        all_chunks: list[str],
    ) -> dict:
        """
        Analyze why retrieval may have failed.

        Returns diagnosis with:
        - Whether the answer exists in any chunk
        - Whether the answer is split across chunks
        - Similarity scores
        - Suggested fix
        """
        query_emb = self.model.encode(query, normalize_embeddings=True)
        answer_emb = self.model.encode(expected_answer, normalize_embeddings=True)

        # Check if answer appears in any single chunk
        answer_in_single_chunk = any(
            expected_answer.lower() in chunk.lower()
            for chunk in all_chunks
        )

        # Check if answer appears in retrieved chunks
        answer_in_retrieved = any(
            expected_answer.lower() in chunk.lower()
            for chunk in retrieved_chunks
        )

        # Score all chunks against the answer
        chunk_embs = self.model.encode(all_chunks, normalize_embeddings=True)
        answer_sims = np.dot(chunk_embs, answer_emb)
        query_sims = np.dot(chunk_embs, query_emb)

        # Find best chunk for the answer vs best for the query
        best_answer_idx = int(np.argmax(answer_sims))
        best_query_idx = int(np.argmax(query_sims))

        diagnosis = {
            "answer_in_single_chunk": answer_in_single_chunk,
            "answer_in_retrieved": answer_in_retrieved,
            "query_best_chunk": best_query_idx,
            "answer_best_chunk": best_answer_idx,
            "alignment_match": best_query_idx == best_answer_idx,
            "query_top_score": float(query_sims[best_query_idx]),
            "answer_top_score": float(answer_sims[best_answer_idx]),
        }

        # Determine failure mode
        if not answer_in_single_chunk:
            diagnosis["failure_mode"] = "ANSWER_SPLIT"
            diagnosis["fix"] = "Increase chunk size or change chunking strategy"
        elif not answer_in_retrieved:
            if best_query_idx != best_answer_idx:
                diagnosis["failure_mode"] = "VOCABULARY_MISMATCH"
                diagnosis["fix"] = "Add query rewriting or use hybrid search"
            else:
                diagnosis["failure_mode"] = "LOW_K"
                diagnosis["fix"] = "Increase k or improve ranking"
        elif diagnosis["alignment_match"]:
            diagnosis["failure_mode"] = "NONE"
            diagnosis["fix"] = "Retrieval is working correctly"
        else:
            diagnosis["failure_mode"] = "GRANULARITY_MISMATCH"
            diagnosis["fix"] = "Chunk size doesn't match query granularity"

        return diagnosis
```

---

## Part 2: Structure-Aware Chunking

### Why It Matters

Many documents contain structured elements — **headings, tables, code blocks, lists** — that carry meaning through their structure. Breaking these elements with a chunk boundary destroys their meaning.

```
BAD: Table split across chunks

Chunk 1:                          Chunk 2:
| Model | Params |                | MMLU | Speed |
|-------|--------|                |------|-------|
| GPT-4 | 1.7T   |               | 86.4 | Slow  |

← You can't associate GPT-4 with 86.4 anymore!
```

---

### What to Protect

```
┌─────────────────────────────────────────────────────┐
│           STRUCTURAL ELEMENTS TO PRESERVE            │
│                                                      │
│  📊 TABLES        Keep entire tables in one chunk    │
│  💻 CODE BLOCKS   Never split mid-code               │
│  📋 LISTS         Keep related list items together    │
│  📑 SECTIONS      Respect heading boundaries         │
│  🔢 EQUATIONS     Keep LaTeX/math expressions whole  │
│  📝 BLOCKQUOTES   Preserve quoted passages           │
└─────────────────────────────────────────────────────┘
```

---

### Simple Code — Protecting Structural Elements

````python
"""
Simple structure-aware chunker that keeps
tables, code blocks, and lists intact.
"""

import re


def structure_aware_chunk(
    text: str,
    max_chunk_chars: int = 1000
) -> list[str]:
    """
    Split text while keeping structural elements intact.
    Tables, code blocks, and lists are never split.
    """
    # Identify structural blocks
    blocks = split_into_blocks(text)

    chunks = []
    current_chunk = []
    current_size = 0

    for block in blocks:
        block_size = len(block)

        # If this single block exceeds max size, it becomes its own chunk
        if block_size > max_chunk_chars:
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_size = 0
            chunks.append(block)
            continue

        # If adding this block would exceed max, start new chunk
        if current_size + block_size > max_chunk_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [block]
            current_size = block_size
        else:
            current_chunk.append(block)
            current_size += block_size

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def split_into_blocks(text: str) -> list[str]:
    """
    Split text into structural blocks:
    - Code blocks (```)
    - Tables (| ... |)
    - Lists (consecutive - or * items)
    - Paragraphs (text separated by blank lines)
    """
    blocks = []
    current_block = []
    in_code_block = False
    in_table = False

    for line in text.split('\n'):
        # Code block detection
        if line.strip().startswith('```'):
            if in_code_block:
                current_block.append(line)
                blocks.append('\n'.join(current_block))
                current_block = []
                in_code_block = False
                continue
            else:
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                in_code_block = True
                current_block.append(line)
                continue

        if in_code_block:
            current_block.append(line)
            continue

        # Table detection
        if re.match(r'^\s*\|', line):
            if not in_table and current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
            in_table = True
            current_block.append(line)
            continue
        elif in_table:
            blocks.append('\n'.join(current_block))
            current_block = []
            in_table = False

        # Blank line = paragraph break
        if line.strip() == '':
            if current_block:
                blocks.append('\n'.join(current_block))
                current_block = []
        else:
            current_block.append(line)

    if current_block:
        blocks.append('\n'.join(current_block))

    return [b.strip() for b in blocks if b.strip()]


# Example
document = """# API Reference

The `/users` endpoint returns a list of users.

| Method | Path      | Description        |
|--------|-----------|-------------------|
| GET    | /users    | List all users    |
| POST   | /users    | Create a user     |
| DELETE | /users/:id| Delete a user     |

## Code Example

```python
import requests

response = requests.get("https://api.example.com/users")
users = response.json()
for user in users:
    print(user["name"])
````

## Response Format

The response includes:

- `id`: unique identifier
- `name`: user's display name
- `email`: user's email address
- `created_at`: account creation timestamp
  """

chunks = structure_aware_chunk(document, max_chunk_chars=300)
for i, chunk in enumerate(chunks):
print(f"=== Chunk {i} ({len(chunk)} chars) ===")
print(chunk)
print()

````

---

### Production Code — Full Structure-Aware Chunker

```python
"""
Production structure-aware chunker.
Detects and preserves tables, code blocks, lists, and heading boundaries.

Requirements: pip install tiktoken
"""

import re
import logging
import tiktoken
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class BlockType(Enum):
    PARAGRAPH = "paragraph"
    CODE = "code"
    TABLE = "table"
    LIST = "list"
    HEADING = "heading"
    BLOCKQUOTE = "blockquote"


@dataclass
class ContentBlock:
    """A structural block within a document."""
    type: BlockType
    content: str
    token_count: int
    heading_level: int = 0     # for heading blocks
    language: str = ""         # for code blocks
    is_splittable: bool = True # paragraphs can be split; tables/code cannot


@dataclass
class StructuredChunk:
    text: str
    index: int
    token_count: int
    block_types: list[str]            # types of blocks in this chunk
    contains_table: bool = False
    contains_code: bool = False
    heading_context: str = ""         # parent heading for context


class StructureAwareChunker:
    """
    Chunks documents while preserving structural elements.

    Rules:
    1. Never split a table across chunks
    2. Never split a code block across chunks
    3. Keep list items together when possible
    4. Respect heading boundaries as natural split points
    5. Paragraphs can be split if needed (with sentence awareness)
    """

    def __init__(
        self,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 50,
        encoding: str = "cl100k_base",
    ):
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.tokenizer = tiktoken.get_encoding(encoding)

    def chunk(self, text: str) -> list[StructuredChunk]:
        """Parse document and create structure-aware chunks."""
        blocks = self._parse_blocks(text)
        chunks = self._assemble_chunks(blocks)

        logger.info(
            f"Structure-aware chunking: {len(blocks)} blocks → {len(chunks)} chunks"
        )
        return chunks

    def _parse_blocks(self, text: str) -> list[ContentBlock]:
        """Parse document into typed structural blocks."""
        blocks = []
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]

            # Code block
            if line.strip().startswith('```'):
                lang_match = re.match(r'^```(\w*)', line.strip())
                lang = lang_match.group(1) if lang_match else ""
                code_lines = [line]
                i += 1
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1
                if i < len(lines):
                    code_lines.append(lines[i])
                    i += 1
                content = '\n'.join(code_lines)
                blocks.append(ContentBlock(
                    type=BlockType.CODE,
                    content=content,
                    token_count=len(self.tokenizer.encode(content)),
                    language=lang,
                    is_splittable=False,
                ))
                continue

            # Table
            if re.match(r'^\s*\|', line):
                table_lines = []
                while i < len(lines) and re.match(r'^\s*\|', lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                content = '\n'.join(table_lines)
                blocks.append(ContentBlock(
                    type=BlockType.TABLE,
                    content=content,
                    token_count=len(self.tokenizer.encode(content)),
                    is_splittable=False,
                ))
                continue

            # Heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                blocks.append(ContentBlock(
                    type=BlockType.HEADING,
                    content=line,
                    token_count=len(self.tokenizer.encode(line)),
                    heading_level=len(heading_match.group(1)),
                    is_splittable=False,
                ))
                i += 1
                continue

            # List
            if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+[.)]\s+', line):
                list_lines = [line]
                i += 1
                while i < len(lines) and (
                    re.match(r'^\s*[-*+]\s+', lines[i]) or
                    re.match(r'^\s*\d+[.)]\s+', lines[i]) or
                    (lines[i].startswith('  ') and lines[i].strip())
                ):
                    list_lines.append(lines[i])
                    i += 1
                content = '\n'.join(list_lines)
                blocks.append(ContentBlock(
                    type=BlockType.LIST,
                    content=content,
                    token_count=len(self.tokenizer.encode(content)),
                    is_splittable=len(list_lines) > 5,
                ))
                continue

            # Blockquote
            if line.strip().startswith('>'):
                quote_lines = [line]
                i += 1
                while i < len(lines) and lines[i].strip().startswith('>'):
                    quote_lines.append(lines[i])
                    i += 1
                content = '\n'.join(quote_lines)
                blocks.append(ContentBlock(
                    type=BlockType.BLOCKQUOTE,
                    content=content,
                    token_count=len(self.tokenizer.encode(content)),
                    is_splittable=False,
                ))
                continue

            # Empty line — skip
            if not line.strip():
                i += 1
                continue

            # Paragraph — collect consecutive non-empty, non-special lines
            para_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip() and not any([
                lines[i].strip().startswith('```'),
                re.match(r'^\s*\|', lines[i]),
                re.match(r'^#{1,6}\s+', lines[i]),
                re.match(r'^\s*[-*+]\s+', lines[i]),
                re.match(r'^\s*\d+[.)]\s+', lines[i]),
                lines[i].strip().startswith('>'),
            ]):
                para_lines.append(lines[i])
                i += 1
            content = '\n'.join(para_lines)
            blocks.append(ContentBlock(
                type=BlockType.PARAGRAPH,
                content=content,
                token_count=len(self.tokenizer.encode(content)),
                is_splittable=True,
            ))

        return blocks

    def _assemble_chunks(self, blocks: list[ContentBlock]) -> list[StructuredChunk]:
        """Assemble blocks into chunks respecting size limits."""
        chunks = []
        current_blocks = []
        current_tokens = 0
        current_heading = ""

        for block in blocks:
            # Track current heading
            if block.type == BlockType.HEADING:
                current_heading = block.content.lstrip('#').strip()

            # If this block alone exceeds max and isn't splittable,
            # make it its own chunk
            if block.token_count > self.max_chunk_tokens and not block.is_splittable:
                if current_blocks:
                    chunks.append(self._make_chunk(current_blocks, len(chunks), current_heading))
                    current_blocks = []
                    current_tokens = 0
                chunks.append(self._make_chunk([block], len(chunks), current_heading))
                continue

            # If adding this block exceeds max, finalize current chunk
            if current_tokens + block.token_count > self.max_chunk_tokens and current_blocks:
                # Headings should go with the next chunk, not the previous
                if block.type == BlockType.HEADING:
                    chunks.append(self._make_chunk(current_blocks, len(chunks), current_heading))
                    current_blocks = [block]
                    current_tokens = block.token_count
                else:
                    chunks.append(self._make_chunk(current_blocks, len(chunks), current_heading))
                    current_blocks = [block]
                    current_tokens = block.token_count
            else:
                current_blocks.append(block)
                current_tokens += block.token_count

        if current_blocks:
            chunks.append(self._make_chunk(current_blocks, len(chunks), current_heading))

        # Merge tiny trailing chunks
        if len(chunks) > 1 and chunks[-1].token_count < self.min_chunk_tokens:
            last = chunks.pop()
            chunks[-1].text += '\n\n' + last.text
            chunks[-1].token_count += last.token_count

        return chunks

    def _make_chunk(
        self, blocks: list[ContentBlock], index: int, heading: str
    ) -> StructuredChunk:
        text = '\n\n'.join(b.content for b in blocks)
        return StructuredChunk(
            text=text,
            index=index,
            token_count=sum(b.token_count for b in blocks),
            block_types=[b.type.value for b in blocks],
            contains_table=any(b.type == BlockType.TABLE for b in blocks),
            contains_code=any(b.type == BlockType.CODE for b in blocks),
            heading_context=heading,
        )


# ─── Usage ───
if __name__ == "__main__":
    chunker = StructureAwareChunker(max_chunk_tokens=150)

    document = """# Database Configuration

## Connection Settings

The database connection is configured via environment variables:

| Variable       | Default     | Description              |
|---------------|-------------|--------------------------|
| DB_HOST       | localhost   | Database hostname        |
| DB_PORT       | 5432        | Database port            |
| DB_NAME       | myapp       | Database name            |
| DB_POOL_SIZE  | 10          | Connection pool size     |

## Example Usage

Here's how to connect:

```python
import psycopg2

conn = psycopg2.connect(
    host=os.environ["DB_HOST"],
    port=os.environ["DB_PORT"],
    dbname=os.environ["DB_NAME"],
)
````

## Connection Pooling

Connection pooling reuses database connections to reduce overhead.
The pool size should be set based on your application's concurrency
needs. Too many connections can overwhelm the database, while too
few can cause request queuing.

Key considerations:

- Each connection uses ~10MB of database memory
- Set pool size = number of workers × 2
- Monitor connection wait time in production
  """

      chunks = chunker.chunk(document)
      for chunk in chunks:
          print(f"=== Chunk {chunk.index} ({chunk.token_count} tokens) ===")
          print(f"Types: {chunk.block_types}")
          print(f"Has table: {chunk.contains_table}, Has code: {chunk.contains_code}")
          print(f"Heading: {chunk.heading_context}")
          print(chunk.text[:200])
          print()

````

---

## Pitfalls & Common Mistakes

| Mistake | Impact | Fix |
|---------|--------|-----|
| **Splitting tables** | Data associations broken | Detect tables and keep them whole |
| **Splitting code blocks** | Code becomes unparseable | Detect ``` markers and keep code intact |
| **Ignoring query patterns** | Chunks don't match what users ask | Analyze common queries and align chunk granularity |
| **Not attaching heading context** | Chunks lack topic identification | Always include the parent heading |
| **Same strategy for all doc types** | Tables in API docs ≠ prose in guides | Use structure-aware for mixed content, simpler for prose |

---

## Key Takeaways

1. **Chunk-query alignment** is as important as chunk quality — your chunks must match what users ask.
2. **Never split tables or code blocks** — they lose all meaning when broken.
3. **Use headings as natural boundaries** — they signal topic changes.
4. **Diagnose retrieval failures** by checking if the answer exists in any single chunk.
5. **Mixed-content documents** (tables + prose + code) need structure-aware chunking; pure prose doesn't.
````
