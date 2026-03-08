# Fixed-Size Chunking

## Why This Matters

Fixed-size chunking is the **simplest and most common** chunking strategy. It splits text into chunks of a predetermined size (by character count, word count, or token count). It's the baseline you should understand first — every other strategy is a response to its limitations.

---

## How It Works

```
Input Document (1000 chars):
┌────────────────────────────────────────────────────────┐
│ The quick brown fox jumps over the lazy dog. Lorem     │
│ ipsum dolor sit amet, consectetur adipiscing elit.     │
│ Sed do eiusmod tempor incididunt ut labore et dolore   │
│ magna aliqua. Ut enim ad minim veniam, quis nostrud    │
│ exercitation ullamco laboris nisi ut aliquip ex ea     │
│ commodo consequat...                                   │
└────────────────────────────────────────────────────────┘

Fixed-size chunking (chunk_size=200 chars, no overlap):

┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────┐
│  Chunk 1     │ │  Chunk 2     │ │  Chunk 3     │ │  Chunk 4     │ │Chunk 5 │
│  (200 chars) │ │  (200 chars) │ │  (200 chars) │ │  (200 chars) │ │(remain)│
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └────────┘

Problem: Chunk boundaries may split mid-sentence or mid-word!
```

---

## Simple Code — Character-Based Chunking

```python
def fixed_size_chunk(text: str, chunk_size: int = 500) -> list[str]:
    """Split text into fixed-size chunks by character count."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    return chunks


text = (
    "Vector databases store high-dimensional embeddings. "
    "They enable similarity search across large datasets. "
    "Common algorithms include HNSW, IVF, and DiskANN. "
    "Each has different trade-offs for speed and accuracy."
)

chunks = fixed_size_chunk(text, chunk_size=80)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: '{chunk}'")

# Chunk 0: 'Vector databases store high-dimensional embeddings. They enable similarity sea'
# Chunk 1: 'rch across large datasets. Common algorithms include HNSW, IVF, and DiskANN. '
# Chunk 2: 'Each has different trade-offs for speed and accuracy.'
#
# Notice: "similarity search" got split across chunks 0 and 1!
```

---

## Better Simple Code — Word-Boundary Aware

```python
def fixed_size_chunk_words(text: str, chunk_size: int = 100) -> list[str]:
    """
    Split text into fixed-size chunks, but respect word boundaries.
    chunk_size is in characters, but we don't split mid-word.
    """
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0

    for word in words:
        # +1 for the space
        if current_length + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + (1 if current_chunk else 0)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


text = (
    "Vector databases store high-dimensional embeddings. "
    "They enable similarity search across large datasets. "
    "Common algorithms include HNSW, IVF, and DiskANN. "
    "Each has different trade-offs for speed and accuracy."
)

chunks = fixed_size_chunk_words(text, chunk_size=80)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i} ({len(chunk)} chars): '{chunk}'")

# Now words aren't split mid-word, but sentences may still be split.
```

---

## Production Code — Token-Based Fixed-Size Chunking

```python
"""
Production fixed-size chunking using token counts.
Token-based is more reliable than character-based because:
1. Embedding models have token limits
2. LLMs have context window limits
3. Cost is based on tokens, not characters

Requirements: pip install tiktoken
"""

import tiktoken
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    index: int
    token_count: int
    char_count: int
    start_char: int       # position in original document
    end_char: int


class FixedSizeChunker:
    """
    Token-aware fixed-size chunker with configurable overlap.

    Parameters:
        chunk_size: Target chunk size in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        model: Tokenizer model (for tiktoken)
        respect_sentences: If True, avoid splitting mid-sentence
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model: str = "cl100k_base",
        respect_sentences: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding(model)
        self.respect_sentences = respect_sentences

    def chunk(self, text: str) -> list[Chunk]:
        """Split text into fixed-size token chunks."""
        if not text.strip():
            return []

        if self.respect_sentences:
            return self._chunk_by_sentences(text)
        else:
            return self._chunk_by_tokens(text)

    def _chunk_by_tokens(self, text: str) -> list[Chunk]:
        """Pure token-based chunking (may split mid-sentence)."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Calculate char positions in original text
            # (approximate — token boundaries don't align perfectly with chars)
            char_start = text.find(chunk_text[:20])
            char_start = max(0, char_start)

            chunks.append(Chunk(
                text=chunk_text.strip(),
                index=chunk_index,
                token_count=len(chunk_tokens),
                char_count=len(chunk_text),
                start_char=char_start,
                end_char=char_start + len(chunk_text),
            ))

            # Move forward by (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap
            chunk_index += 1

        return chunks

    def _chunk_by_sentences(self, text: str) -> list[Chunk]:
        """
        Token-based chunking that respects sentence boundaries.
        Fills chunks up to chunk_size tokens, but only breaks at sentence ends.
        """
        import re
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_sentences = []
        current_tokens = 0
        chunk_index = 0
        char_pos = 0

        for sentence in sentences:
            sentence_tokens = len(self.tokenizer.encode(sentence))

            # If adding this sentence exceeds chunk_size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_sentences:
                chunk_text = ' '.join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    index=chunk_index,
                    token_count=current_tokens,
                    char_count=len(chunk_text),
                    start_char=char_pos,
                    end_char=char_pos + len(chunk_text),
                ))
                chunk_index += 1
                char_pos += len(chunk_text) + 1

                # Handle overlap: keep last N tokens worth of sentences
                if self.chunk_overlap > 0:
                    overlap_sentences = []
                    overlap_tokens = 0
                    for s in reversed(current_sentences):
                        s_tokens = len(self.tokenizer.encode(s))
                        if overlap_tokens + s_tokens <= self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_tokens += s_tokens
                        else:
                            break
                    current_sentences = overlap_sentences
                    current_tokens = overlap_tokens
                else:
                    current_sentences = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sentence_tokens

        # Don't forget the last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                index=chunk_index,
                token_count=current_tokens,
                char_count=len(chunk_text),
                start_char=char_pos,
                end_char=char_pos + len(chunk_text),
            ))

        return chunks

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text))


# ─── Usage ───
if __name__ == "__main__":
    chunker = FixedSizeChunker(
        chunk_size=100,       # small for demo
        chunk_overlap=20,
        respect_sentences=True,
    )

    document = """
    Vector databases are specialized database systems designed to store,
    index, and query high-dimensional vector embeddings. These embeddings
    are numerical representations that capture semantic meaning of data
    like text, images, and audio.

    The most common indexing algorithm is HNSW (Hierarchical Navigable
    Small World). It builds a multi-layer graph where each layer contains
    a subset of the data points. The top layer has the fewest points with
    long-range connections, while lower layers have more points with
    shorter connections.

    When a query comes in, the search starts at the top layer and navigates
    down through the layers, getting closer to the nearest neighbors at
    each step. This hierarchical approach enables logarithmic search time
    complexity, making it efficient for large-scale similarity search.

    Other algorithms include IVF (Inverted File Index), which partitions
    the vector space into clusters, and DiskANN, which is optimized for
    disk-based storage of very large datasets.
    """

    chunks = chunker.chunk(document)
    for chunk in chunks:
        print(f"Chunk {chunk.index} ({chunk.token_count} tokens):")
        print(f"  '{chunk.text[:100]}...'")
        print()
```

---

## When to Use Fixed-Size Chunking

```
✅ USE WHEN:                          ❌ AVOID WHEN:

• Starting a new RAG project         • Documents have clear structure
  (baseline strategy)                   (headings, sections)

• Documents are unstructured          • Content is highly variable
  prose (novels, articles)              (code mixed with text)

• You need predictable               • Information density varies
  chunk sizes for cost                  greatly across the document
  estimation
                                      • Tables, lists, and code
• Speed is more important               blocks need to stay intact
  than precision
```

---

## Pitfalls & Common Mistakes

| Mistake                           | Impact                                   | Fix                                               |
| --------------------------------- | ---------------------------------------- | ------------------------------------------------- |
| **Splitting mid-word**            | Broken words in embeddings               | Use word-boundary-aware chunking                  |
| **Splitting mid-sentence**        | Meaning lost across chunk boundary       | Use sentence-aware chunking                       |
| **No overlap**                    | Context lost at boundaries               | Add 10-20% overlap                                |
| **Too-small chunks**              | Insufficient context for embedding       | Minimum ~100 tokens; 256-512 typical              |
| **Too-large chunks**              | Diluted semantic signal                  | Maximum ~1024 tokens; test for your use case      |
| **Character-based sizing**        | Doesn't match LLM/embedding token limits | Use token-based sizing                            |
| **Ignoring the last small chunk** | Losing content at end of document        | Always include the final chunk even if undersized |

---

## Trade-offs

```
Small chunks (100-200 tokens)         Large chunks (500-1000 tokens)
├── More precise retrieval            ├── More context per chunk
├── Less noise per chunk              ├── Better for complex questions
├── Higher storage cost (more chunks) ├── Lower storage cost
├── Risk: insufficient context        ├── Risk: diluted signal
└── Good for: fact lookup, QA         └── Good for: summarization, analysis
```

---

## Key Takeaways

1. **Fixed-size is your baseline** — start here, then improve based on observed failures.
2. **Always use token-based sizing** — characters don't map to model limits.
3. **Respect sentence boundaries** — a split sentence is worse than a slightly oversized chunk.
4. **Add overlap** — 10-20% overlap prevents context loss at boundaries.
5. **Test with real queries** — the "right" chunk size depends on your question types and data.

---

## Popular Libraries

| Library                  | Purpose                       | Install                                |
| ------------------------ | ----------------------------- | -------------------------------------- |
| LangChain text_splitters | Most popular chunking library | `pip install langchain-text-splitters` |
| LlamaIndex node_parser   | Framework-native chunking     | `pip install llama-index`              |
| tiktoken                 | OpenAI token counting         | `pip install tiktoken`                 |

### Quick Example — LangChain RecursiveCharacterTextSplitter

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# The most commonly used splitter in production RAG
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Max characters per chunk
    chunk_overlap=50,      # Overlap between chunks
    separators=["\n\n", "\n", ". ", " ", ""],  # Split priority: paragraphs > lines > sentences
    length_function=len,   # Can swap to token-based: use tiktoken
)

text = "Your long document text here..."
chunks = splitter.split_text(text)
print(f"Created {len(chunks)} chunks, avg size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
```

### Token-Based Sizing with tiktoken

```python
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Use token count instead of character count
enc = tiktoken.encoding_for_model("gpt-4")
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",
    chunk_size=256,      # 256 tokens
    chunk_overlap=30,    # 30 token overlap
)

chunks = splitter.split_text(text)
```

---

## Common Questions

### Q: What chunk size should I start with?

**A:** Start with **256-512 tokens** with 10-20% overlap. This works well for most Q&A use cases. If your questions need more context (e.g., summarization), go larger (512-1024). If you need precise answers (e.g., "what is the price of X?"), go smaller (128-256).

### Q: Characters vs tokens — why does it matter?

**A:** Embedding models and LLMs operate on **tokens**, not characters. The word "tokenization" is 1 word, 14 characters, but ~3 tokens. A chunk of 500 characters might be 100-150 tokens — well under your embedding model's capacity. Always measure in tokens to use your model's full capacity.

### Q: Why RecursiveCharacterTextSplitter over CharacterTextSplitter?

**A:** `RecursiveCharacterTextSplitter` tries multiple separators in order (paragraphs → lines → sentences → words → characters). This means it produces more semantically coherent chunks. `CharacterTextSplitter` only splits on a single separator. Always use Recursive.
