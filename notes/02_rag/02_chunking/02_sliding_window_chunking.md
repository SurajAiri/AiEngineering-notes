# Sliding Window Chunking

## Why This Matters

Sliding window chunking is the simplest improvement over fixed-size chunking. By overlapping consecutive chunks, you ensure that information near chunk boundaries isn't lost. If a key sentence falls right at the boundary between two chunks, at least one chunk will contain the complete sentence.

---

## How It Works

```
Fixed-size (no overlap):
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Chunk 1  │ │ Chunk 2  │ │ Chunk 3  │
│          │ │          │ │          │
└──────────┘ └──────────┘ └──────────┘
             ^
             Information at this boundary
             may be split and lost


Sliding window (with overlap):
┌──────────────┐
│   Chunk 1    │
│         ┌────┼──────────┐
│         │OVER│  Chunk 2 │
└─────────┤LAP │     ┌────┼──────────┐
          │    │     │OVER│  Chunk 3 │
          └────┼─────┤LAP │          │
               │     │    │          │
               └─────┤    │          │
                      └────┴──────────┘

The "OVERLAP" zones exist in BOTH adjacent chunks,
so boundary information is preserved.
```

---

## The Math

```
chunk_size = 500 tokens
overlap    = 100 tokens
step_size  = chunk_size - overlap = 400 tokens

Document: 2000 tokens
Chunks (no overlap):  2000 / 500 = 4 chunks
Chunks (with overlap): ceil((2000 - 100) / 400) + 1 ≈ 6 chunks

More chunks = more storage + compute, but less information loss.
```

---

## Simple Code

```python
def sliding_window_chunk(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[str]:
    """
    Chunk text with overlapping windows.
    chunk_size and overlap are in characters.
    """
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # skip empty chunks
            chunks.append(chunk)
        if end >= len(text):
            break

    return chunks


text = (
    "Sentence one about vectors. "
    "Sentence two about embeddings. "
    "Sentence three about HNSW algorithm. "
    "Sentence four about search. "
    "Sentence five about retrieval."
)

chunks = sliding_window_chunk(text, chunk_size=80, overlap=30)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: '{chunk}'")
    print()

# Notice how consecutive chunks share overlapping text.
```

---

## Visualizing Overlap

```python
"""
Visual demonstration of how overlap works.
"""

def visualize_overlap(text: str, chunk_size: int, overlap: int):
    """Show where each chunk starts and ends."""
    step = chunk_size - overlap
    chunks = []

    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunks.append((start, end))
        if end >= len(text):
            break

    # Visualize
    print(f"Text length: {len(text)} chars")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}, Step: {step}")
    print(f"Total chunks: {len(chunks)}")
    print()

    for i, (start, end) in enumerate(chunks):
        # Create a visual bar
        bar = [' '] * len(text)
        for j in range(start, min(end, len(text))):
            bar[j] = '█'

        # Mark overlap with previous chunk
        if i > 0:
            prev_start, prev_end = chunks[i-1]
            for j in range(start, min(prev_end, end)):
                bar[j] = '▓'  # overlap region

        print(f"Chunk {i} [{start:3d}-{end:3d}]: {''.join(bar[:60])}")

    print(f"\n█ = unique  ▓ = overlap")


text = "A" * 200  # 200-char document
visualize_overlap(text, chunk_size=60, overlap=15)

# Output shows overlap regions clearly:
# Chunk 0 [  0- 60]: ████████████████████████████████████████████████████████████
# Chunk 1 [ 45-105]:                                              ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓████████████████████████████████████████████
# ...
```

---

## Production Code — Token-Based Sliding Window

```python
"""
Production sliding window chunker with token awareness,
sentence boundary respect, and metadata tracking.

Requirements: pip install tiktoken
"""

import re
import tiktoken
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    text: str
    index: int
    token_count: int
    start_char: int
    end_char: int
    overlap_with_previous: int  # tokens shared with previous chunk
    overlap_with_next: int      # tokens shared with next chunk


class SlidingWindowChunker:
    """
    Token-aware sliding window chunker.
    Respects sentence boundaries to avoid mid-sentence splits.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        encoding: str = "cl100k_base",
        min_chunk_size: int = 50,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding(encoding)
        self.min_chunk_size = min_chunk_size

        if overlap >= chunk_size:
            raise ValueError("Overlap must be less than chunk_size")

    def chunk(self, text: str) -> list[Chunk]:
        """Split text using sliding window with sentence awareness."""
        if not text.strip():
            return []

        sentences = self._split_sentences(text)
        sentence_tokens = [
            len(self.tokenizer.encode(s)) for s in sentences
        ]

        chunks = []
        chunk_index = 0
        sent_idx = 0
        char_pos = 0

        while sent_idx < len(sentences):
            # Build a chunk up to chunk_size tokens
            chunk_sentences = []
            chunk_token_count = 0

            # First, add overlap sentences from previous chunk
            overlap_tokens = 0
            if chunks and self.overlap > 0:
                # Look back to find overlap sentences
                prev_chunk_sents = chunks[-1]._sentences
                overlap_sents = []
                for s in reversed(prev_chunk_sents):
                    s_tokens = len(self.tokenizer.encode(s))
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sents.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break
                chunk_sentences.extend(overlap_sents)
                chunk_token_count += overlap_tokens

            # Add new sentences up to chunk_size
            start_sent_idx = sent_idx
            while sent_idx < len(sentences):
                s_tokens = sentence_tokens[sent_idx]
                if chunk_token_count + s_tokens > self.chunk_size and chunk_sentences:
                    break
                chunk_sentences.append(sentences[sent_idx])
                chunk_token_count += s_tokens
                sent_idx += 1

            # Skip if we didn't add any new sentences (prevent infinite loop)
            if sent_idx == start_sent_idx:
                sent_idx += 1
                continue

            chunk_text = ' '.join(chunk_sentences)

            chunk_obj = Chunk(
                text=chunk_text,
                index=chunk_index,
                token_count=chunk_token_count,
                start_char=char_pos,
                end_char=char_pos + len(chunk_text),
                overlap_with_previous=overlap_tokens,
                overlap_with_next=0,  # calculated after
            )
            # Store sentences for overlap calculation (internal use)
            chunk_obj._sentences = chunk_sentences

            # Update previous chunk's overlap_with_next
            if chunks:
                chunks[-1].overlap_with_next = overlap_tokens

            chunks.append(chunk_obj)
            chunk_index += 1
            char_pos += len(chunk_text) - (overlap_tokens * 4)  # approximate

        # Remove internal _sentences attribute
        for c in chunks:
            if hasattr(c, '_sentences'):
                delattr(c, '_sentences')

        # Filter out tiny trailing chunks
        if chunks and chunks[-1].token_count < self.min_chunk_size:
            if len(chunks) > 1:
                # Merge into previous chunk
                last = chunks.pop()
                chunks[-1].text += ' ' + last.text
                chunks[-1].token_count += last.token_count
                chunks[-1].end_char = last.end_char

        logger.info(
            f"Created {len(chunks)} chunks "
            f"(avg {sum(c.token_count for c in chunks) / len(chunks):.0f} tokens)"
        )
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]


# ─── Usage ───
if __name__ == "__main__":
    chunker = SlidingWindowChunker(
        chunk_size=80,    # small for demo
        overlap=20,
    )

    document = """
    Vector databases store high-dimensional embeddings for fast similarity search.
    They are essential for modern AI applications like semantic search and RAG.
    The most popular indexing algorithm is HNSW, which builds a multi-layer graph.
    Each layer contains a subset of data points connected by proximity.
    The top layer has few points with long-range connections for fast navigation.
    Lower layers have more points for precise nearest-neighbor search.
    Queries start at the top and navigate down through the layers.
    This enables logarithmic search time for billion-scale datasets.
    Other algorithms like IVF partition the space into clusters.
    DiskANN enables search on datasets too large to fit in memory.
    """

    chunks = chunker.chunk(document)
    for chunk in chunks:
        print(
            f"Chunk {chunk.index} "
            f"({chunk.token_count} tokens, "
            f"overlap_prev={chunk.overlap_with_previous}, "
            f"overlap_next={chunk.overlap_with_next}):"
        )
        print(f"  {chunk.text[:120]}...")
        print()
```

---

## Choosing Overlap Size

```
Too little overlap (0-5%):
  • Information loss at boundaries
  • Split concepts
  • ❌ Defeats the purpose of sliding window

Good overlap (10-20%):
  • Preserves boundary context
  • Reasonable storage overhead
  • ✅ Sweet spot for most use cases

Too much overlap (>30%):
  • Excessive storage and compute
  • Redundant retrieval results
  • Deduplication needed at query time
  • ❌ Diminishing returns

Rule of thumb: overlap ≈ 1-3 sentences or 10-15% of chunk_size
```

---

## Pitfalls & Common Mistakes

| Mistake                          | Impact                                 | Fix                                         |
| -------------------------------- | -------------------------------------- | ------------------------------------------- |
| **No overlap at all**            | Boundary information lost              | Add 10-20% overlap                          |
| **Overlap too large**            | Redundant chunks, wasted compute       | Keep overlap < 20% of chunk size            |
| **Not deduplicating results**    | Same passage returned multiple times   | Deduplicate overlapping chunks in retrieval |
| **Overlap > chunk_size**         | Infinite loop or broken chunks         | Always validate overlap < chunk_size        |
| **Ignoring sentence boundaries** | Overlapping region starts mid-sentence | Align overlap to sentence boundaries        |

---

## Fixed-Size vs Sliding Window

| Aspect         | Fixed-Size             | Sliding Window           |
| -------------- | ---------------------- | ------------------------ |
| Boundary info  | Lost                   | Preserved                |
| Storage        | Baseline               | 10-30% more              |
| Implementation | Simpler                | Slightly more complex    |
| Retrieval      | May miss boundary info | Better recall            |
| Dedup needed   | No                     | Yes (overlapping chunks) |

---

## Key Takeaways

1. **Sliding window = fixed-size + overlap** — it's the simplest improvement over basic chunking.
2. **10-20% overlap** is the sweet spot — enough to preserve context, not so much it wastes resources.
3. **Always respect sentence boundaries** in the overlap region.
4. **Deduplication matters** — overlapping chunks can cause the same passage to show up multiple times in retrieval.
5. **This is still a simple strategy** — for structured documents, consider semantic or hierarchical chunking instead.
