# Semantic Chunking

## Why This Matters

Fixed-size chunking ignores meaning — it splits by character or token count, regardless of where ideas start and end. **Semantic chunking** splits text where the _meaning changes_, producing chunks that each contain a coherent idea. This gives better embeddings and more relevant retrieval.

---

## How It Works

```
Fixed-size chunking:
"Vector databases store embeddings. | They enable similarity search. HNSW is | a graph algorithm. It uses layers."
  ↑ Split at character count              ↑ Split mid-topic!

Semantic chunking:
"Vector databases store embeddings. They enable similarity search." | "HNSW is a graph algorithm. It uses layers."
  ↑ Split where topic changes                                          ↑ New topic = new chunk
```

### The Algorithm

```
Step 1: Split into sentences
    s1, s2, s3, s4, s5, s6, s7...

Step 2: Embed each sentence (or sentence pair)
    e1, e2, e3, e4, e5, e6, e7...

Step 3: Compute similarity between consecutive sentences
    sim(e1,e2)=0.92, sim(e2,e3)=0.88, sim(e3,e4)=0.41, sim(e4,e5)=0.85...
                                          ↑
                                       LOW similarity = topic change!

Step 4: Split at low-similarity points
    [s1, s2, s3] | [s4, s5, s6, s7]
     Topic A         Topic B
```

```
Similarity between consecutive sentences:

1.0 ┤
    │  ●   ●
0.8 ┤    ●       ●   ●           ●   ●
    │                    ●               ●
0.6 ┤
    │
0.4 ┤         ●                       ●
    │                         ●
0.2 ┤
    │
0.0 ┤─────────────────────────────────────
     s1-s2 s2-s3 s3-s4 s4-s5 ...

         ↑           ↑             ↑
    Split here!  Split here!   Split here!
    (big drops in similarity)
```

---

## Simple Code — Understand the Concept

```python
"""
Minimal semantic chunking: split where meaning changes.
Requirements: pip install sentence-transformers numpy
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_chunk_simple(text: str, threshold: float = 0.5) -> list[str]:
    """
    Split text into chunks where semantic similarity drops.

    1. Split into sentences
    2. Embed each sentence
    3. Find where consecutive similarity drops below threshold
    4. Group sentences between drop points into chunks
    """
    # Step 1: Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return sentences

    # Step 2: Embed all sentences
    embeddings = model.encode(sentences)

    # Step 3: Compute cosine similarity between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        similarities.append(sim)

    # Step 4: Find split points (where similarity drops)
    chunks = []
    current_chunk = [sentences[0]]

    for i, sim in enumerate(similarities):
        if sim < threshold:
            # Topic change detected — start new chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i + 1]]
        else:
            current_chunk.append(sentences[i + 1])

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


# Example with clear topic changes
text = """
Vector databases store high-dimensional embeddings for similarity search.
They are optimized for nearest-neighbor queries over dense vectors.
Common implementations include Pinecone, Weaviate, and Qdrant.

Kubernetes is a container orchestration platform. It manages the deployment
and scaling of containerized applications across clusters.

Neural networks learn through backpropagation. The gradient of the loss
function is computed and used to update weights layer by layer.
"""

chunks = semantic_chunk_simple(text, threshold=0.5)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk[:100]}...")
    print()
```

---

## Better Approach — Percentile-Based Threshold

```python
"""
Instead of a fixed threshold, use statistical methods to find
natural breakpoints in similarity.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import re

model = SentenceTransformer('all-MiniLM-L6-v2')


def semantic_chunk_percentile(
    text: str,
    breakpoint_percentile: int = 25,
    min_chunk_sentences: int = 2,
    max_chunk_sentences: int = 15,
) -> list[str]:
    """
    Use percentile-based breakpoints instead of a fixed threshold.
    Splits at the lowest N% of similarity scores.

    percentile=25 means: split at the 25% lowest similarity points.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= min_chunk_sentences:
        return [' '.join(sentences)]

    embeddings = model.encode(sentences)

    # Consecutive similarities
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = np.dot(embeddings[i], embeddings[i+1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1])
        )
        similarities.append(float(sim))

    # Find threshold using percentile
    threshold = np.percentile(similarities, breakpoint_percentile)

    # Identify breakpoints
    breakpoints = []
    for i, sim in enumerate(similarities):
        if sim < threshold:
            breakpoints.append(i + 1)  # split AFTER sentence i

    # Enforce min/max chunk size constraints
    chunks = []
    start = 0
    for bp in breakpoints:
        chunk_size = bp - start
        if chunk_size >= min_chunk_sentences:
            chunks.append(' '.join(sentences[start:bp]))
            start = bp

    # Don't forget final chunk
    if start < len(sentences):
        remaining = ' '.join(sentences[start:])
        if chunks and len(sentences[start:]) < min_chunk_sentences:
            # Merge tiny tail into last chunk
            chunks[-1] += ' ' + remaining
        else:
            chunks.append(remaining)

    return chunks


# Example
text = """
Machine learning models learn patterns from data. They generalize to
make predictions on unseen examples. This process involves training
on labeled datasets.

The attention mechanism revolutionized NLP. Transformers use
self-attention to weigh the importance of different tokens. This
replaced recurrent architectures like LSTMs.

Docker containers package applications with dependencies. They ensure
consistent behavior across development and production environments.
Container orchestration tools like Kubernetes manage these at scale.
"""

chunks = semantic_chunk_percentile(text, breakpoint_percentile=30)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk[:120]}...")
    print()
```

---

## Production Code — Full Semantic Chunker

```python
"""
Production semantic chunker with:
- Configurable similarity computation (window-based for efficiency)
- Token-aware chunk size limits
- Breakpoint detection methods (percentile, stddev, gradient)
- Metadata output

Requirements: pip install sentence-transformers numpy tiktoken
"""

import re
import logging
import numpy as np
import tiktoken
from dataclasses import dataclass, field
from enum import Enum
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BreakpointMethod(Enum):
    PERCENTILE = "percentile"
    STDDEV = "stddev"
    GRADIENT = "gradient"


@dataclass
class SemanticChunk:
    text: str
    index: int
    sentences: list[str]
    token_count: int
    avg_internal_similarity: float
    breakpoint_score: float  # similarity at the boundary


class SemanticChunker:
    """
    Production semantic chunker.

    Parameters:
        embedding_model: Sentence transformer model name
        breakpoint_method: How to detect topic changes
        breakpoint_threshold: Sensitivity (percentile value or stddev multiplier)
        max_chunk_tokens: Hard limit on chunk size
        min_chunk_sentences: Minimum sentences per chunk
        buffer_size: Window size for smoothing similarity (reduces noise)
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        breakpoint_method: BreakpointMethod = BreakpointMethod.PERCENTILE,
        breakpoint_threshold: float = 25.0,
        max_chunk_tokens: int = 512,
        min_chunk_sentences: int = 2,
        buffer_size: int = 1,
    ):
        self.model = SentenceTransformer(embedding_model)
        self.method = breakpoint_method
        self.threshold = breakpoint_threshold
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_sentences = min_chunk_sentences
        self.buffer_size = buffer_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[SemanticChunk]:
        """Split text into semantically coherent chunks."""
        sentences = self._split_sentences(text)

        if len(sentences) <= self.min_chunk_sentences:
            return [self._make_chunk(sentences, 0, 0.0)]

        # Embed sentences (with buffering for smoother similarity)
        embeddings = self._embed_with_buffer(sentences)

        # Compute consecutive similarities
        similarities = self._compute_similarities(embeddings)

        # Find breakpoints
        breakpoints = self._find_breakpoints(similarities)

        # Build chunks respecting size constraints
        chunks = self._build_chunks(sentences, breakpoints, similarities)

        logger.info(
            f"Semantic chunking: {len(sentences)} sentences → {len(chunks)} chunks"
        )
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr)\.\s', r'\1<DOT> ', text)
        text = re.sub(r'\b(e\.g|i\.e|vs|etc)\.\s', r'\1<DOT> ', text)

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        # Restore dots
        sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
        return sentences

    def _embed_with_buffer(self, sentences: list[str]) -> np.ndarray:
        """
        Embed sentences with context buffer.
        Instead of embedding individual sentences, embed groups of sentences
        for more stable similarity computation.
        """
        if self.buffer_size <= 0:
            return self.model.encode(sentences, normalize_embeddings=True)

        # Create buffered text (sentence + buffer neighbors)
        buffered = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            combined = ' '.join(sentences[start:end])
            buffered.append(combined)

        return self.model.encode(buffered, normalize_embeddings=True)

    def _compute_similarities(self, embeddings: np.ndarray) -> list[float]:
        """Compute cosine similarity between consecutive embeddings."""
        sims = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            sims.append(sim)
        return sims

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        """Find indices where the text should be split."""
        if not similarities:
            return []

        sims = np.array(similarities)

        if self.method == BreakpointMethod.PERCENTILE:
            threshold = np.percentile(sims, self.threshold)
            breakpoints = [i + 1 for i, s in enumerate(sims) if s < threshold]

        elif self.method == BreakpointMethod.STDDEV:
            mean = np.mean(sims)
            std = np.std(sims)
            threshold = mean - (self.threshold * std)
            breakpoints = [i + 1 for i, s in enumerate(sims) if s < threshold]

        elif self.method == BreakpointMethod.GRADIENT:
            # Find sharp drops in similarity
            gradients = np.diff(sims)
            threshold = np.percentile(gradients, self.threshold)
            breakpoints = [i + 2 for i, g in enumerate(gradients) if g < threshold]

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return sorted(set(breakpoints))

    def _build_chunks(
        self,
        sentences: list[str],
        breakpoints: list[int],
        similarities: list[float],
    ) -> list[SemanticChunk]:
        """Build chunks from sentences and breakpoints, respecting size limits."""
        chunks = []
        boundaries = [0] + breakpoints + [len(sentences)]

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            group = sentences[start:end]

            # Check if chunk exceeds token limit
            combined = ' '.join(group)
            token_count = len(self.tokenizer.encode(combined))

            if token_count > self.max_chunk_tokens:
                # Sub-split large chunks using size-based splitting
                sub_chunks = self._split_oversized(group)
                for sub in sub_chunks:
                    chunks.append(sub)
            elif len(group) < self.min_chunk_sentences and chunks:
                # Merge tiny chunks into previous
                chunks[-1].text += ' ' + combined
                chunks[-1].sentences.extend(group)
                chunks[-1].token_count += token_count
            else:
                boundary_score = similarities[end - 1] if end - 1 < len(similarities) else 0.0
                chunks.append(self._make_chunk(group, len(chunks), boundary_score))

        # Re-index
        for i, chunk in enumerate(chunks):
            chunk.index = i

        return chunks

    def _split_oversized(self, sentences: list[str]) -> list[SemanticChunk]:
        """Split oversized sentence groups into smaller chunks."""
        chunks = []
        current = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(self.tokenizer.encode(sent))
            if current_tokens + sent_tokens > self.max_chunk_tokens and current:
                chunks.append(self._make_chunk(current, len(chunks), 0.0))
                current = [sent]
                current_tokens = sent_tokens
            else:
                current.append(sent)
                current_tokens += sent_tokens

        if current:
            chunks.append(self._make_chunk(current, len(chunks), 0.0))

        return chunks

    def _make_chunk(
        self, sentences: list[str], index: int, boundary_score: float
    ) -> SemanticChunk:
        text = ' '.join(sentences)
        token_count = len(self.tokenizer.encode(text))

        # Compute average internal similarity
        if len(sentences) > 1:
            embs = self.model.encode(sentences, normalize_embeddings=True)
            internal_sims = []
            for i in range(len(embs) - 1):
                internal_sims.append(float(np.dot(embs[i], embs[i + 1])))
            avg_sim = np.mean(internal_sims)
        else:
            avg_sim = 1.0

        return SemanticChunk(
            text=text,
            index=index,
            sentences=sentences,
            token_count=token_count,
            avg_internal_similarity=float(avg_sim),
            breakpoint_score=boundary_score,
        )


# ─── Usage ───
if __name__ == "__main__":
    chunker = SemanticChunker(
        breakpoint_method=BreakpointMethod.PERCENTILE,
        breakpoint_threshold=30,
        max_chunk_tokens=200,
        min_chunk_sentences=2,
        buffer_size=1,
    )

    document = """
    Vector databases are specialized systems for storing high-dimensional
    embeddings. They enable fast similarity search using approximate
    nearest neighbor algorithms. Popular options include Pinecone, Qdrant,
    and Weaviate.

    HNSW is the most widely used indexing algorithm. It constructs a
    multi-layer graph where each layer has fewer nodes. Navigation starts
    at the top layer and refines through lower layers.

    Kubernetes orchestrates container deployment across clusters. It
    handles scaling, load balancing, and self-healing. Pods are the
    smallest deployable units in Kubernetes.

    Retrieval-Augmented Generation combines search with language models.
    Documents are chunked, embedded, and stored in vector databases.
    At query time, relevant chunks are retrieved and passed to an LLM.
    """

    chunks = chunker.chunk(document)
    for chunk in chunks:
        print(f"Chunk {chunk.index} ({chunk.token_count} tokens, "
              f"internal_sim={chunk.avg_internal_similarity:.3f}):")
        print(f"  {chunk.text[:120]}...")
        print()
```

---

## Semantic Chunking vs Fixed-Size

```
                    Fixed-Size              Semantic

Split criteria:     Character/token count   Meaning change
Chunk coherence:    ❌ May split ideas      ✅ Preserves ideas
Chunk size:         ✅ Predictable          ❌ Variable
Speed:              ✅ Fast (no embedding)  ❌ Slow (needs embedding)
Cost:               ✅ Free                 ❌ Embedding cost
Quality:            Medium                  High (for varied content)
Best for:           Uniform text            Multi-topic documents
```

---

## Pitfalls & Common Mistakes

| Mistake                        | Impact                                    | Fix                                                   |
| ------------------------------ | ----------------------------------------- | ----------------------------------------------------- |
| **Fixed threshold**            | Too many/few splits depending on document | Use percentile or stddev-based thresholds             |
| **No size limits**             | Huge chunks when similarity stays high    | Enforce max_chunk_tokens                              |
| **No minimum size**            | Tiny 1-sentence chunks                    | Set min_chunk_sentences                               |
| **Embedding individual words** | Noisy similarity scores                   | Use sentence-level or buffered embeddings             |
| **Ignoring cost**              | Embedding every sentence is expensive     | Batch embeddings; consider if simpler methods suffice |
| **Over-splitting**             | Too many small chunks = poor context      | Tune breakpoint_percentile; start at 20-30            |

---

## Trade-offs

| Factor             | Impact                                                        |
| ------------------ | ------------------------------------------------------------- |
| **Quality**        | Better chunk coherence → better embeddings → better retrieval |
| **Cost**           | Requires embedding every sentence (or buffer group)           |
| **Speed**          | Slower than fixed-size (embedding step)                       |
| **Predictability** | Variable chunk sizes complicate token budget planning         |
| **Complexity**     | More parameters to tune (threshold, buffer, min/max sizes)    |

**Use semantic chunking when:**

- Documents cover multiple topics
- Fixed-size chunks are producing poor retrieval
- Topics change within documents unpredictably
- Quality matters more than speed/cost

**Stick with fixed-size when:**

- Documents are homogeneous (single topic)
- Speed/cost is critical
- You're prototyping and need a quick baseline

---

## Key Takeaways

1. **Semantic chunking splits where meaning changes**, not at arbitrary positions.
2. **Use percentile-based thresholds** — they adapt to each document's similarity distribution.
3. **Always enforce size limits** — semantic coherence doesn't guarantee reasonable chunk sizes.
4. **Buffered embeddings** (embedding sentence + neighbors) produce smoother similarity curves.
5. **It's more expensive** than fixed-size — only use it when quality justifies the cost.
