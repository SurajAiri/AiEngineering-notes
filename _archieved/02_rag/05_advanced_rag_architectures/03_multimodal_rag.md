# Multimodal RAG — Images, Tables & Cross-Modal Retrieval

## Why It Matters

Documents aren't just text — they contain **images, charts, tables, and diagrams** that often hold critical information. Standard text RAG ignores visual content entirely. Multimodal RAG retrieves and reasons over multiple content types.

```
STANDARD RAG MISSES VISUAL CONTENT:

  Document: "Server Architecture Overview"
  ┌─────────────────────────────────────┐
  │  Text: "Our system uses a          │
  │   microservice architecture..."     │  ← Text RAG captures this
  │                                     │
  │  [ARCHITECTURE DIAGRAM]             │  ← Text RAG MISSES this entirely
  │   ┌─────┐    ┌────────┐            │     (the diagram shows load balancer,
  │   │ API ├───▶│  Queue  │            │      3 services, and a database)
  │   └──┬──┘    └───┬────┘            │
  │      │           │                  │
  │   ┌──▼──┐    ┌───▼───┐            │
  │   │ DB  │    │Worker │             │
  │   └─────┘    └───────┘            │
  │                                     │
  │  [TABLE: Service SLAs]              │  ← Text RAG may lose table structure
  │   Service  | Latency | Uptime      │
  │   API      | 50ms    | 99.99%      │
  │   Worker   | 200ms   | 99.9%       │
  └─────────────────────────────────────┘

MULTIMODAL RAG:
  • Embeds images alongside text chunks
  • Can answer "What's the architecture?" from the diagram
  • Preserves table structure for precise data retrieval
```

---

## Multimodal RAG Strategies

```
THREE APPROACHES TO MULTIMODAL RAG:

1. DESCRIBE & EMBED (most common, practical)
   Image → VLM describes it → Text description stored + embedded
   Pro: Works with any text vector store
   Con: Description may miss details

2. NATIVE MULTIMODAL EMBEDDING
   Image → Multimodal encoder → Same embedding space as text
   Pro: Direct image-text search
   Con: Needs multimodal embedding model (CLIP, etc.)

3. LATE BINDING (send images to VLM at generation time)
   Retrieve text chunks + associated images → Send both to VLM
   Pro: VLM sees the actual image
   Con: Larger context, higher cost
```

---

## Strategy 1: Describe & Embed (Most Practical)

```python
"""
Multimodal RAG via image description.
Images are described by a VLM, then the descriptions are embedded and
retrieved like regular text chunks.

Requirements: pip install openai sentence-transformers faiss-cpu numpy pillow
"""

import base64
import json
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass
from openai import OpenAI
from sentence_transformers import SentenceTransformer


@dataclass
class MultimodalChunk:
    """A chunk that can be text, image description, or table."""
    content: str          # text content or image description
    content_type: str     # "text" | "image" | "table"
    source_file: str      # original document
    page: int = 0
    image_path: str = ""  # path to original image (if applicable)


class MultimodalRAG:
    """
    RAG system that handles text, images, and tables.

    Strategy: Convert everything to text, embed uniformly.
    Images → VLM description → text embedding
    Tables → Structured text → text embedding
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        vlm_model: str = "gpt-4o-mini",
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.llm = OpenAI()
        self.vlm_model = vlm_model
        self.chunks: list[MultimodalChunk] = []
        self.index = None

    def describe_image(self, image_path: str) -> str:
        """Use a VLM to generate a text description of an image."""
        image_data = Path(image_path).read_bytes()
        base64_image = base64.b64encode(image_data).decode("utf-8")

        # Determine MIME type
        suffix = Path(image_path).suffix.lower()
        mime_types = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
        mime = mime_types.get(suffix, "image/png")

        response = self.llm.chat.completions.create(
            model=self.vlm_model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe this image in detail for a technical documentation "
                            "context. Include all text, labels, numbers, relationships, "
                            "and structure visible. If it's a diagram, describe the "
                            "components and connections. If it's a chart, describe the "
                            "data and trends."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64_image}",
                        },
                    },
                ],
            }],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()

    def add_text_chunk(self, text: str, source: str, page: int = 0):
        """Add a regular text chunk."""
        self.chunks.append(MultimodalChunk(
            content=text,
            content_type="text",
            source_file=source,
            page=page,
        ))

    def add_image(self, image_path: str, source: str, page: int = 0):
        """Add an image by describing it with a VLM."""
        description = self.describe_image(image_path)
        self.chunks.append(MultimodalChunk(
            content=f"[Image description]: {description}",
            content_type="image",
            source_file=source,
            page=page,
            image_path=image_path,
        ))

    def add_table(self, table_text: str, source: str, page: int = 0):
        """Add a table as structured text."""
        self.chunks.append(MultimodalChunk(
            content=f"[Table]: {table_text}",
            content_type="table",
            source_file=source,
            page=page,
        ))

    def build_index(self):
        """Build FAISS index from all chunks."""
        texts = [c.content for c in self.chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))

    def search(self, query: str, k: int = 5) -> list[tuple[MultimodalChunk, float]]:
        """Search across all content types."""
        if self.index is None:
            self.build_index()

        query_emb = self.embedder.encode(query, normalize_embeddings=True)
        scores, indices = self.index.search(
            query_emb.reshape(1, -1).astype(np.float32), min(k, len(self.chunks))
        )

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    def query_with_images(self, question: str, k: int = 5) -> str:
        """
        Full multimodal query: retrieve text + images, send images
        directly to VLM for generation (late binding).
        """
        results = self.search(question, k=k)

        # Separate text and image results
        text_context = []
        image_paths = []

        for chunk, score in results:
            text_context.append(chunk.content)
            if chunk.content_type == "image" and chunk.image_path:
                image_paths.append(chunk.image_path)

        # Build prompt with text context
        context = "\n---\n".join(text_context)

        # Build message content
        content = [
            {
                "type": "text",
                "text": f"Answer the question using the context and images provided.\n\nContext:\n{context}\n\nQuestion: {question}",
            },
        ]

        # Add actual images for VLM to see
        for img_path in image_paths[:3]:  # limit to 3 images
            try:
                img_data = base64.b64encode(Path(img_path).read_bytes()).decode()
                suffix = Path(img_path).suffix.lower()
                mime = {".png": "image/png", ".jpg": "image/jpeg"}.get(suffix, "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_data}"},
                })
            except FileNotFoundError:
                pass

        response = self.llm.chat.completions.create(
            model=self.vlm_model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


# ─── Usage ───
if __name__ == "__main__":
    rag = MultimodalRAG()

    # Add text chunks
    rag.add_text_chunk(
        "Our API service handles 10,000 requests per second with a p99 latency of 45ms.",
        source="architecture.pdf",
        page=1,
    )
    rag.add_text_chunk(
        "The worker service processes background jobs from the message queue.",
        source="architecture.pdf",
        page=2,
    )

    # Add table
    rag.add_table(
        "Service | Latency (p99) | Uptime\n"
        "API     | 45ms          | 99.99%\n"
        "Worker  | 200ms         | 99.9%\n"
        "Queue   | 5ms           | 99.95%",
        source="architecture.pdf",
        page=3,
    )

    # Add image (would normally come from document parsing)
    # rag.add_image("diagrams/architecture.png", source="architecture.pdf", page=2)

    rag.build_index()

    # Search across all content types
    results = rag.search("What are the service latency numbers?")
    for chunk, score in results:
        print(f"[{chunk.content_type}] ({score:.3f}) {chunk.content[:80]}")
```

---

## Table Extraction & Retrieval

```python
"""
Structured table handling for RAG.
Tables need special treatment to preserve structure during chunking.
"""

from dataclasses import dataclass


@dataclass
class ParsedTable:
    headers: list[str]
    rows: list[list[str]]
    caption: str = ""
    source: str = ""


def table_to_searchable_text(table: ParsedTable) -> list[str]:
    """
    Convert a table into multiple searchable text representations.

    Instead of storing the raw table, create one text chunk per row
    so each row is independently searchable.
    """
    chunks = []

    # Full table as structured text (for overview queries)
    full_text = f"Table: {table.caption}\n"
    full_text += " | ".join(table.headers) + "\n"
    for row in table.rows:
        full_text += " | ".join(row) + "\n"
    chunks.append(full_text)

    # Each row as a sentence (for specific value queries)
    for row in table.rows:
        row_text = f"{table.caption}: " if table.caption else ""
        row_text += ", ".join(
            f"{header}: {value}"
            for header, value in zip(table.headers, row)
        )
        chunks.append(row_text)

    return chunks


# Example
table = ParsedTable(
    headers=["Service", "Latency (p99)", "Uptime", "Region"],
    rows=[
        ["API Gateway", "45ms", "99.99%", "us-east-1"],
        ["Auth Service", "120ms", "99.95%", "us-east-1"],
        ["ML Pipeline", "2.5s", "99.5%", "us-west-2"],
    ],
    caption="Service Level Agreements",
)

chunks = table_to_searchable_text(table)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(f"  {chunk}")
    print()

# Chunk 0: Full table
# Chunk 1: "Service Level Agreements: Service: API Gateway, Latency (p99): 45ms, ..."
# Chunk 2: "Service Level Agreements: Service: Auth Service, ..."
# Chunk 3: "Service Level Agreements: Service: ML Pipeline, ..."
```

---

## When to Use Each Multimodal Strategy

```
┌──────────────────────────┬─────────────────────────────────┐
│ Strategy                 │ Best For                        │
├──────────────────────────┼─────────────────────────────────┤
│ Describe & Embed         │ Most cases. Simple, works with  │
│ (VLM → text → embed)    │ any vector store. Good balance. │
│                          │                                 │
│ Native Multimodal Embed  │ Large image collections (e-comm │
│ (CLIP / SigLIP)         │ product search, medical imaging)│
│                          │                                 │
│ Late Binding             │ When visual detail matters.     │
│ (send images to VLM)    │ Diagrams, charts, screenshots.  │
│                          │ Higher cost but highest quality. │
│                          │                                 │
│ Hybrid (Describe +       │ Best quality. Describe for      │
│  Late Bind)              │ retrieval, then send original   │
│                          │ image to VLM for generation.    │
└──────────────────────────┴─────────────────────────────────┘
```

---

## Pitfalls & Common Mistakes

| Mistake                                       | Impact                                   | Fix                                                           |
| --------------------------------------------- | ---------------------------------------- | ------------------------------------------------------------- |
| **Ignoring images entirely**                  | Missing critical information in diagrams | Use VLM to describe images, embed descriptions                |
| **Embedding raw table text**                  | Garbled structure, poor retrieval        | Convert tables to row-level searchable text                   |
| **VLM descriptions too brief**                | "A diagram" isn't useful for retrieval   | Use detailed prompts; ask for labels, numbers, relationships  |
| **Sending too many images to VLM**            | Expensive, slow, may hit token limits    | Limit to 2-3 most relevant images per query                   |
| **No fallback for image extraction failures** | Silent loss of visual content            | Log extraction failures; have text-only fallback              |
| **Same embedding model for images and text**  | Poor cross-modal alignment               | Use dedicated multimodal models (CLIP) or describe-then-embed |

---

## Key Takeaways

1. **Documents are multimodal** — ignoring images and tables means missing information.
2. **Describe & Embed is the pragmatic default** — VLM generates text descriptions, embed those.
3. **Tables need structure preservation** — convert to row-level chunks for precision retrieval.
4. **Late binding (sending images to VLM at generation time) gives highest quality** but costs more.
5. **Start with text-only RAG, add multimodal when visual content clearly matters** (technical docs, medical, product catalogs).
