# 🧠 Transformer Architecture Deep Dive

## 📚 Overview

Transformers are the foundation of modern LLMs and embedding models. Understanding the architecture is essential for prompt engineering, fine-tuning, and debugging.

---

## 🎯 Learning Objectives

- Understand **attention mechanisms** (self-attention, cross-attention)
- Grasp **positional encodings** and context windows
- Learn **tokenization** strategies
- Analyze **compute complexity** and optimizations
- Explore **modern efficiency techniques** (FlashAttention, KV-cache)

---

## 🔬 Core Concepts

### 1. The Transformer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFORMER ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: "The cat sat on the mat"                               │
│          ↓                                                       │
│   ┌─────────────────────────────────────────┐                   │
│   │         TOKENIZATION                     │                   │
│   │   ["The", "cat", "sat", "on", "the", "mat"]                │
│   └────────────────┬────────────────────────┘                   │
│                    ↓                                             │
│   ┌─────────────────────────────────────────┐                   │
│   │      TOKEN + POSITION EMBEDDINGS        │                   │
│   │   [e1+p1, e2+p2, e3+p3, e4+p4, e5+p5, e6+p6]              │
│   └────────────────┬────────────────────────┘                   │
│                    ↓                                             │
│   ┌─────────────────────────────────────────┐                   │
│   │       TRANSFORMER BLOCK (×N)            │                   │
│   │   ┌───────────────────────────────┐     │                   │
│   │   │   Multi-Head Self-Attention   │     │                   │
│   │   └────────────────┬──────────────┘     │                   │
│   │            ↓  + Residual                │                   │
│   │   ┌───────────────────────────────┐     │                   │
│   │   │   Layer Normalization         │     │                   │
│   │   └────────────────┬──────────────┘     │                   │
│   │            ↓                            │                   │
│   │   ┌───────────────────────────────┐     │                   │
│   │   │   Feed-Forward Network        │     │                   │
│   │   └────────────────┬──────────────┘     │                   │
│   │            ↓  + Residual                │                   │
│   │   ┌───────────────────────────────┐     │                   │
│   │   │   Layer Normalization         │     │                   │
│   │   └───────────────────────────────┘     │                   │
│   └────────────────┬────────────────────────┘                   │
│                    ↓                                             │
│   ┌─────────────────────────────────────────┐                   │
│   │            OUTPUT HEAD                   │                   │
│   │   (Classification, Generation, etc.)    │                   │
│   └─────────────────────────────────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Self-Attention Explained

```python
"""
Self-Attention: How tokens attend to each other.

Key insight: Each token creates a Query, Key, and Value.
Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """Single-head self-attention."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Learnable projections
        self.W_q = nn.Linear(d_model, d_model)  # Query
        self.W_k = nn.Linear(d_model, d_model)  # Key
        self.W_v = nn.Linear(d_model, d_model)  # Value
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) - causal mask for decoder
        
        Returns:
            output: (batch, seq_len, d_model)
            attention_weights: (batch, seq_len, seq_len)
        """
        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Compute attention scores
        scores = Q @ K.transpose(-2, -1)  # (batch, seq_len, seq_len)
        scores = scores / math.sqrt(self.d_model)
        
        # Apply mask (for causal/autoregressive decoding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weight values by attention
        output = attn_weights @ V  # (batch, seq_len, d_model)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention: Multiple parallel attention heads."""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Combined projections (more efficient)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V in one shot
        qkv = self.W_qkv(x)  # (batch, seq_len, 3*d_model)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq_len, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        
        # Combine heads
        out = attn @ V  # (batch, n_heads, seq_len, d_k)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        
        return self.W_o(out)


# Why multi-head?
"""
Multiple heads allow the model to attend to different aspects:
- Head 1: Syntax (subject-verb agreement)
- Head 2: Coreference (what does "it" refer to?)
- Head 3: Semantic similarity
- Head 4: Positional patterns

Each head operates independently, then results are concatenated.
"""
```

### 3. Positional Encodings

```python
"""
Positional Encodings: How transformers know token positions.
Without this, "The cat sat on the mat" = "mat the on sat cat The"
"""
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Original positional encoding from "Attention is All You Need".
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Properties:
    - Fixed (not learned)
    - Can extrapolate to longer sequences (in theory)
    - Each position has unique encoding
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor):
        """Add positional encoding to embeddings."""
        return x + self.pe[:, :x.size(1)]


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) - used in Llama, etc.
    
    Key insight: Encode position by rotating embedding vectors.
    q · k = |q||k|cos(θ_q - θ_k)
    
    Benefits:
    - Better length extrapolation
    - Relative position naturally encoded
    - More efficient computation
    """
    
    def __init__(self, d_model: int, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.base = base
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """
        Apply rotary embeddings to queries and keys.
        
        Args:
            q, k: (batch, n_heads, seq_len, head_dim)
            positions: (seq_len,)
        """
        # Compute rotation frequencies
        freqs = 1.0 / (self.base ** (
            torch.arange(0, self.d_model, 2).float() / self.d_model
        ))
        
        # Compute rotation angles
        theta = positions.unsqueeze(1) * freqs.unsqueeze(0)
        
        # Apply rotation
        q_rot = self._rotate(q, theta)
        k_rot = self._rotate(k, theta)
        
        return q_rot, k_rot
    
    def _rotate(self, x: torch.Tensor, theta: torch.Tensor):
        """Rotate pairs of dimensions."""
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos, sin = torch.cos(theta), torch.sin(theta)
        
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)
        
        return rotated.flatten(-2)


# Context window implications
"""
Original Transformer: max 512 tokens (BERT)
GPT-2: 1024 tokens
GPT-3: 2048 tokens
GPT-4: 8K / 32K / 128K tokens
Claude: 100K / 200K tokens

Longer context = more memory, slower inference
Memory: O(n²) for attention weights
Compute: O(n²d) for attention

That's why:
- Long-context models use special attention patterns
- RAG retrieves relevant chunks instead of full documents
"""
```

### 4. Tokenization

```python
"""
Tokenization: Converting text to token IDs.

Methods:
- Word-based: "Hello world" → ["Hello", "world"]
- Character-based: "Hello" → ["H", "e", "l", "l", "o"]
- Subword (BPE, WordPiece, Unigram): "Hello" → ["Hel", "lo"]
"""
import tiktoken

# OpenAI's tokenizer
def explore_tokenization():
    """Explore how different text is tokenized."""
    enc = tiktoken.encoding_for_model("gpt-4o")
    
    examples = [
        "Hello, world!",
        "The quick brown fox",
        "indivisibility",  # Long word
        "GPT-4 is amazing",
        "こんにちは",  # Japanese
        "def foo(): pass",  # Code
        "12345",  # Numbers
    ]
    
    for text in examples:
        tokens = enc.encode(text)
        decoded = [enc.decode([t]) for t in tokens]
        print(f"'{text}' → {len(tokens)} tokens: {decoded}")

# Output:
# 'Hello, world!' → 4 tokens: ['Hello', ',', ' world', '!']
# 'indivisibility' → 4 tokens: ['ind', 'ivis', 'ibility', '']
# 'こんにちは' → 5 tokens (1 per character for non-Latin)


# Token counting for cost estimation
def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens for cost estimation."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_cost(prompt: str, completion: str, model: str = "gpt-4o") -> float:
    """Estimate API cost."""
    enc = tiktoken.encoding_for_model(model)
    
    prompt_tokens = len(enc.encode(prompt))
    completion_tokens = len(enc.encode(completion))
    
    # Rates per 1M tokens (2024 prices)
    rates = {
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0}
    }
    
    r = rates.get(model, rates["gpt-4o"])
    cost = (prompt_tokens * r["input"] + completion_tokens * r["output"]) / 1_000_000
    
    return cost


# Tokenization implications for RAG
"""
When chunking documents for RAG:
1. Token count ≠ word count
2. Different models have different tokenizers
3. Some languages use more tokens per word
4. Code can be token-inefficient

Best practice: Chunk by tokens, not characters
"""
```

### 5. Efficiency Techniques

```python
"""
Modern efficiency techniques for transformers.
"""

# FlashAttention
"""
FlashAttention: Memory-efficient attention.

Standard attention:
1. Compute full attention matrix: O(n²) memory
2. Apply softmax
3. Multiply by V

FlashAttention:
1. Compute attention in blocks (tiles)
2. Never materialize full attention matrix
3. Uses GPU memory hierarchy efficiently

Result: 2-4x faster, much less memory
"""

# KV-Cache for inference
class KVCache:
    """
    Key-Value cache for autoregressive generation.
    
    Problem: Each new token requires attending to ALL previous tokens.
    Native: O(n²) compute for generating n tokens
    
    Solution: Cache K, V for previous positions.
    With cache: O(n) compute for each new token
    """
    
    def __init__(self, n_layers: int, n_heads: int, head_dim: int, max_len: int):
        self.n_layers = n_layers
        self.cache = {}
        
        for layer in range(n_layers):
            self.cache[layer] = {
                'k': torch.zeros(1, n_heads, max_len, head_dim),
                'v': torch.zeros(1, n_heads, max_len, head_dim)
            }
        
        self.current_pos = 0
    
    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Add new K, V to cache."""
        seq_len = k.size(2)
        self.cache[layer]['k'][:, :, self.current_pos:self.current_pos + seq_len] = k
        self.cache[layer]['v'][:, :, self.current_pos:self.current_pos + seq_len] = v
    
    def get(self, layer: int):
        """Get cached K, V up to current position."""
        return (
            self.cache[layer]['k'][:, :, :self.current_pos + 1],
            self.cache[layer]['v'][:, :, :self.current_pos + 1]
        )


# Grouped Query Attention (GQA)
"""
Used in Llama 2, Mistral, etc.

Standard MHA: n_heads query, n_heads key, n_heads value
GQA: n_heads query, n_kv_heads key, n_kv_heads value

n_kv_heads < n_heads (e.g., 8 KV heads for 32 query heads)

Result:
- Smaller KV cache (8x smaller in this example)
- Faster inference
- Minimal quality loss
"""


# Quantization
"""
Reduce precision to save memory and compute.

FP32 (4 bytes) → FP16 (2 bytes) → INT8 (1 byte) → INT4 (0.5 bytes)

Common methods:
- GPTQ: Post-training quantization
- AWQ: Activation-aware quantization
- GGML/GGUF: Format for quantized models (llama.cpp)

Result:
- 70B model in INT4: ~35GB → ~18GB
- 2-4x faster inference
- 1-3% quality loss
"""
```

---

## 📊 Architecture Comparison

| Model | Params | Layers | Heads | d_model | Context |
|-------|--------|--------|-------|---------|---------|
| BERT-base | 110M | 12 | 12 | 768 | 512 |
| GPT-2 | 1.5B | 48 | 25 | 1600 | 1024 |
| GPT-3 | 175B | 96 | 96 | 12288 | 2048 |
| Llama 2 7B | 7B | 32 | 32 | 4096 | 4096 |
| GPT-4o | ~200B? | ? | ? | ? | 128K |

---

## 🚨 Common Misconceptions

| Misconception | Reality |
|---------------|---------|
| "Transformers understand language" | They learn statistical patterns |
| "More parameters = better" | Architecture and training matter more |
| "Attention is interpretable" | Attention patterns can be misleading |
| "Context window = memory" | It's more like working memory, not storage |

---

## ➡️ Next Steps

Continue to **[01_embeddings_fundamentals.md](./01_embeddings_fundamentals.md)** for embeddings.
