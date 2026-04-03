# 🤖 LLM Fundamentals & Prompt Engineering

## 📚 Overview

Understanding LLM behavior and mastering prompt engineering are core skills for AI engineers. This module covers how to effectively use and control LLMs.

---

## 🎯 Learning Objectives

- Understand **how LLMs generate text**
- Master **prompt engineering** techniques
- Apply **output control** (structured outputs, JSON mode)
- Manage **context windows** effectively
- Choose the right **model for the task**

---

## 🔬 Core Concepts

### 1. How LLMs Work

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM TEXT GENERATION                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: "The capital of France is"                             │
│                    ↓                                             │
│   ┌─────────────────────────────────────────┐                   │
│   │           TOKENIZATION                   │                   │
│   │   → [The, capital, of, France, is]      │                   │
│   └────────────────┬────────────────────────┘                   │
│                    ↓                                             │
│   ┌─────────────────────────────────────────┐                   │
│   │        TRANSFORMER FORWARD              │                   │
│   │   Process all tokens in parallel        │                   │
│   └────────────────┬────────────────────────┘                   │
│                    ↓                                             │
│   ┌─────────────────────────────────────────┐                   │
│   │      NEXT TOKEN PREDICTION               │                   │
│   │   Output: probability over vocabulary    │                   │
│   │   P("Paris") = 0.82                      │                   │
│   │   P("Lyon") = 0.05                       │                   │
│   │   P("Berlin") = 0.01                     │                   │
│   │   ...                                    │                   │
│   └────────────────┬────────────────────────┘                   │
│                    ↓                                             │
│   ┌─────────────────────────────────────────┐                   │
│   │          SAMPLING                        │                   │
│   │   Select next token based on strategy    │                   │
│   │   (greedy, top-k, top-p, temperature)   │                   │
│   └────────────────┬────────────────────────┘                   │
│                    ↓                                             │
│   Output: "Paris"                                               │
│                                                                  │
│   Repeat until: EOS token or max_tokens                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Sampling Strategies

```python
"""
Controlling LLM output randomness and diversity.
"""
import numpy as np
from typing import List

def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Apply temperature-scaled softmax."""
    scaled = logits / temperature
    exp_logits = np.exp(scaled - np.max(scaled))
    return exp_logits / exp_logits.sum()


# Temperature
"""
Temperature: Controls randomness in sampling.

temp = 0: Greedy (always pick highest probability)
temp = 0.7: Balanced (some variety, mostly sensible)
temp = 1.0: Model's natural distribution
temp > 1.0: More random, creative, potentially nonsensical

Use cases:
- temp ~ 0: Factual Q&A, code generation
- temp ~ 0.7: Conversational, summarization
- temp ~ 1.0: Creative writing, brainstorming
"""

def sample_with_temperature(logits: np.ndarray, temperature: float) -> int:
    probs = softmax(logits, temperature)
    return np.random.choice(len(probs), p=probs)


# Top-K sampling
"""
Top-K: Only consider the K most probable tokens.

K = 1: Greedy
K = 50: Standard choice
K = 100: More variety

Prevents picking extremely unlikely tokens.
"""

def sample_top_k(logits: np.ndarray, k: int) -> int:
    top_indices = np.argsort(logits)[-k:]
    top_logits = logits[top_indices]
    probs = softmax(top_logits)
    return top_indices[np.random.choice(len(probs), p=probs)]


# Top-P (Nucleus) sampling
"""
Top-P: Consider tokens that make up P probability mass.

P = 0.9: Most common choice (covers 90% of probability)
P = 0.95: More variety
P = 0.5: Conservative

Adapts to probability distribution (vs fixed K).
"""

def sample_top_p(logits: np.ndarray, p: float = 0.9) -> int:
    probs = softmax(logits)
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    cumulative = np.cumsum(sorted_probs)
    cutoff_idx = np.searchsorted(cumulative, p) + 1
    
    top_indices = sorted_indices[:cutoff_idx]
    top_probs = probs[top_indices]
    top_probs = top_probs / top_probs.sum()
    
    return top_indices[np.random.choice(len(top_probs), p=top_probs)]


# Frequency penalty
"""
Frequency penalty: Reduce probability of already-used tokens.
Prevents repetition.

penalty = 0: No effect
penalty = 0.5: Moderate reduction
penalty = 2.0: Strong reduction

Implementation: Subtract penalty * frequency_count from logits
"""


# API usage
from openai import OpenAI

client = OpenAI()

def generate_with_params(prompt: str, **kwargs):
    """Generate with explicit sampling parameters."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=kwargs.get("temperature", 1.0),
        top_p=kwargs.get("top_p", 1.0),
        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
        presence_penalty=kwargs.get("presence_penalty", 0.0),
        max_tokens=kwargs.get("max_tokens", 1000)
    )
    return response.choices[0].message.content
```

### 3. Prompt Engineering Techniques

```python
"""
Prompt engineering patterns and techniques.
"""

# Zero-shot prompting
ZERO_SHOT = """
Classify the sentiment of this text as POSITIVE, NEGATIVE, or NEUTRAL.

Text: {text}

Sentiment:"""


# Few-shot prompting
FEW_SHOT = """
Classify the sentiment of the text.

Text: "I love this product!"
Sentiment: POSITIVE

Text: "This is terrible, never buying again."
Sentiment: NEGATIVE

Text: "It's okay, nothing special."
Sentiment: NEUTRAL

Text: "{text}"
Sentiment:"""


# Chain-of-Thought (CoT)
COT_PROMPT = """
Solve this step by step.

Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

Let me work through this step by step:
1. Roger starts with 5 tennis balls
2. He buys 2 cans
3. Each can has 3 balls, so 2 × 3 = 6 new balls
4. Total: 5 + 6 = 11 balls

Answer: 11

Question: {question}

Let me work through this step by step:"""


# System prompt patterns
SYSTEM_PROMPTS = {
    "assistant": """You are a helpful AI assistant. Be concise and accurate.
If you don't know something, say so.""",

    "expert": """You are an expert in {domain}. 
Provide detailed, technical answers with examples.
Cite sources when possible.""",

    "teacher": """You are a patient teacher explaining concepts to beginners.
Use simple language and analogies.
Check understanding with follow-up questions.""",

    "analyst": """You are a data analyst. 
Present information in structured formats: lists, tables, bullet points.
Always show your reasoning."""
}


# Output format control
FORMAT_CONTROL = """
Extract the following information from the text.
Respond ONLY with valid JSON, no other text.

Required fields:
- name: string
- age: integer
- occupation: string
- skills: list of strings

Text: {text}

JSON:"""


# Structured output with OpenAI
from openai import OpenAI
from pydantic import BaseModel
from typing import List

class Person(BaseModel):
    name: str
    age: int
    occupation: str
    skills: List[str]

def extract_structured(text: str) -> Person:
    """Extract structured data using Pydantic."""
    client = OpenAI()
    
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract person information."},
            {"role": "user", "content": text}
        ],
        response_format=Person
    )
    
    return response.choices[0].message.parsed


# Claude XML patterns
CLAUDE_XML_PROMPT = """
<instructions>
Analyze the following code for security vulnerabilities.
</instructions>

<code>
{code}
</code>

<output_format>
For each vulnerability found:
<vulnerability>
  <type>...</type>
  <severity>HIGH|MEDIUM|LOW</severity>
  <line>...</line>
  <description>...</description>
  <fix>...</fix>
</vulnerability>
</output_format>

Analyze:"""


# Role-playing for better outputs
ROLEPLAY_PROMPT = """
You are a senior software engineer at Google conducting a code review.
Be thorough, point out potential issues, and suggest improvements.
Consider:
- Performance
- Security
- Readability
- Best practices

Review this code:
```python
{code}
```

Code Review:"""


# Prompt chaining for complex tasks
class PromptChain:
    """Chain multiple prompts for complex tasks."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def summarize_and_analyze(self, document: str) -> dict:
        """Multi-step analysis."""
        
        # Step 1: Summarize
        summary = self.llm.invoke(f"""
Summarize this document in 3 sentences:

{document}

Summary:""")
        
        # Step 2: Extract key points
        key_points = self.llm.invoke(f"""
Based on this summary, list the 5 key points:

{summary}

Key Points:""")
        
        # Step 3: Generate questions
        questions = self.llm.invoke(f"""
Generate 3 follow-up questions based on these key points:

{key_points}

Questions:""")
        
        return {
            "summary": summary,
            "key_points": key_points,
            "questions": questions
        }
```

### 4. Context Window Management

```python
"""
Managing context windows effectively.
"""
import tiktoken
from typing import List, Dict

class ContextManager:
    """Manage context window for LLM interactions."""
    
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 128000):
        self.encoding = tiktoken.encoding_for_model(model)
        self.max_tokens = max_tokens
        self.reserved_for_output = 4096
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def count_messages_tokens(self, messages: List[Dict]) -> int:
        """Count tokens in chat messages."""
        total = 3  # Priming tokens
        for msg in messages:
            total += 4  # Message formatting
            for key, value in msg.items():
                total += self.count_tokens(str(value))
        return total
    
    def available_tokens(self, messages: List[Dict]) -> int:
        """Tokens available for new content."""
        used = self.count_messages_tokens(messages)
        return self.max_tokens - used - self.reserved_for_output
    
    def truncate_context(
        self, 
        messages: List[Dict], 
        max_context_tokens: int
    ) -> List[Dict]:
        """
        Truncate message history to fit context.
        Always keep: system message, current user message
        Remove oldest messages first.
        """
        if len(messages) <= 2:
            return messages
        
        system_msg = messages[0] if messages[0]["role"] == "system" else None
        current_msg = messages[-1]
        history = messages[1:-1] if system_msg else messages[:-1]
        
        # Calculate token budgets
        fixed_tokens = 0
        if system_msg:
            fixed_tokens += self.count_tokens(system_msg["content"]) + 4
        fixed_tokens += self.count_tokens(current_msg["content"]) + 4
        
        available = max_context_tokens - fixed_tokens
        
        # Keep as many recent messages as possible
        kept_history = []
        total = 0
        
        for msg in reversed(history):
            msg_tokens = self.count_tokens(msg["content"]) + 4
            if total + msg_tokens <= available:
                kept_history.insert(0, msg)
                total += msg_tokens
            else:
                break
        
        # Reconstruct
        result = []
        if system_msg:
            result.append(system_msg)
        result.extend(kept_history)
        result.append(current_msg)
        
        return result


# Context window strategies
"""
Strategy 1: Sliding Window
Keep last N messages, discard older ones.
Simple but loses important context.

Strategy 2: Summary Window  
Periodically summarize older messages.
Keeps gist but loses details.

Strategy 3: Priority Window
Keep system prompt + summary + recent + current.
Good balance for most use cases.

Strategy 4: Retrieval Window
RAG: Retrieve relevant past context.
Best for long conversations.
"""

class SlidingWindowContext:
    """Simple sliding window context management."""
    
    def __init__(self, max_messages: int = 20):
        self.messages = []
        self.max_messages = max_messages
    
    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        
        # Keep system message + last N
        if len(self.messages) > self.max_messages:
            system = self.messages[0] if self.messages[0]["role"] == "system" else None
            if system:
                self.messages = [system] + self.messages[-(self.max_messages-1):]
            else:
                self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self) -> List[Dict]:
        return self.messages.copy()


class SummaryContext:
    """Context with periodic summarization."""
    
    def __init__(self, llm, summarize_every: int = 10):
        self.llm = llm
        self.summarize_every = summarize_every
        self.summary = ""
        self.recent_messages = []
    
    def add(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})
        
        if len(self.recent_messages) >= self.summarize_every:
            self._create_summary()
    
    def _create_summary(self):
        """Summarize recent messages into summary."""
        msgs_text = "\n".join([
            f"{m['role']}: {m['content']}" 
            for m in self.recent_messages
        ])
        
        new_summary = self.llm.invoke(f"""
Summarize this conversation concisely, preserving key information:

Previous summary:
{self.summary}

Recent messages:
{msgs_text}

Updated summary:""")
        
        self.summary = new_summary
        self.recent_messages = []
    
    def get_messages(self, system_prompt: str) -> List[Dict]:
        messages = [{"role": "system", "content": system_prompt}]
        
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Conversation summary: {self.summary}"
            })
        
        messages.extend(self.recent_messages)
        
        return messages
```

### 5. Model Selection Guide

```python
"""
Choosing the right model for the task.
"""

MODEL_GUIDE = {
    "gpt-4o": {
        "strengths": ["Best overall quality", "Vision", "Large context"],
        "weaknesses": ["Cost", "Latency"],
        "use_cases": ["Complex reasoning", "Code generation", "Analysis"],
        "context": 128_000,
        "cost_per_1m_input": 5.00
    },
    "gpt-4o-mini": {
        "strengths": ["Fast", "Cheap", "Good quality"],
        "weaknesses": ["Less capable than 4o"],
        "use_cases": ["Classification", "Simple tasks", "High volume"],
        "context": 128_000,
        "cost_per_1m_input": 0.15
    },
    "claude-3-5-sonnet": {
        "strengths": ["Long context", "Reasoning", "Following instructions"],
        "weaknesses": ["Sometimes verbose"],
        "use_cases": ["Long documents", "Code review", "Writing"],
        "context": 200_000,
        "cost_per_1m_input": 3.00
    },
    "claude-3-5-haiku": {
        "strengths": ["Very fast", "Cheap"],
        "weaknesses": ["Less capable"],
        "use_cases": ["High volume", "Simple extraction"],
        "context": 200_000,
        "cost_per_1m_input": 0.80
    },
    "gemini-1.5-pro": {
        "strengths": ["1M context", "Multimodal", "Good at code"],
        "weaknesses": ["Sometimes inconsistent"],
        "use_cases": ["Very long documents", "Video/audio", "Code"],
        "context": 1_000_000,
        "cost_per_1m_input": 3.50
    },
    "open-source (llama, mistral)": {
        "strengths": ["Free", "Privacy", "Customizable"],
        "weaknesses": ["Self-hosting required", "Less capable"],
        "use_cases": ["On-prem", "Fine-tuning", "Cost-sensitive"],
        "context": "varies",
        "cost_per_1m_input": 0
    }
}

def select_model(
    task_type: str,
    context_length: int,
    latency_requirement: str,
    budget: float
) -> str:
    """Select appropriate model based on requirements."""
    
    # Very long context needed
    if context_length > 200_000:
        return "gemini-1.5-pro"
    
    # Budget constrained
    if budget < 0.5:  # per 1M tokens
        return "gpt-4o-mini"
    
    # Speed critical
    if latency_requirement == "real-time":
        return "gpt-4o-mini"  # or claude-3-5-haiku
    
    # Complex reasoning
    if task_type in ["analysis", "reasoning", "code_generation"]:
        return "gpt-4o"  # or claude-3-5-sonnet
    
    # Default
    return "gpt-4o-mini"
```

---

## 📊 Prompt Engineering Cheat Sheet

| Technique | When to Use | Example |
|-----------|-------------|---------|
| **Zero-shot** | Simple, well-defined tasks | "Translate to French: Hello" |
| **Few-shot** | Tasks needing examples | Show 2-3 examples first |
| **CoT** | Math, logic, reasoning | "Let's think step by step" |
| **System prompt** | Consistent persona/rules | Define role and constraints |
| **Output format** | Structured data needed | "Respond in JSON only" |
| **Role-play** | Domain expertise needed | "You are a senior engineer..." |

---

## 🚨 Common Mistakes

| Mistake | Fix |
|---------|-----|
| Too vague prompt | Be specific about format, length, style |
| Too much context | Focus on what's relevant |
| No examples | Add few-shot examples |
| Wrong temperature | Low for factual, higher for creative |
| Ignoring system prompt | Use it to set consistent behavior |

---

## ➡️ Next Steps

Continue to **[01_working_with_apis.md](./01_working_with_apis.md)** for API integration patterns.
