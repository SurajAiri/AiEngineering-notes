# 🔌 Working with LLM APIs

## 📚 Overview

This module covers practical patterns for integrating LLM APIs into production applications, including streaming, error handling, and cost management.

---

## 🎯 Learning Objectives

- **Integrate** with major LLM APIs (OpenAI, Anthropic, Google)
- Implement **streaming** for real-time responses
- Handle **errors and retries** gracefully
- Manage **costs** and rate limits
- Build **provider-agnostic** abstractions

---

## 🔬 Core Concepts

### 1. OpenAI API Integration

```python
"""
OpenAI API patterns and best practices.
"""
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, Optional, Generator
import asyncio

# Synchronous client
client = OpenAI()  # Uses OPENAI_API_KEY env var

def simple_completion(prompt: str) -> str:
    """Basic completion call."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


# Chat conversation
def chat_conversation(messages: List[Dict]) -> str:
    """Multi-turn conversation."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content


# Streaming responses
def stream_completion(prompt: str) -> Generator[str, None, None]:
    """Stream response tokens one by one."""
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Usage
for token in stream_completion("Write a short poem"):
    print(token, end="", flush=True)


# Async client for concurrent requests
async_client = AsyncOpenAI()

async def async_completion(prompt: str) -> str:
    """Async completion for concurrent requests."""
    response = await async_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

async def batch_completions(prompts: List[str]) -> List[str]:
    """Process multiple prompts concurrently."""
    tasks = [async_completion(p) for p in prompts]
    return await asyncio.gather(*tasks)


# Structured output (new)
from pydantic import BaseModel

class ExtractedInfo(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None

def extract_structured(text: str) -> ExtractedInfo:
    """Extract structured data with guaranteed schema."""
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract contact information."},
            {"role": "user", "content": text}
        ],
        response_format=ExtractedInfo
    )
    return response.choices[0].message.parsed


# Function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

def chat_with_tools(user_message: str) -> str:
    """Chat with function calling capability."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_message}],
        tools=tools
    )
    
    if response.choices[0].message.tool_calls:
        # Model wants to call a function
        tool_call = response.choices[0].message.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Execute function (implement get_weather)
        result = execute_function(function_name, arguments)
        
        # Send result back to model
        messages = [
            {"role": "user", "content": user_message},
            response.choices[0].message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            }
        ]
        
        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return final_response.choices[0].message.content
    
    return response.choices[0].message.content
```

### 2. Anthropic Claude Integration

```python
"""
Anthropic Claude API patterns.
"""
import anthropic
from typing import List, Dict, Generator

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY

def claude_completion(prompt: str, system: str = None) -> str:
    """Basic Claude completion."""
    kwargs = {
        "model": "claude-3-5-sonnet-20240620",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    if system:
        kwargs["system"] = system
    
    response = client.messages.create(**kwargs)
    return response.content[0].text


def claude_streaming(prompt: str) -> Generator[str, None, None]:
    """Stream Claude responses."""
    with client.messages.stream(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            yield text


# Claude with XML for structured output
def claude_with_xml(prompt: str) -> dict:
    """Use XML for structured output with Claude."""
    system = """You are a helpful assistant. 
When asked to extract information, respond in XML format."""
    
    response = claude_completion(f"""
<task>Extract contact information from this text</task>
<text>{prompt}</text>

<format>
<contact>
  <name>...</name>
  <email>...</email>
  <phone>...</phone>
</contact>
</format>

Respond with only the XML, no other text.""", system=system)
    
    # Parse XML response
    import xml.etree.ElementTree as ET
    root = ET.fromstring(response)
    
    return {
        "name": root.find("name").text,
        "email": root.find("email").text,
        "phone": root.find("phone").text if root.find("phone") is not None else None
    }


# Claude tool use
def claude_with_tools(user_message: str):
    """Claude with tool use."""
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=[{
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }],
        messages=[{"role": "user", "content": user_message}]
    )
    
    # Check if tool use is requested
    if response.stop_reason == "tool_use":
        tool_use = next(
            block for block in response.content 
            if block.type == "tool_use"
        )
        
        # Execute tool
        result = execute_tool(tool_use.name, tool_use.input)
        
        # Continue conversation with tool result
        return client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": json.dumps(result)
                    }]
                }
            ]
        )
    
    return response.content[0].text
```

### 3. Provider-Agnostic Abstraction

```python
"""
Build abstractions to switch providers easily.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, AsyncIterator
from dataclasses import dataclass
from enum import Enum

class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"

@dataclass
class Message:
    role: str  # "system", "user", "assistant"
    content: str

@dataclass
class CompletionResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str

class LLMClient(ABC):
    """Abstract LLM client interface."""
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> CompletionResponse:
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        pass


class OpenAIClient(LLMClient):
    """OpenAI implementation."""
    
    def __init__(self, model: str = "gpt-4o"):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI()
        self.model = model
    
    async def complete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            **kwargs
        )
        
        return CompletionResponse(
            content=response.choices[0].message.content,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            finish_reason=response.choices[0].finish_reason
        )
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        formatted = [{"role": m.role, "content": m.content} for m in messages]
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicClient(LLMClient):
    """Anthropic implementation."""
    
    def __init__(self, model: str = "claude-3-5-sonnet-20240620"):
        import anthropic
        self.client = anthropic.AsyncAnthropic()
        self.model = model
    
    async def complete(self, messages: List[Message], **kwargs) -> CompletionResponse:
        # Extract system message
        system = None
        chat_messages = []
        
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat_messages.append({"role": m.role, "content": m.content})
        
        response = await self.client.messages.create(
            model=self.model,
            system=system or "",
            messages=chat_messages,
            max_tokens=kwargs.get("max_tokens", 1024)
        )
        
        return CompletionResponse(
            content=response.content[0].text,
            model=response.model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            finish_reason=response.stop_reason
        )
    
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[str]:
        # Similar implementation with streaming
        ...


# Factory
def create_client(provider: Provider, model: str = None) -> LLMClient:
    """Create LLM client for given provider."""
    if provider == Provider.OPENAI:
        return OpenAIClient(model or "gpt-4o")
    elif provider == Provider.ANTHROPIC:
        return AnthropicClient(model or "claude-3-5-sonnet-20240620")
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### 4. Error Handling & Retries

```python
"""
Robust error handling for LLM APIs.
"""
import asyncio
from typing import TypeVar, Callable
from functools import wraps
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

class LLMError(Exception):
    """Base LLM error."""
    pass

class RateLimitError(LLMError):
    """Rate limit exceeded."""
    def __init__(self, retry_after: float = 60):
        self.retry_after = retry_after

class TokenLimitError(LLMError):
    """Token limit exceeded."""
    pass

class ContentFilterError(LLMError):
    """Content blocked by filters."""
    pass


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (RateLimitError, ConnectionError, TimeoutError)
):
    """Decorator for async retry with exponential backoff."""
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                
                except retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries reached: {e}")
                        raise
                    
                    # Get retry delay
                    if isinstance(e, RateLimitError):
                        wait = e.retry_after
                    else:
                        wait = min(delay, max_delay)
                        delay *= exponential_base
                    
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait:.1f}s..."
                    )
                    await asyncio.sleep(wait)
            
            raise last_exception
        
        return wrapper
    
    return decorator


# Usage with OpenAI
from openai import RateLimitError as OpenAIRateLimitError

@with_retry(max_retries=3)
async def safe_completion(prompt: str) -> str:
    """Completion with automatic retry."""
    try:
        response = await async_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    
    except OpenAIRateLimitError as e:
        # Convert to our exception
        raise RateLimitError(retry_after=60)


# Circuit breaker pattern
class CircuitBreaker:
    """Prevent cascading failures."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        
        if self.state == "open":
            if asyncio.get_event_loop().time() - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
            else:
                raise LLMError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
            
            return result
        
        except Exception as e:
            self.failures += 1
            self.last_failure_time = asyncio.get_event_loop().time()
            
            if self.failures >= self.failure_threshold:
                self.state = "open"
            
            raise


# Timeout handling
async def completion_with_timeout(prompt: str, timeout: float = 30.0) -> str:
    """Completion with timeout."""
    try:
        return await asyncio.wait_for(
            safe_completion(prompt),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        raise LLMError(f"Request timed out after {timeout}s")
```

### 5. Cost Management

```python
"""
Track and manage LLM costs.
"""
from dataclasses import dataclass
from typing import Dict
import tiktoken
from datetime import datetime

@dataclass
class PricingTier:
    input_per_1m: float
    output_per_1m: float

PRICING = {
    "gpt-4o": PricingTier(5.00, 15.00),
    "gpt-4o-mini": PricingTier(0.15, 0.60),
    "claude-3-5-sonnet": PricingTier(3.00, 15.00),
    "claude-3-5-haiku": PricingTier(0.80, 4.00),
}

class CostTracker:
    """Track LLM API costs."""
    
    def __init__(self):
        self.usage: Dict[str, Dict] = {}
        self.encoding = tiktoken.encoding_for_model("gpt-4o")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(self.encoding.encode(text))
    
    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ):
        """Record API usage."""
        if model not in self.usage:
            self.usage[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "requests": 0
            }
        
        self.usage[model]["prompt_tokens"] += prompt_tokens
        self.usage[model]["completion_tokens"] += completion_tokens
        self.usage[model]["requests"] += 1
    
    def get_costs(self) -> Dict[str, float]:
        """Calculate costs by model."""
        costs = {}
        
        for model, usage in self.usage.items():
            pricing = PRICING.get(model)
            if not pricing:
                continue
            
            input_cost = (usage["prompt_tokens"] * pricing.input_per_1m) / 1_000_000
            output_cost = (usage["completion_tokens"] * pricing.output_per_1m) / 1_000_000
            
            costs[model] = {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": input_cost + output_cost,
                "requests": usage["requests"]
            }
        
        return costs
    
    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        return sum(c["total_cost"] for c in self.get_costs().values())


# Budget management
class BudgetManager:
    """Manage API budget with limits."""
    
    def __init__(self, daily_budget: float = 100.0):
        self.daily_budget = daily_budget
        self.tracker = CostTracker()
        self.start_date = datetime.now().date()
    
    def check_budget(self) -> bool:
        """Check if within budget."""
        # Reset daily
        if datetime.now().date() != self.start_date:
            self.tracker = CostTracker()
            self.start_date = datetime.now().date()
        
        return self.tracker.get_total_cost() < self.daily_budget
    
    def record_and_check(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int
    ) -> bool:
        """Record usage and check budget."""
        self.tracker.record_usage(model, prompt_tokens, completion_tokens)
        
        if not self.check_budget():
            raise LLMError("Daily budget exceeded")
        
        return True


# Cost-aware model selection
def select_model_for_budget(
    prompt: str,
    required_capability: str,
    max_cost: float = 0.01
) -> str:
    """Select cheapest model that meets requirements."""
    tracker = CostTracker()
    
    prompt_tokens = tracker.estimate_tokens(prompt)
    estimated_output = prompt_tokens * 2  # Rough estimate
    
    # Check models from cheapest to most capable
    candidates = [
        ("gpt-4o-mini", ["basic", "classification"]),
        ("gpt-4o", ["complex", "reasoning", "code"]),
    ]
    
    for model, capabilities in candidates:
        pricing = PRICING.get(model)
        if not pricing:
            continue
        
        estimated_cost = (
            (prompt_tokens * pricing.input_per_1m +
             estimated_output * pricing.output_per_1m) / 1_000_000
        )
        
        if required_capability in capabilities and estimated_cost <= max_cost:
            return model
    
    # Fallback to cheapest
    return "gpt-4o-mini"
```

---

## 📊 API Comparison

| Feature | OpenAI | Anthropic | Google |
|---------|--------|-----------|--------|
| Streaming | ✅ | ✅ | ✅ |
| Function calling | ✅ | ✅ | ✅ |
| Vision | ✅ | ✅ | ✅ |
| Structured output | ✅ (native) | XML pattern | JSON schema |
| Max context | 128K | 200K | 1M |

---

## ➡️ Next Steps

Continue to **[02_structured_outputs.md](./02_structured_outputs.md)** for reliable output parsing.
