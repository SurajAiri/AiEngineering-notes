# 🌐 API Design for AI Systems

## 📚 Overview

Well-designed APIs are critical for AI systems. This module covers REST, async patterns, streaming, WebSockets, and error handling for production AI services.

---

## 🎯 Learning Objectives

- Design **RESTful APIs** for AI services
- Implement **async endpoints** with FastAPI
- Build **streaming responses** (SSE, WebSockets)
- Handle **errors gracefully** with retry logic
- Apply **rate limiting** and authentication

---

## 🔬 Core Concepts

### 1. FastAPI for AI Services

```python
"""
FastAPI: Modern, fast, async Python web framework.
Perfect for AI services due to:
- Native async support
- Automatic OpenAPI docs
- Type validation with Pydantic
- Streaming responses
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import asyncio

app = FastAPI(
    title="RAG API",
    description="Production RAG service API",
    version="1.0.0"
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., description="Search query", min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    filters: Optional[dict] = Field(default=None, description="Metadata filters")

class Source(BaseModel):
    content: str
    metadata: dict
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    latency_ms: float
    tokens_used: int

# Sync endpoint (simple)
@app.post("/query", response_model=QueryResponse)
def query_sync(request: QueryRequest):
    """Synchronous RAG query endpoint."""
    # Process query...
    return QueryResponse(
        answer="Example answer",
        sources=[],
        latency_ms=100.0,
        tokens_used=150
    )

# Async endpoint (preferred for I/O-bound)
@app.post("/query/async", response_model=QueryResponse)
async def query_async(request: QueryRequest):
    """Async RAG query - better for concurrent requests."""
    # Async retrieval
    sources = await retrieve_documents(request.query, request.top_k)
    
    # Async generation
    answer = await generate_answer(request.query, sources)
    
    return QueryResponse(
        answer=answer,
        sources=sources,
        latency_ms=150.0,
        tokens_used=200
    )

async def retrieve_documents(query: str, top_k: int) -> List[Source]:
    """Async document retrieval."""
    await asyncio.sleep(0.1)  # Simulate DB call
    return []

async def generate_answer(query: str, sources: List[Source]) -> str:
    """Async LLM generation."""
    await asyncio.sleep(0.5)  # Simulate LLM call
    return "Generated answer"


# Background tasks for async processing
@app.post("/ingest")
async def ingest_documents(
    documents: List[str],
    background_tasks: BackgroundTasks
):
    """Start document ingestion in background."""
    task_id = "task_123"
    background_tasks.add_task(ingest_in_background, documents, task_id)
    return {"task_id": task_id, "status": "started"}

async def ingest_in_background(documents: List[str], task_id: str):
    """Background ingestion task."""
    for doc in documents:
        await process_document(doc)
    # Update task status in database
```

### 2. Streaming Responses (SSE)

```python
"""
Server-Sent Events (SSE): Stream text as it's generated.
Essential for: ChatGPT-like streaming, real-time updates
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator
import json
import asyncio

app = FastAPI()

class StreamQueryRequest(BaseModel):
    query: str
    stream: bool = True

async def stream_llm_response(query: str) -> AsyncGenerator[str, None]:
    """Generate streaming response chunks."""
    # Simulate LLM streaming
    response_text = "This is a streaming response that appears word by word."
    
    for word in response_text.split():
        yield f"data: {json.dumps({'chunk': word + ' '})}\n\n"
        await asyncio.sleep(0.1)
    
    # Final event
    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/query/stream")
async def query_stream(request: StreamQueryRequest):
    """Streaming RAG endpoint (SSE)."""
    if not request.stream:
        # Regular response
        return {"answer": "Full response here"}
    
    return StreamingResponse(
        stream_llm_response(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# Client-side consumption (JavaScript)
"""
const eventSource = new EventSource('/query/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query: 'What is RAG?' })
});

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.done) {
        eventSource.close();
    } else {
        console.log(data.chunk);
    }
};
"""


# OpenAI-compatible streaming format
async def stream_openai_format(query: str) -> AsyncGenerator[str, None]:
    """Stream in OpenAI API format for compatibility."""
    response_text = "This is a streaming response"
    
    for i, word in enumerate(response_text.split()):
        chunk = {
            "id": "chatcmpl-123",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "custom-rag",
            "choices": [{
                "index": 0,
                "delta": {"content": word + " "},
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        await asyncio.sleep(0.05)
    
    # Final chunk
    final = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"
```

### 3. WebSocket for Real-time

```python
"""
WebSocket: Bidirectional real-time communication.
Essential for: Chat applications, voice agents, live updates
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, List
import json
import asyncio

app = FastAPI()

class ConnectionManager:
    """Manage active WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time chat."""
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data["type"] == "query":
                # Process RAG query
                query = data["query"]
                
                # Send acknowledgment
                await manager.send_message(client_id, {
                    "type": "ack",
                    "message": "Processing query..."
                })
                
                # Stream response chunks
                async for chunk in process_query_stream(query):
                    await manager.send_message(client_id, {
                        "type": "chunk",
                        "content": chunk
                    })
                
                # Send completion
                await manager.send_message(client_id, {
                    "type": "complete"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)

async def process_query_stream(query: str):
    """Process query and yield response chunks."""
    words = "This is a streaming response to your query".split()
    for word in words:
        await asyncio.sleep(0.1)
        yield word + " "
```

### 4. Error Handling & Retries

```python
"""
Robust error handling for AI services.
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# Custom exceptions
class RAGException(Exception):
    """Base exception for RAG service."""
    def __init__(self, message: str, error_code: str, status_code: int = 500):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code

class RetrievalError(RAGException):
    """Error during document retrieval."""
    def __init__(self, message: str):
        super().__init__(message, "RETRIEVAL_ERROR", 500)

class GenerationError(RAGException):
    """Error during LLM generation."""
    def __init__(self, message: str):
        super().__init__(message, "GENERATION_ERROR", 500)

class RateLimitError(RAGException):
    """Rate limit exceeded."""
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, "RATE_LIMIT", 429)

# Error response model
class ErrorResponse(BaseModel):
    error: str
    error_code: str
    details: Optional[dict] = None

# Global exception handler
@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    logger.error(f"RAG Exception: {exc.error_code} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "error_code": exc.error_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }
    )


# Retry decorator
def retry_async(max_retries: int = 3, backoff_factor: float = 1.0):
    """Decorator for async retry with exponential backoff."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    wait_time = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
            
            raise last_exception
        
        return wrapper
    return decorator

@retry_async(max_retries=3, backoff_factor=0.5)
async def call_llm_with_retry(prompt: str) -> str:
    """LLM call with automatic retry."""
    # May raise exceptions - will be retried
    return await llm_client.generate(prompt)


# Circuit breaker pattern
class CircuitBreaker:
    """
    Circuit breaker to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failing, reject requests immediately
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"
    
    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        if self.state == "OPEN":
            if asyncio.get_event_loop().time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise RAGException("Service unavailable", "CIRCUIT_OPEN", 503)
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = asyncio.get_event_loop().time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            
            raise

# Usage
llm_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

@app.post("/query/safe")
async def query_with_circuit_breaker(request: QueryRequest):
    """Query with circuit breaker protection."""
    result = await llm_circuit.call(call_llm, request.query)
    return {"answer": result}
```

### 5. Rate Limiting & Authentication

```python
"""
Rate limiting and authentication for production APIs.
"""
from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.security import APIKeyHeader
from typing import Optional
import time
from collections import defaultdict

app = FastAPI()

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

VALID_API_KEYS = {
    "key_123": {"user": "alice", "tier": "pro", "rate_limit": 100},
    "key_456": {"user": "bob", "tier": "free", "rate_limit": 10}
}

async def verify_api_key(api_key: Optional[str] = Depends(api_key_header)):
    if api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    
    if api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return VALID_API_KEYS[api_key]


# In-memory rate limiter (use Redis for production)
class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self):
        self.buckets = defaultdict(lambda: {"tokens": 0, "last_update": time.time()})
    
    def is_allowed(self, key: str, rate_limit: int, window: int = 60) -> bool:
        bucket = self.buckets[key]
        now = time.time()
        
        # Refill tokens
        elapsed = now - bucket["last_update"]
        bucket["tokens"] = min(rate_limit, bucket["tokens"] + elapsed * (rate_limit / window))
        bucket["last_update"] = now
        
        # Check if token available
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return True
        
        return False

rate_limiter = RateLimiter()

async def check_rate_limit(user: dict = Depends(verify_api_key)):
    if not rate_limiter.is_allowed(user["user"], user["rate_limit"]):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )
    return user

@app.post("/query/protected")
async def protected_query(
    request: QueryRequest,
    user: dict = Depends(check_rate_limit)
):
    """Rate-limited and authenticated endpoint."""
    return {
        "answer": "Protected response",
        "user": user["user"]
    }
```

---

## 🚨 Best Practices

| Practice | Why |
|----------|-----|
| **Use async everywhere** | Better throughput for I/O-bound ops |
| **Stream when possible** | Better UX for long-running ops |
| **Validate inputs** | Prevent injection, bound resources |
| **Handle errors gracefully** | Don't expose internals |
| **Add timeouts** | Prevent hanging requests |
| **Rate limit everything** | Protect from abuse |

---

## 📖 Resources

- **FastAPI docs** - Official documentation
- **Starlette** - Underlying ASGI framework
- **httpx** - Async HTTP client

---

## ➡️ Next Steps

Continue to **[02_observability.md](./02_observability.md)** for logging and monitoring.
