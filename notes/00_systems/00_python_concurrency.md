# 🐍 Python & Concurrency for AI Engineering

## 📚 Overview

Production AI systems require robust, concurrent Python code. This module covers essential Python patterns for building reliable AI pipelines.

---

## 🎯 Learning Objectives

- Master **asyncio** for async I/O
- Design **producer-consumer** systems
- Choose **threads vs processes** correctly
- Handle **backpressure** and queue overflow
- Build **frame-based streaming** pipelines

---

## 🔬 Core Concepts

### 1. Asyncio Fundamentals

```python
"""
Asyncio: Cooperative multitasking for I/O-bound operations.
Perfect for: API calls, database queries, file I/O
Not for: CPU-intensive work (use multiprocessing instead)
"""
import asyncio
from typing import List

# Basic coroutine
async def fetch_embedding(text: str) -> List[float]:
    """Simulate async API call for embeddings."""
    await asyncio.sleep(0.1)  # Simulates network I/O
    return [0.1, 0.2, 0.3]

# Running coroutines concurrently
async def fetch_many_embeddings(texts: List[str]) -> List[List[float]]:
    """Fetch embeddings concurrently - much faster than sequential!"""
    tasks = [fetch_embedding(text) for text in texts]
    return await asyncio.gather(*tasks)

# With error handling
async def fetch_with_retry(text: str, max_retries: int = 3) -> List[float]:
    """Async with retry logic."""
    for attempt in range(max_retries):
        try:
            return await fetch_embedding(text)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)  # Exponential backoff

# Semaphore for rate limiting
async def rate_limited_fetch(texts: List[str], max_concurrent: int = 10):
    """Limit concurrent requests to avoid overwhelming APIs."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def bounded_fetch(text: str):
        async with semaphore:
            return await fetch_embedding(text)
    
    return await asyncio.gather(*[bounded_fetch(t) for t in texts])
```

### 2. Producer-Consumer Pattern

```python
"""
Producer-Consumer: Decouple data production from processing.
Essential for: ETL pipelines, streaming data, queue-based systems
"""
import asyncio
from asyncio import Queue
from typing import Any

class ProducerConsumerPipeline:
    """
    Classic pattern for async data processing.
    
    Producer → Queue → Consumer(s)
    """
    
    def __init__(self, queue_size: int = 100):
        self.queue: Queue = asyncio.Queue(maxsize=queue_size)
        self.running = True
    
    async def producer(self, items: List[Any]):
        """Put items into the queue."""
        for item in items:
            await self.queue.put(item)
        
        # Signal end with sentinel value
        await self.queue.put(None)
    
    async def consumer(self, consumer_id: int):
        """Process items from the queue."""
        while True:
            item = await self.queue.get()
            
            if item is None:
                # Pass sentinel to next consumer
                await self.queue.put(None)
                break
            
            # Process item
            result = await self.process(item)
            print(f"Consumer {consumer_id}: processed {item}")
            
            self.queue.task_done()
    
    async def process(self, item: Any) -> Any:
        """Override this for actual processing."""
        await asyncio.sleep(0.1)
        return item
    
    async def run(self, items: List[Any], n_consumers: int = 3):
        """Run the pipeline."""
        producer = asyncio.create_task(self.producer(items))
        consumers = [
            asyncio.create_task(self.consumer(i))
            for i in range(n_consumers)
        ]
        
        await producer
        await asyncio.gather(*consumers)


# Streaming RAG pipeline example
class StreamingRAGPipeline:
    """
    Producer-Consumer for RAG data ingestion.
    
    Document Producer → [Chunking] → [Embedding] → [Indexing]
    """
    
    def __init__(self):
        self.chunk_queue = asyncio.Queue(maxsize=1000)
        self.embed_queue = asyncio.Queue(maxsize=100)
    
    async def document_producer(self, doc_paths: List[str]):
        """Load documents and put into chunk queue."""
        for path in doc_paths:
            doc = await self.load_document(path)
            await self.chunk_queue.put(doc)
        await self.chunk_queue.put(None)  # Sentinel
    
    async def chunker(self):
        """Take documents, produce chunks."""
        while True:
            doc = await self.chunk_queue.get()
            if doc is None:
                await self.embed_queue.put(None)
                break
            
            chunks = self.chunk_document(doc)
            for chunk in chunks:
                await self.embed_queue.put(chunk)
    
    async def embedder_and_indexer(self, batch_size: int = 32):
        """Batch chunks, embed, and index."""
        batch = []
        
        while True:
            chunk = await self.embed_queue.get()
            
            if chunk is None:
                if batch:
                    await self.process_batch(batch)
                break
            
            batch.append(chunk)
            
            if len(batch) >= batch_size:
                await self.process_batch(batch)
                batch = []
    
    async def process_batch(self, chunks: List[str]):
        """Embed and index a batch of chunks."""
        embeddings = await self.embed_texts(chunks)
        await self.index_embeddings(chunks, embeddings)
```

### 3. Threads vs Processes

```python
"""
When to use what:
- threading: I/O-bound (waiting for network/disk)
- multiprocessing: CPU-bound (computation)
- asyncio: Many I/O operations (best for most AI workloads)
"""
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# ThreadPoolExecutor for I/O-bound work
def fetch_sync(url: str) -> str:
    """Sync HTTP request (use in thread pool)."""
    import requests
    return requests.get(url).text

def parallel_fetch_threading(urls: List[str], max_workers: int = 10) -> List[str]:
    """Use threads for I/O-bound parallel work."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(fetch_sync, urls))
    return results


# ProcessPoolExecutor for CPU-bound work
def compute_embedding_sync(text: str) -> List[float]:
    """CPU-intensive embedding (local model)."""
    # Heavy computation here
    return [0.1] * 768

def parallel_embed_multiprocessing(texts: List[str], max_workers: int = 4) -> List[List[float]]:
    """Use processes for CPU-bound parallel work."""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(compute_embedding_sync, texts))
    return results


# Mixing async with threads for sync libraries
async def run_sync_in_thread(func, *args):
    """Run sync function in thread pool (for async context)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

# Example: Use sync library in async code
async def fetch_with_sync_library(url: str):
    """Use requests (sync) in async context."""
    import requests
    return await run_sync_in_thread(requests.get, url)
```

### 4. Backpressure Handling

```python
"""
Backpressure: What happens when producer is faster than consumer?

Without handling: Memory grows unbounded → OOM crash
With handling: Slow producer or drop items gracefully
"""
import asyncio
from dataclasses import dataclass
from enum import Enum

class OverflowPolicy(Enum):
    BLOCK = "block"      # Wait for space (default Queue behavior)
    DROP_OLDEST = "drop_oldest"  # Evict oldest when full
    DROP_NEWEST = "drop_newest"  # Reject new items when full

class BackpressureQueue:
    """Queue with configurable overflow policy."""
    
    def __init__(self, maxsize: int = 100, policy: OverflowPolicy = OverflowPolicy.BLOCK):
        self.queue = asyncio.Queue(maxsize=maxsize)
        self.policy = policy
        self.dropped_count = 0
    
    async def put(self, item):
        """Put with backpressure handling."""
        if self.policy == OverflowPolicy.BLOCK:
            await self.queue.put(item)
        
        elif self.policy == OverflowPolicy.DROP_NEWEST:
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                self.dropped_count += 1
        
        elif self.policy == OverflowPolicy.DROP_OLDEST:
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.dropped_count += 1
                except asyncio.QueueEmpty:
                    pass
            await self.queue.put(item)
    
    async def get(self):
        return await self.queue.get()


# Rate-based backpressure
class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: float, burst: int = 1):
        """
        Args:
            rate: Tokens per second
            burst: Maximum tokens to accumulate
        """
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = asyncio.get_event_loop().time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until a token is available."""
        async with self.lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self.last_update
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            self.last_update = now
            
            if self.tokens < 1:
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= 1
```

### 5. Frame-Based Streaming

```python
"""
Frame-based streaming: Process data in fixed-size chunks.
Essential for: Audio/video processing, real-time AI pipelines
"""
import asyncio
from typing import AsyncIterator, List
import numpy as np

@dataclass
class AudioFrame:
    """Fixed-size audio frame."""
    data: np.ndarray
    sample_rate: int
    timestamp: float

class StreamingProcessor:
    """
    Process streaming audio in frames.
    Core pattern for real-time AI (ASR, TTS, voice agents).
    """
    
    def __init__(self, frame_size: int = 480, sample_rate: int = 16000):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.buffer = np.array([], dtype=np.float32)
    
    def add_samples(self, samples: np.ndarray) -> List[AudioFrame]:
        """Add samples and return complete frames."""
        self.buffer = np.concatenate([self.buffer, samples])
        
        frames = []
        while len(self.buffer) >= self.frame_size:
            frame_data = self.buffer[:self.frame_size]
            self.buffer = self.buffer[self.frame_size:]
            
            frames.append(AudioFrame(
                data=frame_data,
                sample_rate=self.sample_rate,
                timestamp=len(frames) * self.frame_size / self.sample_rate
            ))
        
        return frames
    
    async def process_stream(
        self, 
        audio_stream: AsyncIterator[np.ndarray]
    ) -> AsyncIterator[str]:
        """Process audio stream, yield transcriptions."""
        async for chunk in audio_stream:
            frames = self.add_samples(chunk)
            
            for frame in frames:
                # Process each frame (e.g., VAD, ASR)
                result = await self.process_frame(frame)
                if result:
                    yield result
    
    async def process_frame(self, frame: AudioFrame) -> str:
        """Override for actual processing."""
        await asyncio.sleep(0.01)  # Simulate processing
        return ""


# Pipeline with multiple stages
class MultiStageStreamPipeline:
    """
    Stream through multiple processing stages.
    
    Audio → VAD → ASR → LLM → TTS → Audio Out
    """
    
    def __init__(self):
        self.stages = []
    
    def add_stage(self, stage):
        """Add processing stage."""
        self.stages.append(stage)
        return self
    
    async def process(self, input_stream: AsyncIterator):
        """Run data through all stages."""
        current = input_stream
        
        for stage in self.stages:
            current = stage.process(current)
        
        async for output in current:
            yield output
```

---

## 🚨 Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **GIL blocking** | CPU work blocks async | Use ProcessPoolExecutor |
| **Unbounded queues** | Memory leak, OOM | Set maxsize, handle backpressure |
| **Missing await** | Coroutine never runs | Always await coroutines |
| **Sync in async** | Event loop blocked | Use run_in_executor |
| **Deadlocks** | Program hangs | Avoid circular waits, use timeouts |

---

## 📖 Resources

- **asyncio docs** - Official Python documentation
- **trio** - Alternative async library (stricter)
- **uvloop** - Fast asyncio event loop

---

## ➡️ Next Steps

Continue to **[01_api_design.md](./01_api_design.md)** for async API patterns.
