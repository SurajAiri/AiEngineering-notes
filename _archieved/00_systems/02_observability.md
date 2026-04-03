# 📊 Observability for AI Systems

## 📚 Overview

AI systems need comprehensive observability: logging, metrics, and tracing. This module covers production monitoring patterns for AI/ML workloads.

---

## 🎯 Learning Objectives

- Implement **structured logging** for AI pipelines
- Design **metrics** for LLM and RAG systems
- Build **distributed tracing** across services
- Create **dashboards** for AI workloads
- Set up **alerting** for production issues

---

## 🔬 Core Concepts

### 1. Structured Logging

```python
"""
Structured logging: JSON logs for easy parsing.
Essential for: Log aggregation, debugging, compliance
"""
import logging
import json
from datetime import datetime
from typing import Any, Dict
import sys

class StructuredFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def setup_logging(level: str = "INFO"):
    """Configure structured logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)
    
    return root

# Logger with context
class ContextLogger:
    """Logger with automatic context injection."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Set persistent context for all log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear context."""
        self.context = {}
    
    def _log(self, level: int, message: str, **kwargs):
        extra = {**self.context, **kwargs}
        self.logger.log(level, message, extra={"extra": extra})
    
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

# Usage
logger = ContextLogger("rag_service")
logger.set_context(service="rag", version="1.0.0")
logger.info("Query received", query="What is RAG?", user_id="123")

# Output:
# {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "message": "Query received",
#  "query": "What is RAG?", "user_id": "123", "service": "rag", "version": "1.0.0"}
```

### 2. Metrics for AI Systems

```python
"""
Prometheus metrics for AI/ML workloads.
"""
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from functools import wraps
import time

# Key metrics for RAG systems
class RAGMetrics:
    """Prometheus metrics for RAG pipeline."""
    
    def __init__(self, prefix: str = "rag"):
        # Counters
        self.queries_total = Counter(
            f"{prefix}_queries_total",
            "Total number of queries",
            ["status"]  # success, error
        )
        
        self.retrieval_total = Counter(
            f"{prefix}_retrieval_total",
            "Total retrieval operations",
            ["source"]  # vector, bm25, hybrid
        )
        
        # Histograms for latency
        self.query_latency = Histogram(
            f"{prefix}_query_latency_seconds",
            "Query end-to-end latency",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.retrieval_latency = Histogram(
            f"{prefix}_retrieval_latency_seconds",
            "Retrieval latency",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
        )
        
        self.llm_latency = Histogram(
            f"{prefix}_llm_latency_seconds",
            "LLM generation latency",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        # Histograms for tokens
        self.tokens_used = Histogram(
            f"{prefix}_tokens_used",
            "Tokens used per query",
            ["type"],  # prompt, completion
            buckets=[100, 500, 1000, 2000, 4000, 8000]
        )
        
        # Gauges for current state
        self.documents_indexed = Gauge(
            f"{prefix}_documents_indexed",
            "Total documents in index"
        )
        
        self.active_queries = Gauge(
            f"{prefix}_active_queries",
            "Currently processing queries"
        )
        
        # Retrieval quality
        self.context_relevance = Histogram(
            f"{prefix}_context_relevance",
            "Context relevance score",
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )

metrics = RAGMetrics()

def observe_latency(histogram):
    """Decorator to measure function latency."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                histogram.observe(time.perf_counter() - start)
        return wrapper
    return decorator

# Usage
@observe_latency(metrics.retrieval_latency)
async def retrieve_documents(query: str):
    # Retrieval logic
    pass

@observe_latency(metrics.llm_latency)
async def generate_answer(context: str, query: str):
    # Generation logic
    pass


# LLM-specific metrics
class LLMMetrics:
    """Metrics for LLM usage and performance."""
    
    def __init__(self, prefix: str = "llm"):
        self.requests_total = Counter(
            f"{prefix}_requests_total",
            "Total LLM requests",
            ["model", "status"]
        )
        
        self.tokens_total = Counter(
            f"{prefix}_tokens_total",
            "Total tokens used",
            ["model", "type"]  # prompt, completion
        )
        
        self.cost_total = Counter(
            f"{prefix}_cost_dollars_total",
            "Estimated cost in dollars",
            ["model"]
        )
        
        self.latency = Histogram(
            f"{prefix}_latency_seconds",
            "LLM response latency",
            ["model"],
            buckets=[0.5, 1, 2, 5, 10, 30]
        )
        
        self.cache_hits = Counter(
            f"{prefix}_cache_hits_total",
            "Cache hits"
        )
        
        self.cache_misses = Counter(
            f"{prefix}_cache_misses_total",
            "Cache misses"
        )
    
    def record_request(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency: float,
        success: bool
    ):
        """Record a complete LLM request."""
        status = "success" if success else "error"
        
        self.requests_total.labels(model=model, status=status).inc()
        self.tokens_total.labels(model=model, type="prompt").inc(prompt_tokens)
        self.tokens_total.labels(model=model, type="completion").inc(completion_tokens)
        self.latency.labels(model=model).observe(latency)
        
        # Estimate cost (example rates)
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
        self.cost_total.labels(model=model).inc(cost)
    
    def _estimate_cost(self, model: str, prompt: int, completion: int) -> float:
        """Estimate cost based on model."""
        rates = {
            "gpt-4o": (0.005, 0.015),  # per 1K tokens (prompt, completion)
            "gpt-4o-mini": (0.00015, 0.0006),
            "claude-3-opus": (0.015, 0.075),
            "claude-3-sonnet": (0.003, 0.015)
        }
        
        if model not in rates:
            return 0
        
        prompt_rate, completion_rate = rates[model]
        return (prompt * prompt_rate + completion * completion_rate) / 1000
```

### 3. Distributed Tracing

```python
"""
Distributed tracing with OpenTelemetry.
Essential for: Debugging latency, understanding flow
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from functools import wraps

# Setup tracing
def setup_tracing(service_name: str = "rag-service"):
    """Configure OpenTelemetry tracing."""
    provider = TracerProvider()
    
    # Export to OTLP collector (Jaeger, Tempo, etc.)
    exporter = OTLPSpanExporter(endpoint="http://localhost:4317")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    
    trace.set_tracer_provider(provider)
    
    # Auto-instrument libraries
    FastAPIInstrumentor.instrument()
    HTTPXClientInstrumentor.instrument()
    
    return trace.get_tracer(service_name)

tracer = setup_tracing("rag-service")

# Custom span decorator
def traced(span_name: str = None):
    """Decorator to trace function execution."""
    def decorator(func):
        name = span_name or func.__name__
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(name) as span:
                # Add function arguments as attributes
                span.set_attribute("function", func.__name__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(trace.StatusCode.OK)
                    return result
                except Exception as e:
                    span.set_status(trace.StatusCode.ERROR)
                    span.record_exception(e)
                    raise
        
        return wrapper
    return decorator


# Full RAG pipeline with tracing
class TracedRAGPipeline:
    """RAG pipeline with distributed tracing."""
    
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    @traced("rag.query")
    async def query(self, question: str) -> dict:
        """Full RAG query with tracing."""
        span = trace.get_current_span()
        span.set_attribute("query", question[:100])  # Truncate for safety
        
        # Retrieval trace
        with tracer.start_span("rag.retrieve") as retrieval_span:
            docs = await self.retriever.search(question)
            retrieval_span.set_attribute("n_docs", len(docs))
        
        # Reranking trace
        with tracer.start_span("rag.rerank") as rerank_span:
            reranked = await self.rerank(question, docs)
            rerank_span.set_attribute("n_reranked", len(reranked))
        
        # Context assembly trace
        with tracer.start_span("rag.context_assembly") as context_span:
            context = self.assemble_context(reranked)
            context_span.set_attribute("context_tokens", len(context.split()))
        
        # LLM generation trace
        with tracer.start_span("rag.generate") as gen_span:
            answer = await self.llm.generate(context, question)
            gen_span.set_attribute("answer_length", len(answer))
        
        span.set_attribute("success", True)
        
        return {"answer": answer, "sources": reranked}


# LangSmith tracing (alternative)
"""
LangSmith provides specialized LLM tracing.

from langsmith import Client, traceable

@traceable(name="rag_query")
def rag_query(question: str):
    # Automatically traces all LangChain operations
    ...
"""


# LangFuse tracing (alternative)
"""
LangFuse is open-source LLM observability.

from langfuse.decorators import observe

@observe()
def rag_query(question: str):
    ...
"""
```

### 4. Dashboards & Alerting

```python
"""
Dashboard metrics and alerting rules for AI systems.
"""

# Prometheus alert rules (YAML)
PROMETHEUS_ALERTS = """
groups:
  - name: rag_alerts
    rules:
      # High latency alert
      - alert: RAGHighLatency
        expr: histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m])) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RAG query latency is high"
          description: "P95 latency is {{ $value }}s"
      
      # High error rate
      - alert: RAGHighErrorRate
        expr: rate(rag_queries_total{status="error"}[5m]) / rate(rag_queries_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "RAG error rate above 5%"
      
      # LLM cost spike
      - alert: LLMCostSpike
        expr: rate(llm_cost_dollars_total[1h]) > 10
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "LLM cost exceeds $10/hour"
      
      # Low retrieval quality
      - alert: LowRetrievalQuality
        expr: histogram_quantile(0.50, rag_context_relevance_bucket) < 0.5
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Retrieval quality is degrading"
"""

# Grafana dashboard JSON (simplified)
GRAFANA_DASHBOARD = {
    "title": "RAG Service Dashboard",
    "panels": [
        {
            "title": "Query Rate",
            "type": "graph",
            "targets": [{
                "expr": "rate(rag_queries_total[5m])"
            }]
        },
        {
            "title": "P95 Latency",
            "type": "graph",
            "targets": [{
                "expr": "histogram_quantile(0.95, rate(rag_query_latency_seconds_bucket[5m]))"
            }]
        },
        {
            "title": "Error Rate",
            "type": "gauge",
            "targets": [{
                "expr": "rate(rag_queries_total{status='error'}[5m]) / rate(rag_queries_total[5m])"
            }]
        },
        {
            "title": "Token Usage",
            "type": "graph",
            "targets": [
                {"expr": "rate(llm_tokens_total{type='prompt'}[5m])", "legendFormat": "Prompt"},
                {"expr": "rate(llm_tokens_total{type='completion'}[5m])", "legendFormat": "Completion"}
            ]
        },
        {
            "title": "Hourly Cost",
            "type": "stat",
            "targets": [{
                "expr": "increase(llm_cost_dollars_total[1h])"
            }]
        }
    ]
}


# Health check endpoint
from fastapi import FastAPI
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthChecker:
    """Check health of RAG components."""
    
    def __init__(self):
        self.checks = {}
    
    def add_check(self, name: str, check_func):
        self.checks[name] = check_func
    
    async def run_checks(self) -> dict:
        results = {}
        overall = HealthStatus.HEALTHY
        
        for name, check in self.checks.items():
            try:
                status = await check()
                results[name] = {"status": status.value}
                
                if status == HealthStatus.UNHEALTHY:
                    overall = HealthStatus.UNHEALTHY
                elif status == HealthStatus.DEGRADED and overall != HealthStatus.UNHEALTHY:
                    overall = HealthStatus.DEGRADED
            
            except Exception as e:
                results[name] = {"status": "error", "error": str(e)}
                overall = HealthStatus.UNHEALTHY
        
        return {
            "status": overall.value,
            "checks": results
        }

# Usage
app = FastAPI()
health_checker = HealthChecker()

async def check_vector_db():
    # Try to ping vector DB
    return HealthStatus.HEALTHY

async def check_llm():
    # Try simple completion
    return HealthStatus.HEALTHY

health_checker.add_check("vector_db", check_vector_db)
health_checker.add_check("llm", check_llm)

@app.get("/health")
async def health():
    return await health_checker.run_checks()
```

---

## 📊 Key Metrics Summary

| Category | Metric | Purpose |
|----------|--------|---------|
| **Latency** | P50, P95, P99 | User experience |
| **Throughput** | Queries/sec | Capacity planning |
| **Errors** | Error rate, types | Reliability |
| **Tokens** | Prompt, completion | Cost tracking |
| **Quality** | Relevance scores | Model performance |
| **Resources** | CPU, memory, GPU | Infrastructure |

---

## ➡️ Next Steps

Continue to **[03_data_engineering.md](./03_data_engineering.md)** for data pipeline patterns.
