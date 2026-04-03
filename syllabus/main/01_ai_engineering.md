# AI Engineering — Core Roadmap

> **This file is the job.** Every skill here maps directly to what hiring managers test for.  
> **Libraries are listed per stage** — know the ecosystem, not just the concepts.  
> **Foundations** (Python, SQL, Docker, Cloud) live in `00_foundations.md`. Finish that first.

---

## Progress Overview

| Stage | Topic                  | Time Estimate | Status         |
| ----- | ---------------------- | ------------- | -------------- |
| 1     | LLM Fundamentals       | 3–4 weeks     | ⬜ Not started |
| 2     | RAG — Build Phase      | 6–8 weeks     | ⬜ Not started |
| 3     | RAG — Evaluate Phase   | 2–3 weeks     | ⬜ Not started |
| 4     | Agents                 | 5–6 weeks     | ⬜ Not started |
| 5     | Memory Systems         | 3–4 weeks     | ⬜ Not started |
| 6     | Production & LLMOps    | 4–5 weeks     | ⬜ Not started |
| 7     | Differentiators        | Ongoing       | ⬜ Not started |

> ⬜ Not started / 🔄 In progress / ✅ Done

**Portfolio rule:** Every checkpoint = a GitHub repo with a clear README. Hiring managers check GitHub before your resume.

---

## ★ Milestone: Junior AI Engineer (After Stage 3)
## ◆ Milestone: Mid-level AI Engineer (After Stage 5)

---

## STAGE 1 — LLM Fundamentals

> **Goal:** Understand how LLMs behave and reliably control their output.  
> **Mindset:** You don't need to understand the math. You need to understand the _behavior_.

**Key libraries for this stage:**
| Library | What it does | When you use it |
|---|---|---|
| `openai` | Official OpenAI Python SDK | Calling GPT-4o, embeddings, function calling |
| `anthropic` | Official Anthropic Python SDK | Calling Claude models |
| `litellm` | Unified interface for 100+ LLM providers | Switch between OpenAI, Anthropic, Bedrock, etc. with one line |
| `instructor` | Typed, validated LLM outputs using Pydantic | When you need structured JSON from any LLM reliably |
| `pydantic` | Data validation and typed models | Everywhere — the backbone of structured AI outputs |
| `tiktoken` | OpenAI tokenizer | Counting tokens before making API calls |

### 1.1 How LLMs Work (Conceptual)

- [ ] Explain tokenization and how it affects cost and behavior
- [ ] Explain context windows and what happens when you exceed them
- [ ] Explain KV cache and why it matters for latency and cost
- [ ] Explain why hallucinations happen (training distribution, not lying)
- [ ] Explain the difference between system, user, and assistant messages

### 1.2 Inference & Sampling

- [ ] Understand temperature, top-k, and top-p — when to use each
- [ ] Know when to use deterministic (temp=0) vs creative (temp>0) settings
- [ ] Understand streaming vs non-streaming and when each is appropriate
- [ ] Know what stop sequences are and how to use them

### 1.3 Working with LLM APIs

- [ ] Call OpenAI and Anthropic APIs confidently
- [ ] Handle streaming responses and display them progressively
- [ ] Use **LiteLLM** to call any provider with a unified interface — this is production best practice
- [ ] Implement retry logic with exponential backoff
- [ ] Handle API errors gracefully (rate limits, timeouts, model errors)
- [ ] Track and log token usage per call

### 1.4 Structured Outputs & Function Calling ⚡

- [ ] Use JSON mode / structured output APIs reliably
- [ ] Write function/tool schemas (OpenAI and Anthropic formats)
- [ ] Validate structured outputs with **Pydantic**
- [ ] Use **instructor** for typed LLM outputs — this eliminates most parsing headaches
- [ ] Handle malformed structured output and retry gracefully
- [ ] Understand constrained decoding (what it is, why it helps)

### 1.5 Prompt Engineering

- [ ] Write clear, unambiguous system prompts
- [ ] Use few-shot examples effectively
- [ ] Apply chain-of-thought prompting for complex tasks
- [ ] Use prompt chaining (multi-step pipelines)
- [ ] Version and track prompts — treat them like code
- [ ] Test prompts on edge cases, not just happy paths

### 1.6 Context Engineering ⚡

> The evolution beyond prompt engineering. The best AI engineers design the entire information environment the model sees, not just the user prompt.

- [ ] Understand that context = system prompt + retrieved docs + tool schemas + history + few-shot examples — all competing for the same token budget
- [ ] Know the "lost in the middle" problem: models attend better to start and end of context
- [ ] Design context that degrades gracefully when components are missing
- [ ] Treat context assembly as engineering — measure the token cost of each component

### 1.7 Model Selection

- [ ] Know the current landscape: GPT-4o, Claude Sonnet/Opus, Gemini, Llama 3 (open)
- [ ] Understand when to use open vs closed models
- [ ] Understand context window, quality, and cost tradeoffs for your specific use case

---

**✅ Stage 1 Checkpoint:** Build a structured data extractor. Feed it messy text (emails, meeting notes) and output validated, typed JSON. Use `instructor` + Pydantic. Handle malformed outputs and retry. Log all calls with token costs. Push to GitHub.

---

## STAGE 2 — RAG: Build Phase

> **Goal:** Build a production-quality RAG pipeline from scratch.  
> **Mindset:** Most engineers stop at vector search. Go deeper — hybrid retrieval, reranking, failure modes.

**Key libraries for this stage:**
| Library | What it does | When you use it |
|---|---|---|
| `llama-index` | RAG framework — ingestion, indexing, querying | Building end-to-end RAG pipelines quickly |
| `langchain` | LLM application framework — chains, RAG, agents | Widely used in industry; important to know even if you don't always use it |
| `unstructured` | Parse PDFs, HTML, Word docs, images | Data ingestion from messy real-world documents |
| `sentence-transformers` | Run embedding models locally | BGE, E5, and other HuggingFace embedding models |
| `qdrant-client` | Vector DB client for Qdrant | Local and cloud vector storage |
| `pgvector` | Postgres extension for vector search | Production-grade vector storage in your existing DB |
| `rank-bm25` | BM25 keyword search in Python | The keyword half of hybrid retrieval |
| `cohere` | Reranking, embeddings, and LLM | Cross-encoder reranking (`co.rerank()`) |
| `transformers` | HuggingFace model loading | Running any HuggingFace model locally |

### 2.1 Data Ingestion

- [ ] Handle clean vs noisy documents (PDFs, HTML, markdown, scanned text)
- [ ] Use **Unstructured** and **LlamaParse** for document parsing — know their failure modes
- [ ] Deduplicate documents: exact, near-duplicate, semantic
- [ ] Design a metadata schema (source, date, section, document ID, version)
- [ ] Implement source attribution — know where every chunk came from
- [ ] Build stable chunk IDs for update and refresh pipelines

### 2.2 Chunking ⚡

- [ ] Implement fixed-size chunking (baseline)
- [ ] Implement sliding window chunking with configurable overlap
- [ ] Implement semantic chunking (split on meaning, not character count)
- [ ] Implement hierarchical chunking (parent doc + child chunks) — LlamaIndex calls this "node parsers"
- [ ] Handle structure-aware chunking: headings, tables, code blocks separately
- [ ] Understand how chunk size affects recall, cost, and generation quality

### 2.3 Embeddings

- [ ] Understand what embeddings are and what similarity means in practice
- [ ] Know the current landscape: `text-embedding-3-large` (OpenAI), `embed-v3` (Cohere), BGE-M3, E5-large
- [ ] Use **sentence-transformers** to run embedding models locally
- [ ] Know how to handle embedding model updates (re-embed vs migrate)
- [ ] Understand normalized vs unnormalized embeddings and cosine vs dot-product similarity

### 2.3.1 HuggingFace Ecosystem ⚡

> JDs explicitly list "experience with HuggingFace." This is also required for fine-tuning in Stage 7A.

- [ ] Load any model using `AutoModel` and `AutoTokenizer` from `transformers`
- [ ] Use the HuggingFace Inference API to call models without local GPU
- [ ] Navigate model cards: understand model size, benchmarks, and license
- [ ] Use the `datasets` library to load evaluation datasets

### 2.4 Vector Databases (Hands-on) ⚡

- [ ] Build with **Qdrant** (great local + cloud, good Python SDK)
- [ ] Build with **pgvector** (Postgres extension — used heavily in production)
- [ ] Know the others exist: Pinecone, Weaviate, Milvus — know what problem each solves
- [ ] Understand HNSW index structure and tuning (ef, M parameters)
- [ ] Know approximate vs exact search and error bounds
- [ ] Implement index refresh strategies for dynamic data
- [ ] Design collection schemas with metadata fields

### 2.5 Retrieval ⚡

- [ ] Implement vector similarity search with `k` tuning
- [ ] Implement BM25 keyword search using `rank-bm25`
- [ ] Implement hybrid retrieval (vector + BM25, with RRF or weighted fusion)
- [ ] Implement metadata filtering (date, source, category)
- [ ] Implement query rewriting (expand vague queries using LLM)
- [ ] Implement multi-query expansion (generate multiple variants)
- [ ] Implement HyDE (Hypothetical Document Embeddings)
- [ ] Implement retrieval abstention (return nothing if no good match)

### 2.6 Reranking & Context Assembly ⚡

- [ ] Implement cross-encoder reranking using **Cohere Rerank** or `BGE-Reranker`
- [ ] Understand why cross-encoders outperform bi-encoders for ranking
- [ ] Implement token budget management — fit top-k chunks into the context limit
- [ ] Align citations to specific retrieved chunks in the final answer
- [ ] Order context for answerability (most relevant first vs last matters)

### 2.7 RAG Frameworks: LangChain vs LlamaIndex

> Know both. They're the two most common frameworks in production RAG systems.

- [ ] Build a RAG pipeline using **LlamaIndex** — use its ingestion pipelines, node parsers, and query engines
- [ ] Build the same pipeline using **LangChain** — use LCEL (LangChain Expression Language) chains
- [ ] Know when to use a framework vs building the pipeline yourself (frameworks add abstraction; sometimes you need control)
- [ ] Know the LangChain ecosystem: LCEL, LangChain Hub (prompt registry), LangSmith (tracing)
- [ ] Know the LlamaIndex ecosystem: SimpleDirectoryReader, VectorStoreIndex, RetrieverQueryEngine

### 2.8 Failure Modes

- [ ] Over-retrieval hallucination: model answers from noise, not signal
- [ ] Missing context: answer not in retrieved chunks
- [ ] Stale data: outdated content retrieved as current
- [ ] Chunk boundary hallucination: answer spans two chunks, neither retrieved
- [ ] Query-document vocabulary mismatch: user says "ML", doc says "machine learning"

---

**✅ Stage 2 Checkpoint:** Build a document QA system for a 100+ page document set using hybrid retrieval + Cohere reranking + source attribution. Build it once with LlamaIndex, once with LangChain. Push both to GitHub. Interviewers will ask which framework you used and why.

---

## STAGE 3 — RAG: Evaluate Phase

> **Goal:** Prove your RAG system works. Know when it doesn't.  
> **Mindset:** If you can't measure it, you can't improve it. Evaluation is a first-class engineering skill.

**Key libraries for this stage:**
| Library | What it does | When you use it |
|---|---|---|
| `ragas` | RAG evaluation framework (faithfulness, relevance, recall) | Automated RAG quality measurement |
| `langsmith` | LangChain's tracing + prompt registry | Pipeline tracing and eval runs |
| `langfuse` | Open-source LLM observability | Tracing, evals, cost tracking — works with any framework |
| `phoenix` (Arize) | ML + LLM observability | Span-level tracing, embedding visualization |
| `deepeval` | LLM evaluation framework | Unit testing for LLM outputs |

### 3.1 Building Eval Datasets

- [ ] Create a golden query set (50–200 Q&A pairs from your documents)
- [ ] Stratify by difficulty: easy, medium, hard, adversarial (out-of-scope)
- [ ] Create document-answer alignment pairs (which chunks answer which question)
- [ ] Use LLM-assisted eval generation — then validate its output

### 3.2 Retrieval Metrics

- [ ] Implement and interpret Recall@k
- [ ] Implement MRR (Mean Reciprocal Rank)
- [ ] Implement nDCG (Normalized Discounted Cumulative Gain)
- [ ] Attribute retrieval performance to each component (vector vs BM25)

### 3.3 Generation / Faithfulness Evaluation

- [ ] Evaluate faithfulness using **RAGAS** — does the answer come from retrieved context only?
- [ ] Evaluate answer relevance — does it address the question?
- [ ] Evaluate citation accuracy — do citations match the answer content?
- [ ] Test refusal conditions — does it refuse appropriately for out-of-scope questions?
- [ ] Use **DeepEval** for unit-testing individual LLM responses

### 3.4 Observability & Tracing

- [ ] Trace full pipelines using **LangFuse** or **Phoenix**
- [ ] Log every retrieval call: query, top-k results, scores, latency
- [ ] Log chunk contribution (which chunks contributed to the final answer)
- [ ] Implement A/B testing between retrieval strategies

### 3.5 Regression & Drift

- [ ] Detect index drift (document updates not reflected in index)
- [ ] Detect silent degradation (quality drops without errors)
- [ ] Set up regression tests that run after any pipeline change

---

**✅ Stage 3 Checkpoint:** Formally evaluate your Stage 2 RAG system. Document: Recall@5, top 3 failure modes with examples, one improvement with before/after metrics. Write the evaluation report as a README section — this is your strongest portfolio piece so far.

---

## ★ MILESTONE: Junior AI Engineer Hireable

> Build, debug, and evaluate RAG systems. Top 20% of candidates. Ready for junior roles.

---

## STAGE 4 — Agents

> **Goal:** Build reliable agents that use tools, reason through problems, and know when to stop.  
> **Mindset:** Build from scratch first. A reliable single-agent system beats a chaotic multi-agent one.

**Key libraries for this stage:**
| Library | What it does | When you use it |
|---|---|---|
| `langgraph` | Stateful, graph-based agent workflows | Production agents — the current industry standard |
| `crewai` | Role-based multi-agent coordination | When you need multiple specialized agents working together |
| `pydantic-ai` | Type-safe agent outputs using Pydantic | When you need strict, validated agent responses |
| `langchain` | Agent chains, tools, memory primitives | Widely used in existing codebases |
| `mcp` | Model Context Protocol Python SDK | Connecting agents to external tools/services via MCP |
| `langsmith` | Agent trajectory tracing | Debugging multi-step agent runs |
| `langfuse` | Cost + tracing across any framework | Token cost per trajectory, not just per call |

### 4.1 Agent Core Loop

- [ ] Implement observe → think → act → reflect from scratch — no framework first
- [ ] Track agent state explicitly (don't let it exist only in the prompt)
- [ ] Manage the token budget: system prompt + tool schemas + history + task state all compete for the same window
- [ ] Design system prompts that define: role, tools, decision rules, output format, stop conditions

### 4.2 Reasoning Frameworks ⚡

- [ ] Implement **ReAct** (Reason + Act): interleave reasoning traces with tool calls — the most common production pattern
- [ ] Know where ReAct fails: it over-commits to bad plans and doesn't backtrack
- [ ] Implement **Chain-of-Thought in agents**: explicit reasoning steps before acting
- [ ] Implement **Reflexion**: self-critique step where the agent scores and retries its own output
- [ ] Know when to use each: ReAct for tool-heavy tasks, Reflexion for quality-critical outputs

### 4.3 Tool Use ⚡

- [ ] Write clean, well-documented tool schemas
- [ ] Validate tool outputs before passing back to the agent
- [ ] Handle tool failures gracefully (retry, fallback, inform user)
- [ ] Build tool reliability scoring (track which tools fail often)
- [ ] Design tools to be idempotent where possible
- [ ] Handle OAuth and authentication for real external tools — not just API keys

### 4.4 Control & Safety ⚡

- [ ] Implement loop detection (detect infinite tool-call cycles)
- [ ] Implement self-stop conditions (agent declares task complete or impossible)
- [ ] Implement max-step limits with graceful degradation
- [ ] Implement human-in-the-loop escalation — when to ask before acting
- [ ] Handle partial task completion (makes progress but doesn't finish)

### 4.5 Session Memory (Short-Term)

- [ ] Implement buffer memory (full history with token management)
- [ ] Implement sliding window memory (keep last N turns)
- [ ] Implement summary memory (compress older turns with LLM)

### 4.6 Agent Frameworks ⚡

- [ ] Build one full agent from scratch first — then reach for a framework
- [ ] Use **LangGraph**: understand nodes, edges, state, interrupts, and human-in-the-loop
- [ ] Use **CrewAI**: understand role-based agents, crews, and task delegation
- [ ] Know **PydanticAI** for type-safe, validated agent outputs
- [ ] Know when each framework is appropriate vs when to go custom

### 4.7 MCP (Model Context Protocol)

- [ ] Understand MCP architecture: client, server, tools, resources, prompts
- [ ] Integrate an existing MCP server into your agent using the `mcp` SDK
- [ ] Understand security: MCP tool access is a significant attack surface
- [ ] Know MCP is now the de facto standard for agent-tool connectivity — supported by all major AI providers

### 4.8 Agent Skills / SKILL.md ⚡

> Open standard originally from Anthropic (December 2025). Now adopted across Claude Code, GitHub Copilot, Cursor, Windsurf, and OpenCode. MCP connects agents to tools. Skills give agents domain expertise and reusable workflows. They are complementary, not competing.

- [ ] Understand what Agent Skills are: directories with a `SKILL.md` file that agents load on-demand for domain-specific expertise
- [ ] Understand progressive disclosure: the agent sees only `name` and `description` upfront, loads the full `SKILL.md` body only when the task matches — keeps context efficient
- [ ] Know the SKILL.md format: YAML frontmatter (`name`, `description`) + markdown body (instructions, workflows, examples)
- [ ] Build a custom skill: package a repeatable workflow (e.g., "how to debug this codebase", "how to format this report type") into a SKILL.md
- [ ] Know the difference from MCP: MCP = connect to external services; Skills = domain expertise + internal workflows
- [ ] Know the security risk: SKILL.md files are a prompt injection vector — never load skills from untrusted sources

### 4.9 Agent Evaluation ⚡

- [ ] Understand trajectory-level vs outcome-level evaluation: did it take reasonable steps AND reach the right result?
- [ ] Build task-level test cases: input, expected outcome, acceptable tool call sequences
- [ ] Implement **LLM-as-judge**: use a separate LLM to score whether the agent's output meets task criteria
- [ ] Detect waste: unnecessary tool calls, redundant steps, runaway loops that inflate cost
- [ ] Build a regression suite — run it after every prompt or tool schema change
- [ ] Track over time: success rate, average steps, cost per task

### 4.10 AgentOps Basics

- [ ] Log every agent step: tool called, tool output, reasoning trace, tokens used
- [ ] Set cost budgets per agent run with hard stops — a runaway agent can cost 100× a single call
- [ ] Log enough state to replay and debug failed trajectories

---

**✅ Stage 4 Checkpoint:** Build a research agent. Version 1: ReAct from scratch. Version 2: LangGraph. Create a `SKILL.md` that packages your research workflow as a reusable skill. Evaluate 20 tasks with LLM-as-judge. Document success rate, steps, and cost per task. Push both versions to GitHub.

---

## STAGE 5 — Memory Systems

> **Goal:** Build agents that remember across sessions. Memory should change decisions, not just add tokens.

**Key libraries for this stage:**
| Library | What it does | When you use it |
|---|---|---|
| `redis` (Python) | Fast session memory with TTL | Short-term, expiring memory per user session |
| `langgraph` | SqliteSaver, MemorySaver for agent state | Checkpointing agent memory across runs |
| `qdrant-client` | Vector store for semantic memory | Retrieving relevant past interactions |
| `mem0` | Memory layer for AI applications | High-level memory management for agents |

### 5.1 Memory Foundations

- [ ] Distinguish memory vs RAG vs chat history — they solve different problems
- [ ] Understand working, short-term, and long-term memory
- [ ] Map long-term subtypes: episodic (events), semantic (facts), procedural (preferences)
- [ ] Know when memory is right vs when RAG is the better tool

### 5.2 Memory Types (Implementation)

- [ ] Implement episodic memory (store and retrieve past interaction events)
- [ ] Implement semantic memory (facts about user, world model)
- [ ] Implement procedural memory (learned preferences and behaviors)
- [ ] Handle session vs user isolation (multi-tenant architectures)

### 5.3 Memory Lifecycle ⚡

- [ ] Implement explicit memory creation ("remember this")
- [ ] Implement implicit memory creation (LLM-judged importance)
- [ ] Implement importance scoring (recency, goal relevance)
- [ ] Implement temporal decay (recent memories more relevant)
- [ ] Implement forgetting strategies (LRU, importance threshold, capacity limit)
- [ ] Handle contradictions: newer memory vs higher-confidence older memory

### 5.4 Storage Backends

- [ ] SQLite for lightweight persistent memory
- [ ] Redis for fast session memory with TTL expiry
- [ ] Vector store (Qdrant or Chroma) for semantic/episodic memory
- [ ] LangGraph SqliteSaver or MemorySaver for agent checkpointing
- [ ] Design memory record schema: content, type, timestamp, importance, user_id

### 5.5 Memory Retrieval ⚡

- [ ] Implement multi-signal ranking: recency × relevance × importance
- [ ] Implement retrieval gating — should memory even be injected for this query?
- [ ] Handle memory-RAG conflicts (memory says X, document says Y)
- [ ] Prevent memory hallucination (only inject high-confidence memories)

### 5.6 Memory Compression

- [ ] LLM-based summarization for old memory batches
- [ ] Daily/weekly temporal rollups
- [ ] Understand compression vs forgetting tradeoffs

### 5.7 Privacy & Compliance

- [ ] Implement user memory deletion (GDPR right to erasure)
- [ ] Separate memory by user with proper isolation
- [ ] Understand consent requirements for persistent memory

---

**✅ Stage 5 Checkpoint:** Build a multi-session assistant (study tutor or personal coach) that remembers preferences, past interactions, and topic mastery — and adapts over 5+ sessions. Push to GitHub with session transcripts showing memory in action.

---

## ◆ MILESTONE: Mid-Level AI Engineer Hireable

> End-to-end depth: build, evaluate, agents, memory. Rare combination that gets mid-level roles.

---

## STAGE 6 — Production & LLMOps

> **Goal:** Ship your AI system, keep it running, know exactly what it costs.

**Key libraries for this stage:**
| Library | What it does | When you use it |
|---|---|---|
| `fastapi` | Production-grade Python API framework | Wrapping every AI pipeline |
| `litellm` | Unified LLM proxy with cost tracking + caching | Production LLM routing, fallbacks, semantic caching |
| `langfuse` | Open-source LLM observability | Logging, cost tracking, eval runs |
| `langsmith` | LangChain observability + prompt registry | Tracing LangChain/LangGraph pipelines |
| `guardrails-ai` | Input/output validation and guardrails | Blocking injection, validating outputs |
| `prometheus-client` | Metrics collection | Latency, error rate, cost metrics |
| `mlflow` or `wandb` | Experiment tracking | Prompt experiments, model comparisons |

### 6.1 Serving & APIs

- [ ] Wrap your AI pipeline in a production **FastAPI** application
- [ ] Implement streaming endpoints (SSE or WebSocket) for real-time output
- [ ] Implement request queuing for high-load scenarios
- [ ] Handle concurrent requests safely (no shared mutable state)
- [ ] Write health check and readiness endpoints

### 6.2 Cloud Deployment ⚡

- [ ] Deploy your app to a cloud provider (AWS ECS/Lambda, GCP Cloud Run, Azure Container Apps)
- [ ] Use a cloud-managed Postgres + pgvector instance as your production vector store
- [ ] Store documents and embeddings in blob storage (S3/GCS/Blob)
- [ ] Manage secrets using the cloud secrets manager — not `.env` in production
- [ ] Configure auto-scaling (min and max instances)
- [ ] Know what Kubernetes does conceptually — you won't operate it at first, but interviews ask about it

### 6.3 Cost Engineering ⚡

- [ ] Model token cost per request (input + output tokens × rate)
- [ ] Use **LiteLLM** as a proxy for unified cost tracking across providers
- [ ] Implement semantic caching in LiteLLM — skip the LLM call if a similar query was cached
- [ ] Evaluate cheaper models for pipeline steps that don't need flagship quality
- [ ] Track cost per agent trajectory — not just per call (agents amplify costs across steps)
- [ ] Set cost alerts and hard budget limits

### 6.4 Reliability & Fallbacks

- [ ] Implement circuit breaker pattern for external API calls
- [ ] Implement fallback models using **LiteLLM** routing — if primary model fails, route to secondary
- [ ] Implement exponential backoff with jitter
- [ ] Handle partial failures gracefully (system degrades, not crashes)
- [ ] Implement rollback strategy for prompt or model changes

### 6.5 Basic Guardrails ⚡

> These are now production baseline — not optional safety extras.

- [ ] Implement input validation: detect and block prompt injection attempts
- [ ] Understand prompt injection: what it is, why it's dangerous, and how attackers exploit it
- [ ] Implement output validation: check response matches expected format and content constraints
- [ ] Block sensitive data leakage in RAG: don't expose system prompts or private documents
- [ ] Implement topic/scope guardrails: detect and handle off-topic or harmful queries
- [ ] Use **Guardrails AI** or **NeMo Guardrails** — know the tradeoff vs building your own
- [ ] Know that SKILL.md files are a prompt injection vector — validate before deploying

### 6.6 CI/CD for AI Systems ⚡

- [ ] Write prompt regression tests (snapshot testing for prompt outputs)
- [ ] Run eval suite automatically on every code/prompt change
- [ ] Implement canary deployments for model or prompt updates
- [ ] Use feature flags to gradually roll out AI changes
- [ ] Track experiments with **MLflow** or **W&B**

### 6.7 Monitoring & Observability

- [ ] Log every LLM call: model, prompt hash, tokens, latency, cost — use **LangFuse**
- [ ] Trace retrieval quality scores per request
- [ ] Alert on: latency spikes, error rate increases, cost anomalies
- [ ] Track p95/p99 latency, not just averages
- [ ] Collect user feedback and connect it to failure analysis

---

**✅ Stage 6 Checkpoint:** Deploy your RAG + agent system to cloud with: cost monitoring dashboard, p95 latency tracking, prompt regression tests in CI, canary deployment, and guardrails blocking prompt injection. Document the cost per 1,000 queries. Make the README exceptional — this is your flagship portfolio project.

---

## STAGE 7 — Differentiators: Pick 1–2

> Deep on one beats shallow on all. Choose based on the companies you're targeting.

### 7A — Fine-Tuning

**Key libraries:** `transformers`, `peft`, `trl`, `bitsandbytes`, `datasets`

- [ ] Understand when fine-tuning is right: behavioral shaping, style, domain-specific format — not knowledge injection (use RAG for that)
- [ ] Curate a high-quality instruction dataset
- [ ] Implement LoRA fine-tuning using **HuggingFace PEFT**
- [ ] Implement QLoRA using **bitsandbytes** for memory-efficient training
- [ ] Evaluate fine-tuned model rigorously against base model
- [ ] Know the risks: catastrophic forgetting, alignment drift, overfitting

### 7B — Advanced RAG

**Key libraries:** `neo4j`, `colbert-ai`, `haystack`

- [ ] Implement GraphRAG (knowledge graph + vector retrieval)
- [ ] Implement Agentic RAG (iterative, self-reflective retrieval)
- [ ] Implement CRAG (Corrective RAG — assess quality, fall back to web search)
- [ ] Use late-interaction models: ColBERT, ColPali
- [ ] Fine-tune an embedding model on domain-specific data

### 7C — Multi-Agent Systems

**Key libraries:** `crewai`, `autogen`, `langgraph`

- [ ] Implement orchestrator/subagent delegation patterns
- [ ] Build with **CrewAI**: role-based agents, crews, task flows
- [ ] Build with **AutoGen**: conversational multi-agent patterns
- [ ] Implement supervisor patterns: a meta-agent that monitors subagent behavior
- [ ] Understand failure cascades: one bad agent can poison the whole crew
- [ ] Know the tradeoff: more agents = more capability but more cost, latency, and failure surface

### 7D — AI Safety & Red Teaming

- [ ] Red team your own RAG and agent systems systematically
- [ ] Build adversarial test suites for your agents
- [ ] Understand how multi-agent systems create new attack surfaces
- [ ] Know constitutional AI and RLHF basics

### 7E — Edge / On-Device Inference

**Key libraries:** `llama-cpp-python`, `ollama`, `ctransformers`

- [ ] Run models locally with **llama.cpp** or **Ollama**
- [ ] Understand quantization: GGUF formats, Q4 vs Q8 tradeoffs
- [ ] Build a hybrid architecture (local model for fast tasks, cloud for complex ones)

---

**✅ Stage 7 Checkpoint:** Publish something — GitHub repo, blog post, local meetup talk, open-source contribution. By now you should have 6–8 repos from checkpoints. Pin the best ones and write a portfolio README that tells your learning story.

---

## Continuous Learning

- [ ] Read 1 new architecture/systems paper per week (arXiv Sanity, Karpathy's recommendations)
- [ ] Follow: Jason Wei, Tri Dao, Tim Dettmers, Sebastian Raschka, Jeremy Howard
- [ ] Reproduce one non-trivial technique every 2–3 months
- [ ] Contribute to open-source RAG/agent frameworks
- [ ] Write about what you learn — forces clarity, builds visibility

---

## Quick Library Reference

| Library | Stage | Must Know? | What It Does |
|---|---|---|---|
| `openai` | 1 | ✅ Yes | Official OpenAI SDK |
| `anthropic` | 1 | ✅ Yes | Official Anthropic SDK |
| `litellm` | 1, 6 | ✅ Yes | Unified interface for 100+ LLM providers |
| `instructor` | 1 | ✅ Yes | Typed LLM outputs via Pydantic |
| `pydantic` | 1+ | ✅ Yes | Data validation — used everywhere |
| `langchain` | 2, 4 | ✅ Yes | LLM app framework, widely used in industry |
| `llama-index` | 2 | ✅ Yes | RAG framework — ingestion, indexing, querying |
| `sentence-transformers` | 2 | ✅ Yes | Local embedding models |
| `qdrant-client` | 2, 5 | ✅ Yes | Vector DB client |
| `pgvector` | 2, 6 | ✅ Yes | Postgres vector search |
| `cohere` | 2 | ✅ Yes | Reranking and embeddings |
| `transformers` | 2, 7A | ✅ Yes | HuggingFace model hub |
| `ragas` | 3 | ✅ Yes | RAG evaluation framework |
| `langfuse` | 3, 6 | ✅ Yes | Open-source LLM observability |
| `langsmith` | 3, 4 | ✅ Yes | LangChain tracing + eval |
| `deepeval` | 3 | 👍 Good | LLM unit testing |
| `langgraph` | 4 | ✅ Yes | Stateful agent workflows |
| `crewai` | 4, 7C | 👍 Good | Multi-agent coordination |
| `pydantic-ai` | 4 | 👍 Good | Type-safe agents |
| `mcp` | 4 | ✅ Yes | Model Context Protocol SDK |
| `mem0` | 5 | 👍 Good | Memory layer for agents |
| `fastapi` | 6 | ✅ Yes | Production API framework |
| `guardrails-ai` | 6 | 👍 Good | Input/output guardrails |
| `mlflow` or `wandb` | 6 | 👍 Good | Experiment tracking |
| `peft` | 7A | Optional | HuggingFace fine-tuning (LoRA, QLoRA) |
| `trl` | 7A | Optional | Reinforcement learning from human feedback |
| `ollama` | 7E | Optional | Run models locally |
