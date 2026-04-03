# AI Engineer Foundations

> **This file is the enabler, not the job.** These skills let you do AI engineering — they're not what you're hired for.  
> **Complete these before starting `01_ai_engineering.md`.** You don't need to master them — "good enough to build" is the bar.  
> **If you already work as a software engineer**, you likely have most of Stage 0 done. Audit and move on.

---

## Progress Overview

| Stage | Topic                          | Time Estimate | Status         |
| ----- | ------------------------------ | ------------- | -------------- |
| 0.1   | Python Proficiency             | 2–3 weeks     | ⬜ Not started |
| 0.2   | API & HTTP Basics              | 3–5 days      | ⬜ Not started |
| 0.3   | Developer Tooling              | 3–5 days      | ⬜ Not started |
| 0.4   | Docker Basics                  | 1 week        | ⬜ Not started |
| 0.5   | Data & File Handling           | 3–5 days      | ⬜ Not started |
| 0.6   | SQL Fundamentals               | 1–2 weeks     | ⬜ Not started |
| 0.7   | Cloud Platform Orientation     | 1 week        | ⬜ Not started |

> ⬜ Not started / 🔄 In progress / ✅ Done

---

## 0.1 Python Proficiency

> Python is in ~100% of AI engineer job postings. This is non-negotiable.

- [ ] Write and reason about async functions (`async/await`, `asyncio`) — critical for LLM APIs
- [ ] Use OOP confidently: classes, inheritance, dataclasses
- [ ] Work with standard data structures: dicts, lists, sets, queues, deques
- [ ] Handle errors cleanly: try/except, custom exceptions, structured logging
- [ ] Use type hints consistently — modern Python codebases all use them
- [ ] Write clean, readable code — not clever code

**You're done when:** You can write an async FastAPI endpoint that calls an external API, validates the response with Pydantic, handles errors, and logs the result — without looking anything up.

---

## 0.2 API & HTTP Basics

> You'll call LLM APIs, vector DB APIs, and build your own — daily.

- [ ] Call REST APIs using `requests` and `httpx` (prefer `httpx` for async)
- [ ] Understand HTTP methods, status codes, headers, and authentication (Bearer tokens, API keys)
- [ ] Handle pagination and rate limits from external APIs
- [ ] Parse and validate JSON responses
- [ ] Understand webhooks — what they are and when APIs use them

---

## 0.3 Developer Tooling

> The infrastructure of professional software development. Non-negotiable in any engineering role.

- [ ] Use Git fluently: branching, PRs, rebasing, resolving conflicts, stashing
- [ ] Write a proper `.gitignore` and `README.md` — every repo needs both
- [ ] Use virtual environments (`venv` or `uv` — `uv` is the modern choice)
- [ ] Write and run tests with `pytest`
- [ ] Use pre-commit hooks (black, ruff) for consistent code quality
- [ ] Know how to read a stack trace and trace a bug systematically

---

## 0.4 Docker Basics

> Docker appears in almost every AI engineer JD. You need to containerize your apps.

- [ ] Write a working `Dockerfile` for a Python app
- [ ] Build and run containers locally
- [ ] Use `docker-compose` for multi-service setups (app + vector DB + postgres)
- [ ] Understand environment variables and secrets management — `.env` for local, secrets manager for production
- [ ] Know the difference between an image and a container
- [ ] Know what Kubernetes does conceptually — container orchestration at scale. You won't operate it at first, but you'll be asked about it.

**You're done when:** You can containerize a FastAPI app with a Postgres dependency using `docker-compose`, run it locally, and explain what each Dockerfile line does.

---

## 0.5 Data & File Handling

> AI pipelines deal with a lot of files. You need to handle them without friction.

- [ ] Read/write CSV, JSON, and Parquet files with `pandas`
- [ ] Work with file systems using `pathlib` (not `os.path` — use pathlib)
- [ ] Handle encoding issues (UTF-8, BOM)
- [ ] Read and write files from/to cloud storage (S3, GCS) — this is in Stage 0.7

---

## 0.6 SQL Fundamentals ⚡

> SQL appears in the majority of AI Engineer JDs. You'll query LLM call logs, debug pipelines, and build evaluation dashboards. You don't need to be a data analyst — just fluent enough to not be blocked.

- [ ] Write SELECT queries with WHERE, ORDER BY, LIMIT
- [ ] Use JOINs (INNER, LEFT) to combine tables
- [ ] Use GROUP BY with aggregates: COUNT, SUM, AVG, MAX
- [ ] Understand window functions (ROW_NUMBER, LAG) — useful for time-series log queries
- [ ] Write CTEs (`WITH ... AS`) for readable multi-step queries
- [ ] Understand indexes — why they exist, when they matter
- [ ] Connect to Postgres from Python using `psycopg2` or `SQLAlchemy`
- [ ] Know the difference between Postgres (relational) and a vector DB (similarity search) — they coexist in production

**Practical task:** Create a table to log LLM calls (model, tokens_in, tokens_out, latency_ms, cost_usd, created_at). Insert 100 rows. Write queries to answer: "What was the average latency by model?" and "What was the total cost in the last 7 days?"

---

## 0.7 Cloud Platform Orientation ⚡

> Cloud appears in 70–80% of AI Engineer JDs. Pick one provider and stay with it.  
> **AWS** is the safest career choice. **GCP** is strong for AI services. **Azure** if targeting enterprise.  
> You're not becoming a DevOps engineer — learn enough to ship and integrate, not enough to architect infrastructure.

### Pick your provider, then complete these:

- [ ] Set up an account and configure the CLI (`aws configure`, `gcloud init`, or `az login`)
- [ ] Understand IAM basics: roles, policies, least-privilege — never commit credentials to Git
- [ ] Use blob storage (S3 / GCS / Azure Blob) — read and write files from Python
- [ ] Deploy a simple Python app to serverless compute (Lambda / Cloud Functions / Azure Functions)
- [ ] Call a cloud-native AI API:
  - AWS: **Amazon Bedrock** (Claude, Llama, Titan)
  - GCP: **Google Vertex AI** (Gemini, Claude)
  - Azure: **Azure OpenAI Service** (GPT-4o, embeddings)
- [ ] Understand why companies use cloud AI APIs instead of direct OpenAI/Anthropic (compliance, billing, data residency)
- [ ] Connect to a cloud-managed Postgres instance from Python — this is your production vector store (with pgvector)
- [ ] Read a monthly cloud cost estimate — know what you're spending before it surprises you

**You're done when:** You can deploy a Python app to cloud serverless, read and write files from blob storage, and call a cloud AI API — all without following a tutorial step-by-step.

---

## ✅ Foundations Checkpoint

Build and deploy a FastAPI app that:
1. Accepts a text payload
2. Calls a cloud-native AI API (Bedrock / Vertex / Azure OpenAI)
3. Logs the call (tokens, latency, cost) to a Postgres database
4. Stores any uploaded files in blob storage
5. Returns structured JSON validated with Pydantic

Run it locally with Docker. Deploy it to a cloud serverless function.  
Write a SQL query that answers: "What was the average cost and latency of the last 100 requests?"  
Push to GitHub with a README and deploy instructions.

**When you can do all of this without a tutorial, start `01_ai_engineering.md`.**

---

## What Each Foundation Skill Unlocks

| Foundation | Why it matters in AI Engineering |
|---|---|
| Python async | LLM APIs are async by nature. Streaming, concurrent requests, parallel tool calls. |
| Type hints + Pydantic | Structured LLM outputs require validation. Pydantic is the backbone. |
| Docker | Every AI app you build gets containerized before it gets deployed. |
| SQL | You'll query LLM call logs, eval results, and metadata tables constantly. |
| Git | Prompts are code. Eval datasets are code. Version them. |
| Cloud | Companies don't run AI on laptops. You need to deploy and integrate cloud-native AI APIs. |
| HTTP / REST | You're calling LLM APIs, vector DB APIs, and building your own — all day. |
