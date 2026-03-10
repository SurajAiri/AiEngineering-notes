# 🔧 Structured Outputs & Function Calling

## 📚 Overview

Getting reliable, structured data from LLMs is one of the most critical skills in AI engineering. Raw text output is useless for pipelines — you need JSON, typed objects, and tool calls that actually parse. This note covers JSON mode, Pydantic-based extraction (Instructor), constrained decoding, grammar-guided generation, function calling deep dives, and MCP (Model Context Protocol) — the emerging standard for tool integration.

> 📌 _Every production LLM system needs structured outputs. If you're regex-parsing LLM text, you're doing it wrong._

---

## 🎯 Learning Objectives

- **Extract** structured data from LLMs using JSON mode and Pydantic schemas
- **Use** Instructor library for validated, retry-aware extraction
- **Implement** function calling with OpenAI and Anthropic (tool use)
- **Understand** constrained decoding and grammar-guided generation (Outlines, llama.cpp GBNF)
- **Design** tool schemas that minimize LLM errors
- **Connect** to MCP servers for standardized tool integration

---

## 🧠 Sections (To Be Written)

### 1. JSON Mode

- OpenAI JSON mode vs response_format
- Anthropic structured output
- Failure modes: invalid JSON, missing fields, extra fields
- Schema design best practices (simple > nested)

### 2. Pydantic-Based Extraction (Instructor)

- Instructor library setup and patterns
- Pydantic models as output schemas
- Validation and automatic retries
- Nested object extraction
- List extraction and streaming
- Partial extraction for long-running tasks

### 3. Function Calling Deep Dive

- OpenAI function calling (tools API)
- Anthropic tool use
- Multi-tool selection and parallel tool calls
- Forced vs auto tool selection
- Error handling in tool execution
- Tool result formatting

### 4. Constrained Decoding

- Outlines library: regex and JSON schema constraints
- llama.cpp GBNF grammars
- Why constrained decoding guarantees valid output
- Performance implications (speed vs flexibility)
- When to use constrained decoding vs JSON mode

### 5. MCP — Model Context Protocol

- MCP architecture (client, server, transport)
- Building MCP tool servers
- Connecting LLMs to MCP tools
- MCP vs function calling (when to use which)
- MCP ecosystem and existing servers

### 6. Common Pitfalls

| Symptom                 | Cause                       | Fix                                       |
| ----------------------- | --------------------------- | ----------------------------------------- |
| Invalid JSON from LLM   | No JSON mode enabled        | Use response_format or Instructor         |
| Wrong tool selected     | Ambiguous tool descriptions | Clear, distinct tool names + descriptions |
| Nested extraction fails | Schema too complex          | Flatten or chain extractions              |
| MCP connection drops    | Transport misconfiguration  | Health checks + reconnection logic        |

---

## 📖 Resources

- Instructor library documentation
- OpenAI function calling guide
- Anthropic tool use documentation
- Outlines: structured generation library
- Model Context Protocol specification

---

## ➡️ Next Steps

Continue to [Model Selection & Comparison](./03_model_selection.md) →
