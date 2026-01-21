# RLM Roadmap

## Vision

A production-ready MCP server implementing Recursive Language Model patterns for handling massive contexts (10M+ tokens) with local LLM inference.

---

## Phase 1: Python Prototype ✅ (Current)

**Status:** Complete

- [x] Core MCP server with 9 tools
- [x] Context loading and persistence
- [x] Simple chunking (lines, chars, paragraphs)
- [x] Sub-query with Ollama and Claude SDK
- [x] Result storage and aggregation
- [x] Batch sub-query with concurrency control
- [x] Validated on 2MB legislation (500K tokens)

**Learnings:**
- Paper's core pattern works: load → chunk → sub-query → aggregate
- Simple strategies sufficient (no need for XML/JSON parsing)
- Trust the LLM to decide decomposition strategy
- Local Ollama enables $0 cost processing

---

## Phase 2: Python Hardening (Next)

- [ ] Add `file_path` parameter to load directly from disk (bypass tool size limits)
- [ ] Model comparison testing (gemma3:27b vs phi4-reasoning vs gemma3:4b)
- [ ] Performance benchmarks (tokens/sec, memory usage)
- [ ] Error handling improvements
- [ ] Logging and observability
- [ ] Unit tests for core functions
- [ ] Documentation and examples

---

## Phase 3: Go Rewrite

**Why Go over Rust:**
- I/O bound workload (HTTP calls, file reads) — Go excels here
- Simpler async model (goroutines vs tokio)
- Faster development cycle
- Trivial cross-compilation
- Excellent JSON/HTTP stdlib
- Good enough performance for orchestration layer

**Deliverables:**
- [ ] Go MCP server with same tool interface
- [ ] Single static binary (~10MB)
- [ ] Cross-platform builds (mac/linux/windows, arm64/amd64)
- [ ] GitHub Actions CI/CD
- [ ] Homebrew formula
- [ ] Drop-in replacement for Python version

**Architecture:**
```
rlm-server (Go binary)
    ├── MCP JSON-RPC handler
    ├── Context store (memory + disk)
    ├── Chunking engine
    ├── Ollama client (HTTP)
    └── Claude SDK client (HTTP)
```

---

## Phase 4: Distribution

- [ ] GitHub Releases with pre-built binaries
- [ ] Homebrew tap: `brew install rlm-mcp`
- [ ] npm wrapper: `npx rlm-mcp`
- [ ] Docker image (optional)
- [ ] One-line install script

**Target MCP config:**
```json
{
  "rlm": {
    "command": "rlm-server"
  }
}
```

---

## Phase 5: Advanced Features

- [ ] Embeddings support (nomic-embed-text for semantic chunking)
- [ ] Caching layer (avoid re-processing same chunks)
- [ ] Streaming responses
- [ ] Multi-model orchestration (route by task complexity)
- [ ] Metrics/telemetry endpoint
- [ ] Plugin system for custom chunking strategies

---

## Non-Goals

- Complex format detection (trust the LLM)
- Auto-escalation between models (let user control)
- Built-in XML/JSON parsing (keep it simple)
- Training or fine-tuning (use existing models)

---

## Success Metrics

1. Process 10M+ token documents at $0 cost (Ollama)
2. Single binary, zero dependencies
3. <100ms tool response latency (excluding LLM time)
4. Works with any Ollama model
5. Drop-in MCP server for Claude Code

---

## Timeline (Rough)

| Phase | Estimate |
|-------|----------|
| Phase 2 (Hardening) | 1-2 weeks |
| Phase 3 (Go Rewrite) | 2-3 weeks |
| Phase 4 (Distribution) | 1 week |
| Phase 5 (Advanced) | Ongoing |
