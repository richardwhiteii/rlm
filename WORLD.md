# World: rlm

> Recursive Language Models integration for Claude Code — treating massive context as external variables, enabling programmatic chunking and recursive sub-calls.

---

## The Paper

**Source**: https://arxiv.org/html/2512.24601v1

**Core Insight**: Instead of feeding massive contexts directly into an LLM, treat the prompt as an external variable in a REPL environment. The model can then:
1. Load context as a Python variable (not in prompt)
2. Inspect structure (print first N chars, count lines)
3. Chunk strategically based on structure
4. Use sub-LLM calls per chunk with targeted queries
5. Aggregate results programmatically
6. Final LLM call to synthesize answer

**Key Patterns**:
- Filter via code (regex/string ops before sub-calls)
- Semantic chunking (by structure, not arbitrary char count)
- Sub-call for classification (one LLM call per chunk)
- Variable buffers (store outputs in Python, aggregate programmatically)
- Answer verification (final sub-call to verify against evidence)

**Result**: Handles 10M+ tokens with comparable cost to base models.

---

## Integration Approaches (Ranked by Autonomy)

| Rank | Option | Score | Rationale |
|------|--------|-------|-----------|
| 1 | MCP Server | ⭐⭐⭐⭐⭐ | Claude invokes tools at will — full control over when/how to apply RLM |
| 2 | Skill | ⭐⭐⭐⭐ | Claude reads SKILL.md, applies pattern — requires remembering to check |
| 3 | Hook | ⭐⭐⭐ | Auto-fires on conditions — good guardrail, poor adaptability |
| 4 | Slash Command | ⭐⭐ | Human must invoke — zero autonomy |

**Decision**: Start with MCP Server for maximum autonomous effectiveness.

---

## Architecture (Mac Studio)

```
/data/rlm/
├── contexts/      # Raw loaded contexts
├── chunks/        # Pre-chunked versions
├── embeddings/    # Optional: for retrieval
└── results/       # Cached sub-call outputs

MCP Server ↔ Filesystem ↔ Ollama (local sub-calls, cost = $0)
```

---

## MCP Server Tools (Planned)

```python
load_context(name, content)     # Load context as external variable
chunk_context(name, strategy)   # Chunk by lines/chars/semantic
sub_llm_query(query, chunk)     # Recursive sub-call
aggregate_results(results)      # Combine chunk outputs
```

---

## Unfinished

- [ ] Build MCP server with core tools
- [ ] Wire Ollama for local sub-calls
- [ ] Test on real massive codebase
- [ ] Add caching for chunked contexts
- [ ] Hook as safety net (auto-intercept >500K direct ingestion)
- [ ] Skill documentation for when to use RLM

---

## Session Log

### 2026-01-16

**Starting point**: Previous conversation in docs/conversation.md explored the RLM paper and four integration approaches.

**Decision**: Build MCP Server first — highest autonomy score, Claude can invoke tools at will and adapt strategy mid-execution.

**Built**: MCP server with 9 tools (load, inspect, chunk, get_chunk, filter, sub_query, store_result, get_results, list_contexts). Wired globally via ~/.claude/.mcp.json.

**Tested**: Loaded 2MB "One Big Beautiful Bill Act" at 10% context remaining. Chunked into 10 pieces. Sub-queried chunk 0 via Ollama (gemma3:27b) — got meaningful summary of tax policy bill. Proved the pattern works.

**Key insight**: For Anthropic sub-calls, use Agent SDK to reuse existing Claude subscription instead of raw API with separate key. Avoids double-billing.

**Next session**:
- [ ] Refactor sub_query to use Agent SDK for Anthropic provider
- [ ] Test full recursive pattern: chunk all → sub-query each → aggregate
- [ ] Add API key passthrough or Agent SDK integration

### 2026-01-16 (Session 2)

**Research finding**: Agent SDK also requires a separate API key — it cannot reuse Claude Code subscription credentials. The original hypothesis about avoiding double-billing through Agent SDK was incorrect.

**New strategy**: Make Ollama the primary provider (free local inference), keep Anthropic as quality fallback.

**Refactored sub_query**:
- Default provider changed: `anthropic` → `ollama`
- Default model per provider: `gemma3:27b` (Ollama), `claude-sonnet-4-20250514` (Anthropic)
- Structured JSON responses with metadata (provider, model, truncation status)
- Better error handling (connection errors, timeouts, structured error responses)
- Context limits: 100K chars (Anthropic), 50K chars (Ollama)
- Timeout increased: 120s → 180s for Ollama

**Refactored again** (per user request):
- Added `claude-code-sdk` as third provider option
- Converted all providers to async/non-blocking:
  - `httpx.AsyncClient` for Ollama
  - `anthropic.AsyncAnthropic` for Anthropic API
  - `claude_query` async generator for Claude Code SDK
- Installed `claude-agent-sdk==0.1.19`

**Two providers now available** (anthropic removed):
| Provider | Default Model | Cost | Notes |
|----------|--------------|------|-------|
| `claude-sdk` | claude-opus-4-5 | API | Uses Claude Agent SDK (default) |
| `ollama` | gemma3:27b | Free | Local inference fallback |

**Tested** ✅:
- Both providers working: `claude-sdk` (claude-sonnet-4) and `ollama` (gemma3:27b)
- Async/non-blocking calls confirmed functional

**Next**:
- [x] Test full recursive pattern: chunk all → sub-query each → aggregate ✅

### Full Recursive Pattern Test (2026-01-16)

**Pattern**: Load → Chunk → Sub-query each → Store → Aggregate

**Test run**:
- Loaded 2,278 char document (4 topics: AI, climate, space, quantum)
- Chunked into 8 paragraphs
- Sub-queried 4 content chunks with `provider="auto"`
- All handled by Ollama gemma3:27b (no escalation)
- Aggregated results successfully

**Validated**: Core RLM pattern works end-to-end at $0 cost.

---

## Adaptive Format Detection

**Decision**: RLM should adapt chunking strategy based on input format.

**Formats detected**:
| Format | Detection | Chunking Strategy |
|--------|-----------|-------------------|
| XML/HTML | `<?xml`, `<html`, `<!DOCTYPE` | By structural elements (lxml/ElementTree) |
| JSON | Starts with `{` or `[`, valid JSON | By array elements or object keys |
| Markdown | `# `, triple backticks | By paragraphs |
| Python | `def `, `class `, `import ` | By paragraphs |
| JavaScript | `function `, `const `, `let ` | By paragraphs |
| Text | Default | By paragraphs |

**XML parsing** (lxml + ElementTree fallback):
- Selector as CSS class: `lbexSectionlevelOLC`
- Selector as XPath: `//div[@class='content']`
- Auto-detect structural tags: section, article, div, chapter

**Tested with**: 2MB legislation XML (H.R.1 One Big Beautiful Bill Act)
- Regex chunking: 310 sections found
- XML parsing with lxml: 35 sections with explicit selector

### LLM-Based Discovery (2026-01-16)

**Insight**: Use cheap LLM (Ollama/Haiku) to understand format before chunking.

**New tool**: `rlm_discover_context`
- Samples first 5-10K chars
- Sends to Ollama gemma3:27b with analysis prompt
- Returns: format, structure, chunking_strategy, selector, estimated_chunks

**Flow**:
```
Load → Discover (LLM) → Get recommendation → Chunk → Sub-query → Aggregate
```

**Supported formats** (LLM can recognize any):
- XML, HTML, JSON, CSV, TSV
- Markdown, plain text
- Python, JavaScript, SQL, etc.
- Excel (needs extraction first)

---

## Model Escalation Strategy

**Decision**: Implement Option A (simple retry) for cost optimization.

**Escalation Chain**:
```
ollama (gemma3:27b) → haiku → sonnet → opus
     $0                ~$0.25   ~$3      ~$15
```

**Escalation Triggers** (Option A):
- Connection/API error
- Empty or very short response (<50 chars)
- Explicit "I don't know" phrases

**Implemented** ✅:
- `provider="auto"` enables escalation (now default)
- `max_escalation` parameter: `haiku`, `sonnet` (default), `opus`
- Response includes `escalation_path` and `final_model`
- Helper `_make_subcall()` for clean async calls
- Helper `_is_poor_response()` for escalation detection

**Future Options to Explore**:
- **Option B (Self-assessment)**: Model rates its own confidence (1-10), escalate if <7
- **Option C (Judge model)**: Separate haiku call evaluates response quality
- **Confidence threshold parameter**: Let caller specify minimum confidence level
- **Task-specific escalation**: Different thresholds for summarization vs. analysis vs. code

### 2026-01-16 (Session 3: Simplification)

**Decision**: Simplify back to paper's core pattern. Trust LLM to control decomposition, not pre-built tooling.

**Removed**:
- `rlm_discover_context` tool (LLM-based format detection)
- `_detect_format()` function
- `_chunk_xml()` function
- `_is_poor_response()` function
- XML/JSON parsing logic (lxml/ElementTree)
- Model escalation chain
- `auto` provider with escalation logic
- `selector` parameter from chunking

**Simplified chunking strategies**:
- Before: `auto`, `lines`, `chars`, `paragraphs`, `regex`, `xml`, `json`
- After: `lines`, `chars`, `paragraphs` only

**Simplified sub_query**:
- Before: `auto` (escalation chain), `ollama`, `claude-sdk`
- After: `ollama` (default), `claude-sdk` only
- No escalation logic — just make the call, return result
- Removed `max_escalation` parameter

**Result**: ~540 lines (was 930), focused on:
1. Load context as external variable
2. Inspect structure
3. Simple chunking (lines/chars/paragraphs)
4. Simple sub-query (ollama or claude-sdk)
5. Store/retrieve results

**Rationale**: The paper's insight is that the LLM decides how to decompose — not the tooling. Adding format detection, auto-chunking strategies, and escalation chains was premature optimization. Let Claude inspect, decide strategy, and choose providers explicitly.

### Full Pattern Test: 2MB Legislation (2026-01-16)

**Document**: H.R.1 One Big Beautiful Bill Act (1.98 MB XML)

**Processing**:
- Chunked: 40 chunks (50K chars each)
- Sampled: 8 chunks (20%)
- Provider: Ollama gemma3:27b
- Success: 100% (8/8)
- Cost: $0

**Topics discovered**: Tax policy, defense, energy, R&D incentives, healthcare, immigration fees

**Patterns identified**: Permanent vs temporary provisions, domestic preference, multi-year funding

**Result**: Core RLM pattern validated on real 2MB legislation at zero cost.

### Future Tests

- [ ] Compare Ollama models (gemma3:4b vs gemma3:27b) on same query
- [ ] Measure speed vs accuracy tradeoff
- [ ] Consider model selection based on task complexity
