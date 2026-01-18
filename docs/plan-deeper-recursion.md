# Plan: Deeper Recursion for RLM

> Enable sub-queries to spawn further RLM calls - true recursive language model patterns.

---

## 1. Current State

The RLM MCP server currently supports:
- Loading contexts as external variables
- Chunking with lines/chars/paragraphs strategies
- Sub-queries via `rlm_sub_query` and `rlm_sub_query_batch`
- Two providers: `ollama` and `claude-sdk`

**Limitation**: Sub-queries are terminal - they cannot spawn further RLM calls. The sub-model receives context but has no access to RLM tools.

---

## 2. Design

### 2.1 Provider Matrix

| Provider | Recursive Support | Mechanism |
|----------|-------------------|-----------|
| `claude-sdk` | Yes | Pass RLM tools via ClaudeAgentOptions |
| `ollama` | Yes | Use `/api/chat` endpoint with tools array |

Both providers now support recursive calls through their respective tool-calling mechanisms.

### 2.2 Ollama Tool Calling Details

**Recommended Model**: `olmo-3.1:32b`
- 32B parameters with strong reasoning capabilities
- 64K token context window (ideal for large chunks)
- Open weights from Allen AI with full training transparency
- Run: `ollama run olmo-3.1:32b`

Ollama supports tool calling via the `/api/chat` endpoint (not `/api/generate`):

- **Endpoint**: `/api/chat` (replaces `/api/generate` for all calls)
- **Tool format**: OpenAI-compatible JSON schema
- **Response**: Includes `tool_calls` array when model decides to use tools
- **Agent loop**: Requires iterative call pattern to handle tool use

**Request format**:
```json
{
    "model": "olmo-3.1:32b",
    "messages": [
        {"role": "user", "content": "Analyze this large codebase..."}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "rlm_chunk_context",
                "description": "Chunk a loaded context...",
                "parameters": { ... }
            }
        }
    ],
    "stream": false
}
```

**Response with tool call**:
```json
{
    "message": {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "rlm_chunk_context",
                    "arguments": "{\"name\": \"codebase\", \"strategy\": \"lines\"}"
                }
            }
        ]
    }
}
```

**Agent loop pattern**:
1. Send initial request with tools
2. If response contains `tool_calls`, execute each tool
3. Append tool results to messages
4. Send follow-up request
5. Repeat until no more tool calls or max iterations reached

### 2.3 Recursion State

Track recursion depth to prevent infinite loops:

```python
@dataclass
class RecursionState:
    current_depth: int = 0
    max_depth: int = 3
    call_trace: list[str] = field(default_factory=list)
```

### 2.4 Updated Function Signatures

**Switch to `/api/chat` for ALL Ollama calls** (consistency):

```python
async def _call_ollama(
    query: str,
    context_content: str,
    model: str,
    tools: list[dict] | None = None,  # NEW
    state: RecursionState | None = None,  # NEW
) -> tuple[Optional[str], Optional[str], RecursionState | None]:
    """Make a sub-call to Ollama using /api/chat endpoint.

    Args:
        query: The question/instruction for the sub-call
        context_content: The context to include in the prompt
        model: Ollama model to use
        tools: Optional list of tool definitions (only include when max_depth > current_depth)
        state: Recursion tracking state

    Returns:
        Tuple of (result, error, updated_state)
    """
```

**Request construction**:
```python
{
    "model": model,
    "messages": [{"role": "user", "content": f"{query}\n\nContext:\n{context_content}"}],
    "tools": tools_array if recursive else None,
    "stream": False,
}
```

Only include `tools` array when `max_depth > current_depth` to allow further recursion.

---

## 3. Implementation Steps

### 3.1 Add RecursionState Dataclass

```python
from dataclasses import dataclass, field

@dataclass
class RecursionState:
    current_depth: int = 0
    max_depth: int = 3
    call_trace: list[str] = field(default_factory=list)

    def can_recurse(self) -> bool:
        return self.current_depth < self.max_depth

    def descend(self, call_id: str) -> "RecursionState":
        return RecursionState(
            current_depth=self.current_depth + 1,
            max_depth=self.max_depth,
            call_trace=self.call_trace + [call_id],
        )
```

### 3.2 Update `_call_ollama()` to Use `/api/chat`

Replace the current `/api/generate` implementation:

```python
async def _call_ollama(
    query: str,
    context_content: str,
    model: str,
    tools: list[dict] | None = None,
    state: RecursionState | None = None,
) -> tuple[Optional[str], Optional[str], RecursionState | None]:
    """Make a sub-call to Ollama using /api/chat endpoint."""
    if not HAS_HTTPX:
        return None, "httpx required for Ollama calls", state

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    # Build request
    request_body = {
        "model": model,
        "messages": [
            {"role": "user", "content": f"{query}\n\nContext:\n{context_content}"}
        ],
        "stream": False,
    }

    # Only include tools if recursion is allowed
    if tools and state and state.can_recurse():
        request_body["tools"] = tools

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{ollama_url}/api/chat",
                json=request_body,
            )
            response.raise_for_status()
            data = response.json()

            message = data.get("message", {})

            # Check for tool calls
            if "tool_calls" in message and message["tool_calls"]:
                # Handle tool calls (agent loop)
                return await _handle_ollama_tool_calls(
                    client, ollama_url, model, message, tools, state
                )

            return message.get("content", ""), None, state

    except Exception as e:
        return None, str(e), state
```

### 3.3 Add Tool Call Handler for Ollama

```python
async def _handle_ollama_tool_calls(
    client: httpx.AsyncClient,
    ollama_url: str,
    model: str,
    message: dict,
    tools: list[dict],
    state: RecursionState,
    max_iterations: int = 5,
) -> tuple[Optional[str], Optional[str], RecursionState]:
    """Handle Ollama tool calls in an agent loop."""
    messages = [message]
    current_state = state.descend("ollama_tool_loop") if state else None

    for _ in range(max_iterations):
        tool_calls = messages[-1].get("tool_calls", [])
        if not tool_calls:
            break

        # Execute each tool call
        for tool_call in tool_calls:
            func = tool_call.get("function", {})
            tool_name = func.get("name")
            tool_args = json.loads(func.get("arguments", "{}"))

            # Execute the tool
            handler = TOOL_HANDLERS.get(tool_name)
            if handler:
                result = await handler(tool_args)
                result_text = result[0].text if result else "Error"
            else:
                result_text = f"Unknown tool: {tool_name}"

            # Append tool result
            messages.append({
                "role": "tool",
                "content": result_text,
            })

        # Continue conversation
        response = await client.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "tools": tools if current_state and current_state.can_recurse() else None,
                "stream": False,
            },
        )
        response.raise_for_status()
        messages.append(response.json().get("message", {}))

    # Return final content
    final_content = messages[-1].get("content", "")
    return final_content, None, current_state
```

### 3.4 Update `_call_claude_sdk()` for Recursion

```python
async def _call_claude_sdk(
    query: str,
    context_content: str,
    model: str,
    tools: list[dict] | None = None,
    state: RecursionState | None = None,
) -> tuple[Optional[str], Optional[str], RecursionState | None]:
    """Make a sub-call to Claude SDK with optional RLM tools."""
    if not HAS_CLAUDE_SDK:
        return None, "claude-agent-sdk required for claude-sdk provider", state

    try:
        prompt = f"{query}\n\nContext:\n{context_content}"

        # Configure options with tools if recursion allowed
        options = ClaudeAgentOptions(
            max_turns=5 if (tools and state and state.can_recurse()) else 1,
            tools=tools if (tools and state and state.can_recurse()) else None,
        )

        texts = []
        async for message in claude_query(prompt=prompt, options=options):
            if hasattr(message, "content"):
                content = message.content
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "text"):
                            texts.append(block.text)
                elif hasattr(content, "text"):
                    texts.append(content.text)
                else:
                    texts.append(str(content))

        result = "\n".join(texts) if texts else ""
        new_state = state.descend("claude_sdk") if state else None
        return result, None, new_state

    except Exception as e:
        return None, str(e), state
```

### 3.5 Generate Tool Definitions for Sub-Calls

```python
def _get_rlm_tools_for_subcall() -> list[dict]:
    """Generate tool definitions in OpenAI-compatible format for sub-calls."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,
            }
        }
        for tool in TOOL_DEFINITIONS
        if tool.name in ["rlm_chunk_context", "rlm_get_chunk",
                          "rlm_filter_context", "rlm_sub_query",
                          "rlm_inspect_context", "rlm_list_contexts"]
    ]
```

### 3.6 Update `_handle_sub_query`

```python
async def _handle_sub_query(arguments: dict) -> list[TextContent]:
    """Make a sub-LLM call with optional recursive depth."""
    query = arguments["query"]
    ctx_name = arguments["context_name"]
    chunk_index = arguments.get("chunk_index")
    provider = arguments.get("provider", "ollama")
    model = arguments.get("model") or DEFAULT_MODELS.get(provider, "olmo-3.1:32b")
    max_depth = arguments.get("max_depth", 0)  # NEW: 0 = no recursion

    # ... existing context loading code ...

    # Build recursion state
    state = RecursionState(max_depth=max_depth) if max_depth > 0 else None
    tools = _get_rlm_tools_for_subcall() if max_depth > 0 else None

    result, error, final_state = await _make_provider_call(
        provider, model, query, context_content, tools, state
    )

    response = {
        "provider": provider,
        "model": model,
        "response": result,
    }

    if final_state:
        response["recursion"] = {
            "max_depth": final_state.max_depth,
            "final_depth": final_state.current_depth,
            "call_trace": final_state.call_trace,
        }

    return _text_response(response)
```

---

## 4. Tool Schema Updates

Add `max_depth` parameter to `rlm_sub_query` and `rlm_sub_query_batch`:

```python
Tool(
    name="rlm_sub_query",
    description="Make a sub-LLM call on a chunk or filtered context. Core of recursive pattern.",
    inputSchema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Question/instruction for the sub-call"},
            "context_name": {"type": "string", "description": "Context identifier to query against"},
            "chunk_index": {"type": "integer", "description": "Optional: specific chunk index"},
            "provider": PROVIDER_SCHEMA,
            "model": {"type": "string", "description": "Model to use"},
            "max_depth": {
                "type": "integer",
                "description": "Maximum recursion depth (0 = no recursion, default)",
                "default": 0,
                "minimum": 0,
                "maximum": 5,
            },
        },
        "required": ["query", "context_name"],
    },
),
```

---

## 5. Risk Assessment

### 5.1 Model Tool Support Varies

**Risk**: Not all Ollama models support tool calling well. Some may ignore tools, hallucinate tool calls, or produce malformed JSON.

**Mitigation**:
- **Recommended model**: `olmo-3.1:32b` (64K context, strong reasoning, tool support)
- Alternative models: `llama3.1:70b`, `qwen2.5:32b`, `mistral-large`
- Graceful fallback if tool call parsing fails
- Validate tool call JSON before execution
- Log warnings for unsupported models

### 5.2 Runaway Recursion

**Risk**: Deep or infinite recursion could exhaust resources.

**Mitigation**:
- Hard cap at `max_depth=5`
- Track call trace for debugging
- Timeout per sub-call (existing 180s)
- Total operation timeout (new: 600s)

### 5.3 Context Explosion

**Risk**: Each recursive level adds context, potentially exceeding model limits.

**Mitigation**:
- Summarize intermediate results before passing down
- Limit tool output size in recursive calls
- Consider token budget tracking

### 5.4 Cost Amplification

**Risk**: Recursive claude-sdk calls could be expensive.

**Mitigation**:
- Default to `max_depth=0` (no recursion)
- Prefer ollama for recursive calls (cost = $0)
- Warn user when recursive claude-sdk calls exceed threshold

---

## 6. Testing Plan

### 6.1 Unit Tests

- `test_recursion_state_tracking`: Verify depth increments correctly
- `test_max_depth_respected`: Ensure calls stop at max_depth
- `test_tool_definition_generation`: Validate OpenAI-compatible format
- `test_ollama_chat_endpoint`: Verify `/api/chat` request format
- `test_ollama_tool_call_parsing`: Parse tool_calls from response

### 6.2 Integration Tests

- `test_single_level_recursion`: Sub-query that chunks and sub-queries once
- `test_multi_level_recursion`: 3-level deep analysis
- `test_recursion_with_claude_sdk`: Verify Claude agent handles tools
- `test_recursion_with_ollama`: Verify Ollama tool loop works
- `test_mixed_provider_recursion`: Start with claude-sdk, recurse to ollama

### 6.3 Edge Cases

- Tool call with invalid arguments
- Tool call to non-existent context
- Timeout during recursive call
- Model that doesn't support tools

---

## 7. Summary

This plan enables true recursive language model patterns by:

1. **Unified endpoint**: Switch Ollama from `/api/generate` to `/api/chat` for all calls
2. **Tool passing**: Both `claude-sdk` and `ollama` providers support passing RLM tools to sub-calls
3. **Depth tracking**: `RecursionState` prevents runaway recursion
4. **Graceful degradation**: Models that don't support tools still work (just non-recursive)
5. **Opt-in recursion**: Default `max_depth=0` maintains backward compatibility

The implementation allows patterns like:
```
Claude (main) -> rlm_sub_query(max_depth=2)
    -> Ollama (depth 1) -> rlm_chunk_context -> rlm_sub_query
        -> Ollama (depth 2) -> final answer
```

This matches the paper's vision of recursive decomposition while maintaining practical safeguards.

---

## 8. Open Questions

1. Should recursive calls share the same context store, or get isolated copies?
2. How to surface intermediate results for debugging?
3. Should we implement streaming for long recursive chains?
4. Consider adding `rlm_sub_query_recursive` as a separate tool vs. extending existing?

---

## 9. Future Enhancement: Recursive Multi-Model Tiering

### Concept

Allow different providers/models at different recursion depths for optimal cost/speed balance:

```
rlm_sub_query(provider="claude-sdk", recursive_provider="ollama", max_depth=2)
  └─ depth 0: claude-sdk (Haiku) ← fast orchestration
      └─ depth 1: ollama (olmo-3.1) ← free deep analysis
          └─ depth 2: ollama (olmo-3.1) ← free deep analysis
```

### Benefits

| Tier | Provider | Cost | Speed | Role |
|------|----------|------|-------|------|
| Top (depth 0) | Haiku | ~$0.80/1M | Fast | Orchestration, decisions |
| Deep (depth 1+) | Ollama | $0 | Slower | Heavy lifting, bulk analysis |

### Proposed Parameters

**Option A: Simple (Recommended)**
```python
rlm_sub_query(
    query="...",
    context_name="...",
    provider="claude-sdk",           # Top-level provider
    recursive_provider="ollama",     # Provider for depth > 0
    recursive_model="olmo-3.1:32b",  # Model for depth > 0
    max_depth=2
)
```

**Option B: Per-Depth Configuration**
```python
rlm_sub_query(
    query="...",
    context_name="...",
    model_tiers={
        0: {"provider": "claude-sdk", "model": "claude-haiku-4-5-20251101"},
        1: {"provider": "ollama", "model": "olmo-3.1:32b"},
        2: {"provider": "ollama", "model": "gemma3:27b"},
    },
    max_depth=2
)
```

**Option C: Preset Strategies**
```python
rlm_sub_query(
    query="...",
    context_name="...",
    tier_strategy="cost-optimized",  # or "speed-optimized", "quality-first"
    max_depth=2
)
```

### Implementation Notes

1. Extend `RecursionState` to track which provider/model to use at each depth
2. Modify `_handle_ollama_tool_calls` and `_handle_sub_query` to check depth and select provider
3. Default behavior: same provider at all depths (backward compatible)
4. New behavior: switch provider when `recursive_provider` is specified

### Use Cases

1. **Cost-optimized analysis**: Haiku decides what to analyze, Ollama does the heavy lifting
2. **Speed-optimized**: Haiku at all levels for fastest response
3. **Quality-first**: Sonnet/Opus at top, Haiku for bulk work
4. **Fully local**: Ollama at all levels for $0 cost

### Priority

Medium - implement after core recursion is validated in production.

---

## Changelog

- **2026-01-17**: Added future enhancement section for Recursive Multi-Model Tiering
- **2026-01-17**: Initial plan created
- **2026-01-17**: Updated provider matrix - both providers now support recursive calls
- **2026-01-17**: Added Ollama `/api/chat` endpoint details and tool calling format
- **2026-01-17**: Updated `_call_ollama()` signature with tools and state parameters
- **2026-01-17**: Revised risk assessment - removed "Ollama doesn't support tools" risk, added "Model tool support varies"
- **2026-01-17**: Set recommended Ollama model to `olmo-3.1:32b` (64K context, strong reasoning)
