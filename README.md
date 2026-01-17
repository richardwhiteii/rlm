# RLM MCP Server

Recursive Language Model patterns for Claude Code — handle massive contexts (10M+ tokens) by treating them as external variables.

Based on: https://arxiv.org/html/2512.24601v1

## Core Idea

Instead of feeding massive contexts directly into the LLM:
1. **Load** context as external variable (stays out of prompt)
2. **Inspect** structure programmatically
3. **Chunk** strategically (lines, chars, or paragraphs)
4. **Sub-query** recursively on chunks
5. **Aggregate** results for final synthesis

## Quick Start

### Installation

```bash
git clone https://github.com/richardwhiteii/rlm.git
cd rlm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Wire to Claude Code

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "rlm": {
      "command": "/path/to/rlm/.venv/bin/python",
      "args": ["-m", "src.rlm_mcp_server"],
      "cwd": "/path/to/rlm",
      "env": {
        "RLM_DATA_DIR": "/path/to/data/rlm"
      }
    }
  }
}
```

Replace `/path/to/rlm` with your actual installation path.

## Tools

| Tool | Purpose |
|------|---------|
| `rlm_load_context` | Load context as external variable |
| `rlm_inspect_context` | Get structure info without loading into prompt |
| `rlm_chunk_context` | Chunk by lines/chars/paragraphs |
| `rlm_get_chunk` | Retrieve specific chunk |
| `rlm_filter_context` | Filter with regex (keep/remove matching lines) |
| `rlm_sub_query` | Make sub-LLM call on chunk |
| `rlm_sub_query_batch` | Process multiple chunks in parallel |
| `rlm_store_result` | Store sub-call result for aggregation |
| `rlm_get_results` | Retrieve stored results |
| `rlm_list_contexts` | List all loaded contexts |

## Providers

By default, sub-queries use **Claude Haiku 4.5** via the Claude Agent SDK. This works out-of-the-box if you have a Claude API key configured.

| Provider | Default Model | Cost | Use Case |
|----------|--------------|------|----------|
| `claude-sdk` | claude-haiku-4-5 | ~$0.80/1M input | Default, works everywhere |
| `ollama` | gemma3:27b | $0 | Local inference, requires Ollama |

## Autonomous Usage

Enable Claude to use RLM tools automatically without manual invocation:

**1. CLAUDE.md Integration**
Copy `CLAUDE.md.example` content to your project's `CLAUDE.md` (or `~/.claude/CLAUDE.md` for global) to teach Claude when to reach for RLM tools automatically.

**2. Hook Installation**
Copy the `.claude/hooks/` directory to your project to auto-suggest RLM when reading files >10KB:
```bash
cp -r .claude/hooks/ /path/to/your-project/.claude/hooks/
```
The hook provides guidance but doesn't block reads.

**3. Skill Reference**
Copy the `.claude/skills/` directory for comprehensive RLM guidance:
```bash
cp -r .claude/skills/ /path/to/your-project/.claude/skills/
```

With these in place, Claude will autonomously detect when to use RLM instead of reading large files directly into context.

### Using Ollama (Free Local Inference)

If you have [Ollama](https://ollama.ai) installed locally, you can run sub-queries at zero cost:

1. **Install Ollama** and pull a model:
   ```bash
   ollama pull gemma3:27b
   ```

2. **Add Ollama URL** to your MCP config:
   ```json
   {
     "mcpServers": {
       "rlm": {
         "command": "/path/to/rlm/.venv/bin/python",
         "args": ["-m", "src.rlm_mcp_server"],
         "cwd": "/path/to/rlm",
         "env": {
           "RLM_DATA_DIR": "/path/to/data/rlm",
           "OLLAMA_URL": "http://localhost:11434"
         }
       }
     }
   }
   ```

3. **Specify provider** in your sub-queries:
   ```
   rlm_sub_query(
       query="Summarize this section",
       context_name="my_doc",
       chunk_index=0,
       provider="ollama"
   )
   ```

Or for batch processing:
```
rlm_sub_query_batch(
    query="Extract key points",
    context_name="my_doc",
    chunk_indices=[0, 1, 2, 3],
    provider="ollama",
    concurrency=4
)
```

## Usage Example

### Basic Pattern

```
# 1. Load a large document
rlm_load_context(name="report", content=<large document>)

# 2. Inspect structure
rlm_inspect_context(name="report", preview_chars=500)

# 3. Chunk into manageable pieces
rlm_chunk_context(name="report", strategy="paragraphs", size=1)

# 4. Sub-query chunks in parallel
rlm_sub_query_batch(
    query="What is the main topic? Reply in one sentence.",
    context_name="report",
    chunk_indices=[0, 1, 2, 3],
    provider="ollama",  # or omit for claude-sdk default
    concurrency=4
)

# 5. Store results for aggregation
rlm_store_result(name="topics", result=<response>)

# 6. Retrieve all results
rlm_get_results(name="topics")
```

### Processing a 2MB Document

Tested with H.R.1 "One Big Beautiful Bill Act" (2MB XML):

```
# Load
rlm_load_context(name="bill", content=<2MB XML>)

# Chunk into 40 pieces (50K chars each)
rlm_chunk_context(name="bill", strategy="chars", size=50000)

# Sample 8 chunks (20%) with parallel queries
rlm_sub_query_batch(
    query="What topics does this section cover?",
    context_name="bill",
    chunk_indices=[0, 5, 10, 15, 20, 25, 30, 35],
    provider="ollama",
    concurrency=4
)
```

Result: Comprehensive topic extraction at $0 cost.

## Data Storage

```
$RLM_DATA_DIR/
├── contexts/     # Raw contexts (.txt + .meta.json)
├── chunks/       # Chunked versions (by context name)
└── results/      # Stored sub-call results (.jsonl)
```

Contexts persist across sessions. Chunked contexts are cached for reuse.

## Architecture

```
Claude Code
    │
    ▼
RLM MCP Server
    │
    ├─► claude-sdk (Haiku 4.5) ─► Anthropic API
    │
    └─► ollama ─► Local LLM (gemma3:27b, llama3, etc.)
```

The key insight: **context stays external**. Instead of stuffing 2MB into your prompt, load it once, chunk it, and make targeted sub-queries. Claude orchestrates; sub-models do the heavy lifting.

## Learning Prompts

Use these prompts with Claude Code to explore the codebase and learn RLM patterns. The code is the single source of truth.

### Understanding the Tools

```
Read src/rlm_mcp_server.py and list all RLM tools with their parameters and purpose.
```

```
Explain the chunking strategies available in rlm_chunk_context.
When would I use each one?
```

```
What's the difference between rlm_sub_query and rlm_sub_query_batch?
Show me the implementation.
```

### Understanding the Architecture

```
Read src/rlm_mcp_server.py and explain how contexts are stored and persisted.
Where does the data live?
```

```
How does the claude-sdk provider extract text from responses?
Walk me through _call_claude_sdk.
```

```
What happens when I call rlm_load_context? Trace the full flow.
```

### Hands-On Learning

```
Load the README as a context, chunk it by paragraphs,
and run a sub-query on the first chunk to summarize it.
```

```
Show me how to process a large file in parallel using rlm_sub_query_batch.
Use a real example.
```

```
I have a 1MB log file. Walk me through the RLM pattern to extract all errors.
```

### Extending RLM

```
Read the test file and explain what scenarios are covered.
What edge cases should I be aware of?
```

```
How would I add a new chunking strategy (e.g., by regex delimiter)?
Show me where to modify the code.
```

```
How would I add a new provider (e.g., OpenAI)?
What functions need to change?
```

## License

MIT
