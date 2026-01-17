# RLM MCP Server

Recursive Language Model patterns for Claude Code — handle massive contexts (10M+ tokens) by treating them as external variables.

Based on: https://arxiv.org/html/2512.24601v1

## Core Idea

Instead of feeding massive contexts directly into the LLM:
1. Load context as external variable
2. Inspect structure programmatically
3. Chunk strategically
4. Sub-call recursively on chunks
5. Aggregate results

## Tools

| Tool | Purpose |
|------|---------|
| `rlm_load_context` | Load context as external variable |
| `rlm_inspect_context` | Get structure info without loading into prompt |
| `rlm_chunk_context` | Chunk by lines/chars/paragraphs/regex |
| `rlm_get_chunk` | Retrieve specific chunk |
| `rlm_filter_context` | Filter with regex (keep/remove matching lines) |
| `rlm_sub_query` | Make sub-LLM call on chunk (Anthropic or Ollama) |
| `rlm_store_result` | Store sub-call result for aggregation |
| `rlm_get_results` | Retrieve stored results |
| `rlm_list_contexts` | List all loaded contexts |

## Setup

```bash
cd /Users/richard/projects/fun/rlm
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Wire to Claude Code

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "rlm": {
      "command": "/Users/richard/projects/fun/rlm/.venv/bin/python",
      "args": ["-m", "src.server"],
      "cwd": "/Users/richard/projects/fun/rlm",
      "env": {
        "RLM_DATA_DIR": "/Users/richard/data/rlm",
        "OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

## Usage Example

```
1. Load a massive codebase:
   rlm_load_context(name="repo", content=<all files concatenated>)

2. Inspect structure:
   rlm_inspect_context(name="repo", preview_chars=1000)

3. Chunk by files (using regex):
   rlm_chunk_context(name="repo", strategy="regex", pattern="^### FILE: ")

4. Sub-query each chunk:
   for i in range(chunk_count):
       result = rlm_sub_query(
           query="Extract all function signatures",
           context_name="repo",
           chunk_index=i,
           provider="ollama",  # Free local inference
           model="llama3.2"
       )
       rlm_store_result(name="signatures", result=result)

5. Aggregate results:
   results = rlm_get_results(name="signatures")
   final = rlm_sub_query(
       query="Synthesize a complete API reference",
       context_name="signatures_combined"
   )
```

## Data Storage

```
$RLM_DATA_DIR/
├── contexts/     # Raw contexts (.txt + .meta.json)
├── chunks/       # Chunked versions (by context name)
└── results/      # Stored sub-call results (.jsonl)
```

Contexts persist across sessions. Cache chunked codebases for reuse.

## Providers

- **anthropic**: Uses Claude API (default)
- **ollama**: Local inference, cost = $0

Set `OLLAMA_URL` env var if not localhost:11434.
