# RLM MCP Server

Recursive Language Model patterns for Claude Code — handle massive contexts (10M+ tokens) by treating them as external variables.

Based on: https://arxiv.org/html/2512.24601v1

## How Users Interact

You don't call RLM tools directly. You ask Claude to analyze large files, and Claude uses RLM behind the scenes.

**Example:**
- You say: "Analyze this 2MB log file for errors"
- Claude uses RLM tools internally
- You get: "I found 3 error patterns: database timeouts (47), auth failures (23)..."

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
uv sync
```

Or with pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure for Claude Code

First, find your installation path:
```bash
cd rlm && pwd
# Example output: /Users/your_username/projects/rlm
```

Add to `~/.claude/.mcp.json`, replacing the example paths with your own:

```json
{
  "mcpServers": {
    "rlm": {
      "command": "uv",
      "args": ["run", "--directory", "/Users/your_username/projects/rlm", "python", "-m", "src.rlm_mcp_server"],
      "env": {
        "RLM_DATA_DIR": "/Users/your_username/.rlm-data"
      }
    }
  }
}
```

> **Note**: Replace `/Users/your_username/projects/rlm` with the output from `pwd` above. The `RLM_DATA_DIR` is where RLM stores contexts and results — you can use any directory you prefer.

**Alternative** (if not using uv):
```json
{
  "mcpServers": {
    "rlm": {
      "command": "/Users/your_username/projects/rlm/.venv/bin/python",
      "args": ["-m", "src.rlm_mcp_server"],
      "cwd": "/Users/your_username/projects/rlm",
      "env": {
        "RLM_DATA_DIR": "/Users/your_username/.rlm-data"
      }
    }
  }
}
```

## Enable Auto-Detection

Enable Claude to use RLM tools automatically without manual invocation:

**1. CLAUDE.md Integration**
Copy `CLAUDE.md.example` content to your project's `CLAUDE.md` (or `~/.claude/CLAUDE.md` for global) to teach Claude when to reach for RLM tools automatically.

**2. Hook Installation**
Copy the `.claude/hooks/` directory to your project to auto-suggest RLM when reading files >25KB:
```bash
cp -r .claude/hooks/ /Users/your_username/your-project/.claude/hooks/
```
The hook provides guidance but doesn't block reads.

**3. Skill Reference**
Copy the `.claude/skills/` directory for comprehensive RLM guidance:
```bash
cp -r .claude/skills/ /Users/your_username/your-project/.claude/skills/
```

With these in place, Claude will autonomously detect when to use RLM instead of reading large files directly into context.

## Tools

> These tools are used by Claude internally when processing large contexts. You don't call them directly—you just ask Claude to analyze large files.

| Tool | Purpose |
|------|---------|
| `rlm_auto_analyze` | **One-step analysis** — auto-detects type, chunks, and queries |
| `rlm_load_context` | Load context as external variable |
| `rlm_inspect_context` | Get structure info without loading into prompt |
| `rlm_chunk_context` | Chunk by lines/chars/paragraphs |
| `rlm_get_chunk` | Retrieve specific chunk |
| `rlm_filter_context` | Filter with regex (keep/remove matching lines) |
| `rlm_exec` | Execute Python code against loaded context (sandboxed) |
| `rlm_sub_query` | Make sub-LLM call on chunk |
| `rlm_sub_query_batch` | Process multiple chunks in parallel |
| `rlm_store_result` | Store sub-call result for aggregation |
| `rlm_get_results` | Retrieve stored results |
| `rlm_list_contexts` | List all loaded contexts |

### Quick Analysis with `rlm_auto_analyze`

For most use cases, Claude uses `rlm_auto_analyze` — it handles everything automatically:

```python
rlm_auto_analyze(
    name="my_file",
    content=file_content,
    goal="find_bugs"  # or: summarize, extract_structure, security_audit, answer:<question>
)
```

**What it does automatically:**
1. Detects content type (Python, JSON, Markdown, logs, prose, code)
2. Selects optimal chunking strategy
3. Adapts the query for the content type
4. Runs parallel sub-queries
5. Returns aggregated results

**Supported goals:**

| Goal | Description |
|------|-------------|
| `summarize` | Summarize content purpose and key points |
| `find_bugs` | Identify errors, issues, potential problems |
| `extract_structure` | List functions, classes, schema, headings |
| `security_audit` | Find vulnerabilities and security issues |
| `answer:<question>` | Answer a custom question about the content |

### Programmatic Analysis with `rlm_exec`

For deterministic pattern matching and data extraction, Claude can use `rlm_exec` to run Python code directly against a loaded context. This is closer to the paper's REPL approach and provides full control over analysis logic.

**Tool**: `rlm_exec`

**Purpose**: Execute arbitrary Python code against a loaded context in a sandboxed subprocess.

**Parameters**:
- `code` (required): Python code to execute. Set the `result` variable to capture output.
- `context_name` (required): Name of a previously loaded context.
- `timeout` (optional, default 30): Maximum execution time in seconds.

**Features**:
- Context available as read-only `context` variable
- Pre-imported modules: `re`, `json`, `collections`
- Subprocess isolation (won't crash the server)
- Timeout enforcement
- Works on any system with Python (no Docker needed)

**Example — Finding patterns in a loaded context**:

```python
# After loading a context
rlm_exec(
    code="""
import re
amounts = re.findall(r'\$[\d,]+', context)
result = {'count': len(amounts), 'sample': amounts[:5]}
""",
    context_name="bill"
)
```

**Example Response**:

```json
{
  "result": {
    "count": 1247,
    "sample": ["$500", "$1,000", "$250,000", "$100,000", "$50"]
  },
  "stdout": "",
  "stderr": "",
  "return_code": 0,
  "timed_out": false
}
```

**Example — Extracting structured data**:

```python
rlm_exec(
    code="""
import re
import json

# Find all email addresses
emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', context)

# Count by domain
from collections import Counter
domains = [e.split('@')[1] for e in emails]
domain_counts = Counter(domains)

result = {
    'total_emails': len(emails),
    'unique_domains': len(domain_counts),
    'top_domains': domain_counts.most_common(5)
}
""",
    context_name="dataset",
    timeout=60
)
```

**When to use `rlm_exec` vs `rlm_sub_query`**:

| Use Case | Tool | Why |
|----------|------|-----|
| Extract all dates, IDs, amounts | `rlm_exec` | Regex is deterministic and fast |
| Find security vulnerabilities | `rlm_sub_query` | Requires reasoning and context |
| Parse JSON/XML structure | `rlm_exec` | Standard libraries work perfectly |
| Summarize themes or tone | `rlm_sub_query` | Natural language understanding needed |
| Count word frequencies | `rlm_exec` | Simple computation, no AI needed |
| Answer "Why did X happen?" | `rlm_sub_query` | Requires inference and reasoning |

**Tip**: For large contexts, combine both — use `rlm_exec` to filter/extract, then `rlm_sub_query` for semantic analysis of filtered results.

## Providers

By default, sub-queries use **Claude Haiku 4.5** via the Claude Agent SDK. This works out-of-the-box if you have a Claude API key configured.

| Provider | Default Model | Cost | Use Case |
|----------|--------------|------|----------|
| `claude-sdk` | claude-haiku-4-5 | ~$0.80/1M input | Default, works everywhere |
| `ollama` | olmo-3.1:32b | $0 | Local inference, requires Ollama |

## Recursive Sub-Queries

The `rlm_sub_query` and `rlm_sub_query_batch` tools support hierarchical decomposition via the `max_depth` parameter:

- `max_depth=0` (default): Flat call, no recursion
- `max_depth=1-5`: Sub-LLM can use RLM tools (chunk, filter, sub_query, etc.)

Example: Analyzing a massive codebase with 2-level recursion:
```python
rlm_sub_query(
    query="Find all security vulnerabilities",
    context_name="codebase",
    chunk_index=0,
    max_depth=2  # Allow sub-queries to further decompose
)
```

**How it works:**
1. When `max_depth > 0`, the sub-LLM receives RLM tools in its function calling context
2. If the sub-LLM decides to use a tool (e.g., `rlm_chunk_context`), the agent loop handles it
3. Each recursive call decrements the depth limit until `max_depth` is reached
4. The response includes recursion metadata: `depth_reached` and `call_trace`

**Recommended for recursive calls**: Use a local model like `gemma3:27b` via Ollama to avoid cost escalation from deep recursion.

## Using Ollama (Free Local Inference)

With [Ollama](https://ollama.ai) installed locally, you can run sub-queries at zero cost:

1. **Install Ollama** and pull a model:
   ```bash
   ollama pull gemma3:27b
   ```

2. **Add Ollama URL** to your MCP config:
   ```json
   {
     "mcpServers": {
       "rlm": {
         "command": "uv",
         "args": ["run", "--directory", "/Users/your_username/projects/rlm", "python", "-m", "src.rlm_mcp_server"],
         "env": {
           "RLM_DATA_DIR": "/Users/your_username/.rlm-data",
           "OLLAMA_URL": "http://localhost:11434"
         }
       }
     }
   }
   ```

3. **Specify provider** in sub-queries:
   ```
   rlm_sub_query(
       query="Summarize this section",
       context_name="my_doc",
       chunk_index=0,
       provider="ollama"  # Use local Ollama instead of default claude-sdk
   )
   ```

Or for batch processing:
```
rlm_sub_query_batch(
    query="Extract key points",
    context_name="my_doc",
    chunk_indices=[0, 1, 2, 3],
    provider="ollama",  # Use local Ollama instead of default claude-sdk
    concurrency=4
)
```

## Usage Examples

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
    concurrency=4  # uses claude-sdk by default
)

# 5. Store results for aggregation
rlm_store_result(name="topics", result=<response>)

# 6. Retrieve all results
rlm_get_results(name="topics")
```

### Analyzing Encyclopedia Britannica (11MB)

The flagship example of RLM capabilities — processing the Encyclopedia Britannica, 11th Edition from Project Gutenberg:

```python
# Load the full encyclopedia (11MB, ~2M tokens)
content = open("docs/encyclopedia/merged_encyclopedia.txt").read()
rlm_load_context(name="encyclopedia", content=content)

# Inspect
rlm_inspect_context(name="encyclopedia")
# → 11MB, 184K lines, ~2M tokens

# Chunk for processing
rlm_chunk_context(name="encyclopedia", strategy="paragraphs", size=30)

# Query across the corpus
rlm_sub_query_batch(
    query="Summarize the main topics in this section",
    context_name="encyclopedia",
    chunk_indices=[0, 50, 100, 150],
    provider="claude-sdk"  # or "ollama" for free local inference
)
```

**Extract Topic Catalog**:

```python
# Filter for specific subject
rlm_filter_context(
    name="encyclopedia",
    output_name="botany",
    pattern="(?i)(botan|plant|flora|flower|genus)",
    mode="keep"
)

# Analyze filtered content
rlm_auto_analyze(
    name="botany_analysis",
    content=filtered_content,
    goal="answer:List all botanical articles with brief descriptions"
)
```

| Metric | Value |
|--------|-------|
| File size | 11 MB |
| Lines | 184,000 |
| Tokens | ~2M |
| Processing cost | $0 (Ollama) or ~$1.60 (Haiku) |

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

The key insight: **context stays external, not in your prompt**. Claude orchestrates; sub-models analyze.

## For Contributors: Learning the Codebase

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

## Test Corpus: Encyclopedia Britannica

The repository includes excerpts from the **Encyclopedia Britannica, 11th Edition** (1910-1911) from Project Gutenberg for testing RLM capabilities on large reference documents.

### Included Files

| File | Size | Description |
|------|------|-------------|
| `docs/encyclopedia/merged_encyclopedia.txt` | 11MB | All slices merged (~2M tokens) |

### Using for Testing

```python
# Load the full encyclopedia
content = open("docs/encyclopedia/merged_encyclopedia.txt").read()
rlm_load_context(name="encyclopedia", content=content)

# Inspect
rlm_inspect_context(name="encyclopedia")
# → 11MB, 184K lines, ~2M tokens

# Chunk for processing
rlm_chunk_context(name="encyclopedia", strategy="paragraphs", size=30)

# Query across the corpus
rlm_sub_query_batch(
    query="Summarize the main topics in this section",
    context_name="encyclopedia",
    chunk_indices=[0, 50, 100, 150],
    provider="claude-sdk"  # or "ollama" for free local inference
)
```

### Example: Extract Topic Catalog

```python
# Filter for specific subject
rlm_filter_context(
    name="encyclopedia",
    output_name="botany",
    pattern="(?i)(botan|plant|flora|flower|genus)",
    mode="keep"
)

# Analyze filtered content
rlm_auto_analyze(
    name="botany_analysis",
    content=filtered_content,
    goal="answer:List all botanical articles with brief descriptions"
)
```

### Download More Slices

Additional encyclopedia volumes available from Project Gutenberg:
- Search: https://www.gutenberg.org/ebooks/search/?query=encyclopaedia+britannica+11th
- ~130 slices available covering A-Z

### Attribution

Encyclopedia Britannica, 11th Edition (1910-1911) sourced from [Project Gutenberg](https://www.gutenberg.org/). Public domain.

## License

MIT
