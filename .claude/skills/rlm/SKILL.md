# RLM (Recursive Language Model) Skill

## Overview

The RLM pattern enables processing massive contexts (10M+ tokens) that exceed Claude's context window by recursively chunking, processing, and aggregating results. Instead of failing on large files, use RLM to break them into manageable pieces.

## When to Use RLM

Use RLM when you encounter:

- **Large files**: Any file >100KB or >2000 lines
- **Multi-file analysis**: Processing multiple files together (combined size matters)
- **Context exceeded**: User asks to analyze content that won't fit in context window
- **Aggregation tasks**: Summarizing logs, finding patterns across large datasets, counting/filtering operations
- **Deep codebase analysis**: Understanding architecture across many files
- **Document processing**: Analyzing reports, research papers, documentation sets

**Don't use RLM for:**
- Small files (<100KB)
- Single-pass tasks that fit in context
- Interactive editing (use standard tools)

## The RLM Pattern

The core workflow is: **Load → Inspect → Chunk → Sub-Query → Aggregate**

### Step 1: Load Context
```bash
# Load large content into RLM memory
rlm_load_context(
  name="codebase",
  content=file_contents  # Full file content
)
```

Returns: `{name, size_bytes, size_chars, line_count, loaded: true}`

### Step 2: Inspect Context
```bash
# Understand structure without loading into prompt
rlm_inspect_context(
  name="codebase",
  preview_chars=500  # Optional preview
)
```

Returns: Metadata + preview (first N chars)

### Step 3: Chunk Context
```bash
# Break into manageable pieces
rlm_chunk_context(
  name="codebase",
  strategy="lines",  # or "chars" or "paragraphs"
  size=100  # Lines per chunk (or chars if strategy=chars)
)
```

**Chunking Strategies:**
- `lines` (default): Split by line count - best for code, logs, structured data
- `chars`: Split by character count - best for prose, unstructured text
- `paragraphs`: Split by blank lines - best for documents, markdown

Returns: `{name, chunk_count, strategy, size_per_chunk}`

### Step 4: Sub-Query (Process Chunks)

**Single chunk:**
```bash
rlm_sub_query(
  context_name="codebase",
  chunk_index=0,
  query="Extract all function names",
  provider="claude-sdk"  # or "ollama"
)
```

**Batch processing (parallel):**
```bash
rlm_sub_query_batch(
  context_name="codebase",
  chunk_indices=[0, 1, 2, 3],
  query="Extract all function names",
  provider="claude-sdk",
  concurrency=4  # Max parallel requests (max: 8)
)
```

Returns: Array of results, one per chunk

### Step 5: Store Results (Optional)
```bash
# Store intermediate results for later aggregation
rlm_store_result(
  name="function_names",
  result=sub_query_response,
  metadata={"chunk": 0}  # Optional
)
```

### Step 6: Aggregate
```bash
# Retrieve all stored results
rlm_get_results(name="function_names")
```

Then synthesize final answer from all chunk results.

## Provider Options

### claude-sdk (Default - Recommended)
- Model: **Haiku 4.5** (fast, accurate, cost-effective)
- Cost: ~$0.25 per 1M input tokens
- Best for: Most tasks requiring accuracy
- Usage: `provider="claude-sdk"`

### ollama (Local, Free)
- Model: User's local Ollama instance
- Cost: Free (runs on your hardware)
- Best for: Experimentation, privacy-sensitive data, budget constraints
- Usage: `provider="ollama"`

**Choosing a Provider:**
- Default to `claude-sdk` for production tasks
- Use `ollama` when cost/privacy is primary concern
- Haiku 4.5 is fast enough for batch processing

## Example Workflows

### Workflow 1: Analyze Large Codebase

**Task:** Find all TODO comments across 50 Python files

```python
# 1. Read all files and combine
all_code = ""
for file in python_files:
    all_code += read_file(file)

# 2. Load into RLM
rlm_load_context(name="codebase", content=all_code)

# 3. Inspect to understand size
metadata = rlm_inspect_context(name="codebase")
# Shows: 15,000 lines, 500KB

# 4. Chunk by lines (code is line-oriented)
rlm_chunk_context(
    name="codebase",
    strategy="lines",
    size=200  # 200 lines per chunk
)
# Result: 75 chunks

# 5. Process in batches
for batch_start in range(0, 75, 4):
    batch_indices = list(range(batch_start, min(batch_start+4, 75)))
    results = rlm_sub_query_batch(
        context_name="codebase",
        chunk_indices=batch_indices,
        query="Extract all TODO comments with line context",
        concurrency=4
    )

    # 6. Store results
    for i, result in enumerate(results):
        rlm_store_result(
            name="todos",
            result=result,
            metadata={"chunk": batch_start + i}
        )

# 7. Aggregate
all_results = rlm_get_results(name="todos")
# Synthesize final TODO list
```

### Workflow 2: Process Large Log File

**Task:** Summarize errors from 10MB log file

```python
# 1. Load logs
logs = read_file("/var/log/app.log")
rlm_load_context(name="logs", content=logs)

# 2. Chunk by lines (logs are line-oriented)
rlm_chunk_context(name="logs", strategy="lines", size=500)

# 3. Filter to error lines only
rlm_filter_context(
    name="logs",
    output_name="errors",
    pattern=r"ERROR|CRITICAL",
    mode="keep"
)

# 4. Chunk filtered results
rlm_chunk_context(name="errors", strategy="lines", size=100)

# 5. Batch process errors
chunk_metadata = rlm_inspect_context(name="errors")
num_chunks = chunk_metadata["chunk_count"]

all_indices = list(range(num_chunks))
results = rlm_sub_query_batch(
    context_name="errors",
    chunk_indices=all_indices,
    query="Group errors by type and count occurrences",
    concurrency=8
)

# 6. Aggregate error summary
# Synthesize from results array
```

### Workflow 3: Multi-Document Q&A

**Task:** Answer questions from 20 research papers

```python
# 1. Load all papers
combined_docs = "\n\n=== DOCUMENT BREAK ===\n\n".join(papers)
rlm_load_context(name="research", content=combined_docs)

# 2. Chunk by paragraphs (prose is paragraph-oriented)
rlm_chunk_context(name="research", strategy="paragraphs", size=50)

# 3. Ask question across all chunks
results = rlm_sub_query_batch(
    context_name="research",
    chunk_indices=list(range(chunk_count)),
    query="Does this section mention climate change impacts on agriculture? If yes, summarize key points.",
    concurrency=8
)

# 4. Filter relevant results
relevant = [r for r in results if "yes" in r.lower()]

# 5. Final synthesis
# Combine relevant excerpts into answer
```

## Tool Reference

### rlm_load_context
Load large content into RLM memory without consuming context window.
- `name`: Identifier for this context
- `content`: Full text content to load

### rlm_inspect_context
Get metadata and preview without loading full content.
- `name`: Context identifier
- `preview_chars`: Number of characters to preview (default: 500)

### rlm_chunk_context
Split context into manageable chunks.
- `name`: Context identifier
- `strategy`: `"lines"`, `"chars"`, or `"paragraphs"`
- `size`: Chunk size (meaning depends on strategy)

### rlm_get_chunk
Retrieve specific chunk by index.
- `name`: Context identifier
- `chunk_index`: Zero-based chunk index

### rlm_filter_context
Filter context using regex, creates new filtered context.
- `name`: Source context identifier
- `output_name`: Name for filtered context
- `pattern`: Regex pattern to match
- `mode`: `"keep"` (keep matches) or `"remove"` (remove matches)

### rlm_sub_query
Process single chunk with sub-LLM call.
- `context_name`: Context identifier
- `query`: Question/instruction for sub-call
- `chunk_index`: Optional specific chunk (otherwise uses whole context)
- `provider`: `"claude-sdk"` or `"ollama"`
- `model`: Optional model override

### rlm_sub_query_batch
Process multiple chunks in parallel (recommended).
- `context_name`: Context identifier
- `query`: Question/instruction for each chunk
- `chunk_indices`: Array of chunk indices to process
- `provider`: `"claude-sdk"` or `"ollama"`
- `concurrency`: Max parallel requests (default: 4, max: 8)

### rlm_store_result
Store sub-call result for later aggregation.
- `name`: Result set identifier
- `result`: Result content to store
- `metadata`: Optional metadata about result

### rlm_get_results
Retrieve all stored results for aggregation.
- `name`: Result set identifier

### rlm_list_contexts
List all loaded contexts and their metadata.

## Best Practices

### Chunking Strategy Selection
- **Code/logs/CSV**: Use `lines` (structured, line-oriented)
- **Prose/articles**: Use `paragraphs` (semantic boundaries)
- **Unstructured text**: Use `chars` (uniform distribution)

### Chunk Size Guidelines
- **Lines**: 100-500 (balance between context and granularity)
- **Chars**: 2000-10000 (roughly 500-2500 tokens)
- **Paragraphs**: 20-100 (depends on paragraph length)

### Efficient Processing
1. **Use batch processing**: `rlm_sub_query_batch` is much faster than sequential calls
2. **Set appropriate concurrency**: 4-8 parallel requests balances speed and resource usage
3. **Filter before chunking**: Use `rlm_filter_context` to reduce data volume
4. **Inspect first**: Always check context size before chunking

### Cost Optimization
- Use `claude-sdk` (Haiku 4.5) for most tasks - fast and cheap
- Use `ollama` for experimentation or when processing very large volumes
- Filter contexts before processing to reduce token usage
- Chunk at appropriate granularity (bigger chunks = fewer calls)

## Common Patterns

### Map-Reduce Pattern
```python
# Map: Process each chunk
results = rlm_sub_query_batch(
    context_name="data",
    chunk_indices=all_indices,
    query="Extract key information",
    concurrency=8
)

# Reduce: Aggregate results
final = synthesize(results)
```

### Filter-Process Pattern
```python
# Filter to relevant content
rlm_filter_context(
    name="all_logs",
    output_name="errors",
    pattern="ERROR",
    mode="keep"
)

# Process filtered content
results = rlm_sub_query_batch(
    context_name="errors",
    chunk_indices=all_indices,
    query="Categorize error type"
)
```

### Hierarchical Processing Pattern
```python
# First pass: Summarize each chunk
summaries = rlm_sub_query_batch(
    context_name="docs",
    chunk_indices=all_indices,
    query="Summarize key points"
)

# Second pass: Aggregate summaries
rlm_load_context(name="summaries", content="\n".join(summaries))
final = rlm_sub_query(
    context_name="summaries",
    query="Create overall summary from chunk summaries"
)
```

## Troubleshooting

### "Context too large" errors
- You're trying to process chunks that are still too big
- Solution: Reduce chunk size or filter content first

### Slow processing
- Sequential sub-queries instead of batch
- Solution: Use `rlm_sub_query_batch` with appropriate concurrency

### Poor aggregation quality
- Chunks too small (losing context)
- Solution: Increase chunk size to maintain semantic coherence

### High costs
- Using wrong provider or inefficient chunking
- Solution: Use Haiku 4.5 (`claude-sdk`) and filter before processing

## Summary

RLM unlocks massive context processing for Claude Code:
- ✅ Handle files >100KB easily
- ✅ Process multiple files together
- ✅ Parallelize for speed (batch processing)
- ✅ Cost-effective with Haiku 4.5
- ✅ Flexible chunking strategies
- ✅ Map-reduce pattern for aggregation

**Default workflow:** Load → Inspect → Chunk (lines, 200) → Batch Sub-Query (claude-sdk, concurrency=4) → Aggregate
