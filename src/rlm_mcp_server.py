#!/usr/bin/env python3
"""
RLM MCP Server - Recursive Language Model patterns for massive context handling.

Implements the core insight from https://arxiv.org/html/2512.24601v1:
Treat context as external variable, chunk programmatically, sub-call recursively.
"""

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from claude_agent_sdk import ClaudeAgentOptions, query as claude_query
    HAS_CLAUDE_SDK = True
except ImportError:
    HAS_CLAUDE_SDK = False

# Storage directories
DATA_DIR = Path(os.environ.get("RLM_DATA_DIR", "/tmp/rlm"))
CONTEXTS_DIR = DATA_DIR / "contexts"
CHUNKS_DIR = DATA_DIR / "chunks"
RESULTS_DIR = DATA_DIR / "results"

for directory in [CONTEXTS_DIR, CHUNKS_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# In-memory context storage (also persisted to disk)
contexts: dict[str, dict] = {}

server = Server("rlm")

# Default models per provider
DEFAULT_MODELS = {
    "ollama": "gemma3:27b",
    "claude-sdk": "claude-haiku-4-5-20251101",
}


def _hash_content(content: str) -> str:
    """Create short hash for content identification."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _load_context_from_disk(name: str) -> Optional[dict]:
    """Load context from disk if it exists."""
    meta_path = CONTEXTS_DIR / f"{name}.meta.json"
    content_path = CONTEXTS_DIR / f"{name}.txt"

    if not (meta_path.exists() and content_path.exists()):
        return None

    meta = json.loads(meta_path.read_text())
    meta["content"] = content_path.read_text()
    return meta


def _save_context_to_disk(name: str, content: str, meta: dict) -> None:
    """Persist context to disk."""
    (CONTEXTS_DIR / f"{name}.txt").write_text(content)
    meta_without_content = {k: v for k, v in meta.items() if k != "content"}
    (CONTEXTS_DIR / f"{name}.meta.json").write_text(
        json.dumps(meta_without_content, indent=2)
    )


def _ensure_context_loaded(name: str) -> Optional[str]:
    """Ensure context is loaded into memory. Returns error message if not found."""
    if name in contexts:
        return None

    disk_context = _load_context_from_disk(name)
    if disk_context:
        content = disk_context.pop("content")
        contexts[name] = {"meta": disk_context, "content": content}
        return None

    return f"Context '{name}' not found"


def _text_response(data: Any) -> list[TextContent]:
    """Create a JSON text response."""
    if isinstance(data, str):
        return [TextContent(type="text", text=data)]
    return [TextContent(type="text", text=json.dumps(data, indent=2))]


def _error_response(code: str, message: str) -> list[TextContent]:
    """Create a structured error response."""
    return _text_response({"error": code, "message": message})


async def _call_ollama(query: str, context_content: str, model: str) -> tuple[Optional[str], Optional[str]]:
    """Make a sub-call to Ollama. Returns (result, error)."""
    if not HAS_HTTPX:
        return None, "httpx required for Ollama calls"

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": f"{query}\n\nContext:\n{context_content}",
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json().get("response", ""), None
    except Exception as e:
        return None, str(e)


async def _call_claude_sdk(query: str, context_content: str, model: str) -> tuple[Optional[str], Optional[str]]:
    """Make a sub-call to Claude SDK. Returns (result, error)."""
    if not HAS_CLAUDE_SDK:
        return None, "claude-agent-sdk required for claude-sdk provider"

    try:
        prompt = f"{query}\n\nContext:\n{context_content}"
        options = ClaudeAgentOptions(max_turns=1)

        texts = []
        async for message in claude_query(prompt=prompt, options=options):
            if hasattr(message, "content"):
                content = message.content
                # Extract text from TextBlock objects
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "text"):
                            texts.append(block.text)
                elif hasattr(content, "text"):
                    texts.append(content.text)
                else:
                    texts.append(str(content))

        result = "\n".join(texts) if texts else ""
        return result, None
    except Exception as e:
        return None, str(e)


async def _make_provider_call(
    provider: str,
    model: str,
    query: str,
    context_content: str,
) -> tuple[Optional[str], Optional[str]]:
    """Route a sub-call to the appropriate provider. Returns (result, error)."""
    if provider == "ollama":
        return await _call_ollama(query, context_content, model)
    elif provider == "claude-sdk":
        return await _call_claude_sdk(query, context_content, model)
    else:
        return None, f"Unknown provider: {provider}"


def _chunk_content(content: str, strategy: str, size: int) -> list[str]:
    """Chunk content using the specified strategy."""
    if strategy == "lines":
        lines = content.split("\n")
        return ["\n".join(lines[i : i + size]) for i in range(0, len(lines), size)]
    elif strategy == "chars":
        return [content[i : i + size] for i in range(0, len(content), size)]
    elif strategy == "paragraphs":
        paragraphs = re.split(r"\n\s*\n", content)
        return [
            "\n\n".join(paragraphs[i : i + size])
            for i in range(0, len(paragraphs), size)
        ]
    return []


# Tool definitions
TOOL_DEFINITIONS = [
    Tool(
        name="rlm_load_context",
        description="Load a large context as an external variable. Returns metadata without the content itself.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Identifier for this context"},
                "content": {"type": "string", "description": "The full context content"},
            },
            "required": ["name", "content"],
        },
    ),
    Tool(
        name="rlm_inspect_context",
        description="Inspect a loaded context - get structure info without loading full content into prompt.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Context identifier"},
                "preview_chars": {
                    "type": "integer",
                    "description": "Number of chars to preview (default 500)",
                    "default": 500,
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="rlm_chunk_context",
        description="Chunk a loaded context by strategy. Returns chunk metadata, not full content.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Context identifier"},
                "strategy": {
                    "type": "string",
                    "enum": ["lines", "chars", "paragraphs"],
                    "description": "Chunking strategy",
                    "default": "lines",
                },
                "size": {
                    "type": "integer",
                    "description": "Chunk size (lines/chars depending on strategy)",
                    "default": 100,
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="rlm_get_chunk",
        description="Get a specific chunk by index. Use after chunking to retrieve individual pieces.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Context identifier"},
                "chunk_index": {"type": "integer", "description": "Index of chunk to retrieve"},
            },
            "required": ["name", "chunk_index"],
        },
    ),
    Tool(
        name="rlm_filter_context",
        description="Filter context using regex/string operations. Creates a new filtered context.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Source context identifier"},
                "output_name": {"type": "string", "description": "Name for filtered context"},
                "pattern": {"type": "string", "description": "Regex pattern to match"},
                "mode": {
                    "type": "string",
                    "enum": ["keep", "remove"],
                    "description": "Keep or remove matching lines",
                    "default": "keep",
                },
            },
            "required": ["name", "output_name", "pattern"],
        },
    ),
    Tool(
        name="rlm_sub_query",
        description="Make a sub-LLM call on a chunk or filtered context. Core of recursive pattern.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question/instruction for the sub-call"},
                "context_name": {"type": "string", "description": "Context identifier to query against"},
                "chunk_index": {"type": "integer", "description": "Optional: specific chunk index"},
                "provider": {
                    "type": "string",
                    "enum": ["ollama", "claude-sdk"],
                    "description": "LLM provider for sub-call",
                    "default": "ollama",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use (provider-specific defaults apply)",
                },
            },
            "required": ["query", "context_name"],
        },
    ),
    Tool(
        name="rlm_store_result",
        description="Store a sub-call result for later aggregation.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Result set identifier"},
                "result": {"type": "string", "description": "Result content to store"},
                "metadata": {"type": "object", "description": "Optional metadata about this result"},
            },
            "required": ["name", "result"],
        },
    ),
    Tool(
        name="rlm_get_results",
        description="Retrieve stored results for aggregation.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Result set identifier"},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="rlm_list_contexts",
        description="List all loaded contexts and their metadata.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="rlm_sub_query_batch",
        description="Process multiple chunks in parallel. Respects concurrency limit to manage system resources.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question/instruction for each sub-call"},
                "context_name": {"type": "string", "description": "Context identifier"},
                "chunk_indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of chunk indices to process",
                },
                "provider": {
                    "type": "string",
                    "enum": ["ollama", "claude-sdk"],
                    "description": "LLM provider for sub-call",
                    "default": "ollama",
                },
                "model": {
                    "type": "string",
                    "description": "Model to use (provider-specific defaults apply)",
                },
                "concurrency": {
                    "type": "integer",
                    "description": "Max parallel requests (default 4, max 8)",
                    "default": 4,
                },
            },
            "required": ["query", "context_name", "chunk_indices"],
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available RLM tools."""
    return TOOL_DEFINITIONS


# Tool handlers
async def _handle_load_context(arguments: dict) -> list[TextContent]:
    """Load a large context as an external variable."""
    ctx_name = arguments["name"]
    content = arguments["content"]

    meta = {
        "name": ctx_name,
        "length": len(content),
        "lines": content.count("\n") + 1,
        "hash": _hash_content(content),
        "chunks": None,
    }
    contexts[ctx_name] = {"meta": meta, "content": content}
    _save_context_to_disk(ctx_name, content, meta)

    return _text_response({
        "status": "loaded",
        "name": ctx_name,
        "length": meta["length"],
        "lines": meta["lines"],
        "hash": meta["hash"],
    })


async def _handle_inspect_context(arguments: dict) -> list[TextContent]:
    """Inspect a loaded context."""
    ctx_name = arguments["name"]
    preview_chars = arguments.get("preview_chars", 500)

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _text_response(error)

    ctx = contexts[ctx_name]
    content = ctx["content"]
    chunk_meta = ctx["meta"].get("chunks")

    return _text_response({
        "name": ctx_name,
        "length": len(content),
        "lines": content.count("\n") + 1,
        "preview": content[:preview_chars],
        "has_chunks": chunk_meta is not None,
        "chunk_count": len(chunk_meta) if chunk_meta else 0,
    })


async def _handle_chunk_context(arguments: dict) -> list[TextContent]:
    """Chunk a loaded context by strategy."""
    ctx_name = arguments["name"]
    strategy = arguments.get("strategy", "lines")
    size = arguments.get("size", 100)

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _text_response(error)

    content = contexts[ctx_name]["content"]
    chunks = _chunk_content(content, strategy, size)

    chunk_meta = [
        {"index": i, "length": len(chunk), "preview": chunk[:100]}
        for i, chunk in enumerate(chunks)
    ]

    contexts[ctx_name]["meta"]["chunks"] = chunk_meta
    contexts[ctx_name]["chunks"] = chunks

    chunk_dir = CHUNKS_DIR / ctx_name
    chunk_dir.mkdir(exist_ok=True)
    for i, chunk in enumerate(chunks):
        (chunk_dir / f"{i}.txt").write_text(chunk)

    return _text_response({
        "status": "chunked",
        "name": ctx_name,
        "strategy": strategy,
        "chunk_count": len(chunks),
        "chunks": chunk_meta,
    })


async def _handle_get_chunk(arguments: dict) -> list[TextContent]:
    """Get a specific chunk by index."""
    ctx_name = arguments["name"]
    chunk_index = arguments["chunk_index"]

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _text_response(error)

    chunks = contexts[ctx_name].get("chunks")
    if not chunks:
        chunk_path = CHUNKS_DIR / ctx_name / f"{chunk_index}.txt"
        if chunk_path.exists():
            return _text_response(chunk_path.read_text())
        return _text_response(f"Context '{ctx_name}' not chunked")

    if chunk_index >= len(chunks):
        return _text_response(
            f"Chunk index {chunk_index} out of range (max {len(chunks) - 1})"
        )

    return _text_response(chunks[chunk_index])


async def _handle_filter_context(arguments: dict) -> list[TextContent]:
    """Filter context using regex."""
    src_name = arguments["name"]
    out_name = arguments["output_name"]
    pattern = arguments["pattern"]
    mode = arguments.get("mode", "keep")

    error = _ensure_context_loaded(src_name)
    if error:
        return _text_response(error)

    content = contexts[src_name]["content"]
    lines = content.split("\n")
    regex = re.compile(pattern)

    if mode == "keep":
        filtered = [line for line in lines if regex.search(line)]
    else:
        filtered = [line for line in lines if not regex.search(line)]

    new_content = "\n".join(filtered)
    meta = {
        "name": out_name,
        "length": len(new_content),
        "lines": len(filtered),
        "hash": _hash_content(new_content),
        "source": src_name,
        "filter_pattern": pattern,
        "filter_mode": mode,
        "chunks": None,
    }
    contexts[out_name] = {"meta": meta, "content": new_content}
    _save_context_to_disk(out_name, new_content, meta)

    return _text_response({
        "status": "filtered",
        "name": out_name,
        "original_lines": len(lines),
        "filtered_lines": len(filtered),
        "length": len(new_content),
    })


async def _handle_sub_query(arguments: dict) -> list[TextContent]:
    """Make a sub-LLM call on a chunk or context."""
    query = arguments["query"]
    ctx_name = arguments["context_name"]
    chunk_index = arguments.get("chunk_index")
    provider = arguments.get("provider", "ollama")
    model = arguments.get("model") or DEFAULT_MODELS.get(provider, "gemma3:27b")

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _error_response("context_not_found", error)

    if chunk_index is not None:
        chunks = contexts[ctx_name].get("chunks")
        if not chunks or chunk_index >= len(chunks):
            return _error_response(
                "chunk_not_available", f"Chunk {chunk_index} not available"
            )
        context_content = chunks[chunk_index]
    else:
        context_content = contexts[ctx_name]["content"]

    result, error = await _make_provider_call(provider, model, query, context_content)

    if error:
        return _text_response({
            "error": "provider_error",
            "provider": provider,
            "model": model,
            "message": error,
        })

    return _text_response({
        "provider": provider,
        "model": model,
        "response": result,
    })


async def _handle_store_result(arguments: dict) -> list[TextContent]:
    """Store a sub-call result for later aggregation."""
    result_name = arguments["name"]
    result = arguments["result"]
    metadata = arguments.get("metadata", {})

    results_file = RESULTS_DIR / f"{result_name}.jsonl"
    with open(results_file, "a") as f:
        f.write(json.dumps({"result": result, "metadata": metadata}) + "\n")

    return _text_response(f"Result stored to '{result_name}'")


async def _handle_get_results(arguments: dict) -> list[TextContent]:
    """Retrieve stored results for aggregation."""
    result_name = arguments["name"]
    results_file = RESULTS_DIR / f"{result_name}.jsonl"

    if not results_file.exists():
        return _text_response(f"No results found for '{result_name}'")

    results = [json.loads(line) for line in results_file.read_text().splitlines()]

    return _text_response({
        "name": result_name,
        "count": len(results),
        "results": results,
    })


async def _handle_list_contexts(_arguments: dict) -> list[TextContent]:
    """List all loaded contexts and their metadata."""
    ctx_list = [
        {
            "name": name,
            "length": ctx["meta"]["length"],
            "lines": ctx["meta"]["lines"],
            "chunked": ctx["meta"].get("chunks") is not None,
        }
        for name, ctx in contexts.items()
    ]

    for meta_file in CONTEXTS_DIR.glob("*.meta.json"):
        disk_name = meta_file.stem.replace(".meta", "")
        if disk_name not in contexts:
            meta = json.loads(meta_file.read_text())
            ctx_list.append({
                "name": disk_name,
                "length": meta["length"],
                "lines": meta["lines"],
                "chunked": meta.get("chunks") is not None,
                "disk_only": True,
            })

    return _text_response({"contexts": ctx_list})


async def _handle_sub_query_batch(arguments: dict) -> list[TextContent]:
    """Process multiple chunks in parallel."""
    query = arguments["query"]
    ctx_name = arguments["context_name"]
    chunk_indices = arguments["chunk_indices"]
    provider = arguments.get("provider", "ollama")
    model = arguments.get("model") or DEFAULT_MODELS.get(provider, "gemma3:27b")
    concurrency = min(arguments.get("concurrency", 4), 8)

    error = _ensure_context_loaded(ctx_name)
    if error:
        return _error_response("context_not_found", error)

    chunks = contexts[ctx_name].get("chunks")
    if not chunks:
        return _error_response(
            "context_not_chunked",
            f"Context '{ctx_name}' has not been chunked yet",
        )

    invalid_indices = [idx for idx in chunk_indices if idx >= len(chunks)]
    if invalid_indices:
        return _error_response(
            "invalid_chunk_indices",
            f"Invalid chunk indices: {invalid_indices} (max: {len(chunks) - 1})",
        )

    semaphore = asyncio.Semaphore(concurrency)

    async def process_chunk(chunk_idx: int) -> dict:
        async with semaphore:
            chunk_content = chunks[chunk_idx]
            result, error = await _make_provider_call(
                provider, model, query, chunk_content
            )

            if error:
                return {
                    "chunk_index": chunk_idx,
                    "error": "provider_error",
                    "message": error,
                }

            return {
                "chunk_index": chunk_idx,
                "response": result,
                "provider": provider,
                "model": model,
            }

    results = await asyncio.gather(*[process_chunk(idx) for idx in chunk_indices])

    successful = sum(1 for r in results if "response" in r)
    failed = len(results) - successful

    return _text_response({
        "status": "completed",
        "total_chunks": len(chunk_indices),
        "successful": successful,
        "failed": failed,
        "concurrency": concurrency,
        "results": results,
    })


# Tool dispatch table
TOOL_HANDLERS = {
    "rlm_load_context": _handle_load_context,
    "rlm_inspect_context": _handle_inspect_context,
    "rlm_chunk_context": _handle_chunk_context,
    "rlm_get_chunk": _handle_get_chunk,
    "rlm_filter_context": _handle_filter_context,
    "rlm_sub_query": _handle_sub_query,
    "rlm_store_result": _handle_store_result,
    "rlm_get_results": _handle_get_results,
    "rlm_list_contexts": _handle_list_contexts,
    "rlm_sub_query_batch": _handle_sub_query_batch,
}


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Route tool calls to their handlers."""
    handler = TOOL_HANDLERS.get(name)
    if handler:
        return await handler(arguments)
    return _text_response(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
