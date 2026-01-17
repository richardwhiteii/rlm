#!/usr/bin/env python3
"""
RLM MCP Server - Recursive Language Model patterns for massive context handling.

Implements the core insight from https://arxiv.org/html/2512.24601v1:
Treat context as external variable, chunk programmatically, sub-call recursively.
"""

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Optional
import hashlib

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from claude_agent_sdk import query as claude_query, ClaudeAgentOptions
    HAS_CLAUDE_SDK = True
except ImportError:
    HAS_CLAUDE_SDK = False

# Storage directory
DATA_DIR = Path(os.environ.get("RLM_DATA_DIR", "/tmp/rlm"))
CONTEXTS_DIR = DATA_DIR / "contexts"
CHUNKS_DIR = DATA_DIR / "chunks"
RESULTS_DIR = DATA_DIR / "results"

# Ensure directories exist
for d in [CONTEXTS_DIR, CHUNKS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# In-memory state (also persisted to disk)
contexts: dict[str, dict] = {}

server = Server("rlm")


def _hash_content(content: str) -> str:
    """Create short hash for content identification."""
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def _load_from_disk(name: str) -> Optional[dict]:
    """Load context from disk if exists."""
    meta_path = CONTEXTS_DIR / f"{name}.meta.json"
    content_path = CONTEXTS_DIR / f"{name}.txt"
    if meta_path.exists() and content_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["content"] = content_path.read_text()
        return meta
    return None


def _save_to_disk(name: str, content: str, meta: dict):
    """Persist context to disk."""
    (CONTEXTS_DIR / f"{name}.txt").write_text(content)
    meta_copy = {k: v for k, v in meta.items() if k != "content"}
    (CONTEXTS_DIR / f"{name}.meta.json").write_text(json.dumps(meta_copy, indent=2))


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available RLM tools."""
    return [
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
                    "preview_chars": {"type": "integer", "description": "Number of chars to preview (default 500)", "default": 500},
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
                    "size": {"type": "integer", "description": "Chunk size (lines/chars depending on strategy)", "default": 100},
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
                    "model": {"type": "string", "description": "Model to use (provider-specific defaults apply)"},
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
                        "description": "List of chunk indices to process"
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["ollama", "claude-sdk"],
                        "description": "LLM provider for sub-call",
                        "default": "ollama",
                    },
                    "model": {"type": "string", "description": "Model to use (provider-specific defaults apply)"},
                    "concurrency": {
                        "type": "integer",
                        "description": "Max parallel requests (default 4, max 8)",
                        "default": 4
                    },
                },
                "required": ["query", "context_name", "chunk_indices"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "rlm_load_context":
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
        _save_to_disk(ctx_name, content, meta)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "loaded",
                "name": ctx_name,
                "length": meta["length"],
                "lines": meta["lines"],
                "hash": meta["hash"],
            }, indent=2)
        )]

    elif name == "rlm_inspect_context":
        ctx_name = arguments["name"]
        preview_chars = arguments.get("preview_chars", 500)

        if ctx_name not in contexts:
            disk_ctx = _load_from_disk(ctx_name)
            if disk_ctx:
                contexts[ctx_name] = {"meta": disk_ctx, "content": disk_ctx.pop("content")}
            else:
                return [TextContent(type="text", text=f"Context '{ctx_name}' not found")]

        ctx = contexts[ctx_name]
        content = ctx["content"]

        return [TextContent(
            type="text",
            text=json.dumps({
                "name": ctx_name,
                "length": len(content),
                "lines": content.count("\n") + 1,
                "preview": content[:preview_chars],
                "has_chunks": ctx["meta"].get("chunks") is not None,
                "chunk_count": len(ctx["meta"]["chunks"]) if ctx["meta"].get("chunks") else 0,
            }, indent=2)
        )]

    elif name == "rlm_chunk_context":
        ctx_name = arguments["name"]
        strategy = arguments.get("strategy", "lines")
        size = arguments.get("size", 100)

        if ctx_name not in contexts:
            return [TextContent(type="text", text=f"Context '{ctx_name}' not found")]

        content = contexts[ctx_name]["content"]
        chunks = []

        if strategy == "lines":
            lines = content.split("\n")
            chunks = ["\n".join(lines[i:i+size]) for i in range(0, len(lines), size)]
        elif strategy == "chars":
            chunks = [content[i:i+size] for i in range(0, len(content), size)]
        elif strategy == "paragraphs":
            paragraphs = re.split(r"\n\s*\n", content)
            chunks = ["\n\n".join(paragraphs[i:i+size]) for i in range(0, len(paragraphs), size)]

        chunk_meta = [
            {
                "index": i,
                "length": len(c),
                "preview": c[:100],
            }
            for i, c in enumerate(chunks)
        ]

        contexts[ctx_name]["meta"]["chunks"] = chunk_meta
        contexts[ctx_name]["chunks"] = chunks

        # Save chunks to disk
        chunk_dir = CHUNKS_DIR / ctx_name
        chunk_dir.mkdir(exist_ok=True)
        for i, chunk in enumerate(chunks):
            (chunk_dir / f"{i}.txt").write_text(chunk)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "chunked",
                "name": ctx_name,
                "strategy": strategy,
                "chunk_count": len(chunks),
                "chunks": chunk_meta,
            }, indent=2)
        )]

    elif name == "rlm_get_chunk":
        ctx_name = arguments["name"]
        chunk_index = arguments["chunk_index"]

        if ctx_name not in contexts:
            return [TextContent(type="text", text=f"Context '{ctx_name}' not found")]

        chunks = contexts[ctx_name].get("chunks")
        if not chunks:
            # Try loading from disk
            chunk_path = CHUNKS_DIR / ctx_name / f"{chunk_index}.txt"
            if chunk_path.exists():
                return [TextContent(type="text", text=chunk_path.read_text())]
            return [TextContent(type="text", text=f"Context '{ctx_name}' not chunked")]

        if chunk_index >= len(chunks):
            return [TextContent(type="text", text=f"Chunk index {chunk_index} out of range (max {len(chunks)-1})")]

        return [TextContent(type="text", text=chunks[chunk_index])]

    elif name == "rlm_filter_context":
        src_name = arguments["name"]
        out_name = arguments["output_name"]
        pattern = arguments["pattern"]
        mode = arguments.get("mode", "keep")

        if src_name not in contexts:
            return [TextContent(type="text", text=f"Context '{src_name}' not found")]

        content = contexts[src_name]["content"]
        lines = content.split("\n")
        regex = re.compile(pattern)

        if mode == "keep":
            filtered = [l for l in lines if regex.search(l)]
        else:
            filtered = [l for l in lines if not regex.search(l)]

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
        _save_to_disk(out_name, new_content, meta)

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "filtered",
                "name": out_name,
                "original_lines": len(lines),
                "filtered_lines": len(filtered),
                "length": len(new_content),
            }, indent=2)
        )]

    elif name == "rlm_sub_query":
        query = arguments["query"]
        ctx_name = arguments["context_name"]
        chunk_index = arguments.get("chunk_index")
        provider = arguments.get("provider", "ollama")
        model = arguments.get("model")

        # Default models per provider
        if not model:
            model = "gemma3:27b" if provider == "ollama" else "claude-opus-4-5-20251101"

        # Get context content
        if ctx_name not in contexts:
            return [TextContent(type="text", text=json.dumps({
                "error": "context_not_found",
                "message": f"Context '{ctx_name}' not found"
            }))]

        if chunk_index is not None:
            chunks = contexts[ctx_name].get("chunks")
            if not chunks or chunk_index >= len(chunks):
                return [TextContent(type="text", text=json.dumps({
                    "error": "chunk_not_available",
                    "message": f"Chunk {chunk_index} not available"
                }))]
            context_content = chunks[chunk_index]
        else:
            context_content = contexts[ctx_name]["content"]

        # Make the sub-call
        result = None
        error = None

        if provider == "ollama":
            if not HAS_HTTPX:
                return [TextContent(type="text", text=json.dumps({
                    "error": "httpx_not_installed",
                    "message": "httpx required for Ollama calls"
                }))]

            ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(
                        f"{ollama_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": f"{query}\n\nContext:\n{context_content}",
                            "stream": False,
                        }
                    )
                    response.raise_for_status()
                    result = response.json().get("response", "")
            except Exception as e:
                error = str(e)

        elif provider == "claude-sdk":
            if not HAS_CLAUDE_SDK:
                return [TextContent(type="text", text=json.dumps({
                    "error": "claude_sdk_not_installed",
                    "message": "claude-agent-sdk required for claude-sdk provider"
                }))]

            try:
                prompt = f"{query}\n\nContext:\n{context_content}"
                options = ClaudeAgentOptions(max_turns=1)

                messages = []
                async for message in claude_query(prompt=prompt, options=options):
                    if hasattr(message, 'content'):
                        messages.append(message.content)

                result = "\n".join(str(m) for m in messages) if messages else ""
            except Exception as e:
                error = str(e)

        else:
            return [TextContent(type="text", text=json.dumps({
                "error": "unknown_provider",
                "message": f"Unknown provider: {provider}"
            }))]

        if error:
            return [TextContent(type="text", text=json.dumps({
                "error": "provider_error",
                "provider": provider,
                "model": model,
                "message": error,
            }))]

        return [TextContent(type="text", text=json.dumps({
            "provider": provider,
            "model": model,
            "response": result,
        }))]

    elif name == "rlm_store_result":
        result_name = arguments["name"]
        result = arguments["result"]
        metadata = arguments.get("metadata", {})

        results_file = RESULTS_DIR / f"{result_name}.jsonl"
        with open(results_file, "a") as f:
            f.write(json.dumps({"result": result, "metadata": metadata}) + "\n")

        return [TextContent(type="text", text=f"Result stored to '{result_name}'")]

    elif name == "rlm_get_results":
        result_name = arguments["name"]
        results_file = RESULTS_DIR / f"{result_name}.jsonl"

        if not results_file.exists():
            return [TextContent(type="text", text=f"No results found for '{result_name}'")]

        results = []
        with open(results_file) as f:
            for line in f:
                results.append(json.loads(line))

        return [TextContent(
            type="text",
            text=json.dumps({"name": result_name, "count": len(results), "results": results}, indent=2)
        )]

    elif name == "rlm_list_contexts":
        ctx_list = []
        for name, ctx in contexts.items():
            ctx_list.append({
                "name": name,
                "length": ctx["meta"]["length"],
                "lines": ctx["meta"]["lines"],
                "chunked": ctx["meta"].get("chunks") is not None,
            })

        # Also check disk for any not in memory
        for meta_file in CONTEXTS_DIR.glob("*.meta.json"):
            name = meta_file.stem.replace(".meta", "")
            if name not in contexts:
                meta = json.loads(meta_file.read_text())
                ctx_list.append({
                    "name": name,
                    "length": meta["length"],
                    "lines": meta["lines"],
                    "chunked": meta.get("chunks") is not None,
                    "disk_only": True,
                })

        return [TextContent(
            type="text",
            text=json.dumps({"contexts": ctx_list}, indent=2)
        )]

    elif name == "rlm_sub_query_batch":
        query = arguments["query"]
        ctx_name = arguments["context_name"]
        chunk_indices = arguments["chunk_indices"]
        provider = arguments.get("provider", "ollama")
        model = arguments.get("model")
        concurrency = min(arguments.get("concurrency", 4), 8)  # Cap at 8

        # Default models per provider
        if not model:
            model = "gemma3:27b" if provider == "ollama" else "claude-opus-4-5-20251101"

        # Verify context exists
        if ctx_name not in contexts:
            return [TextContent(type="text", text=json.dumps({
                "error": "context_not_found",
                "message": f"Context '{ctx_name}' not found"
            }))]

        # Verify chunks exist
        chunks = contexts[ctx_name].get("chunks")
        if not chunks:
            return [TextContent(type="text", text=json.dumps({
                "error": "context_not_chunked",
                "message": f"Context '{ctx_name}' has not been chunked yet"
            }))]

        # Validate chunk indices
        invalid_indices = [idx for idx in chunk_indices if idx >= len(chunks)]
        if invalid_indices:
            return [TextContent(type="text", text=json.dumps({
                "error": "invalid_chunk_indices",
                "message": f"Invalid chunk indices: {invalid_indices} (max: {len(chunks)-1})"
            }))]

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)

        async def process_chunk(chunk_idx: int) -> dict:
            """Process a single chunk with semaphore control."""
            async with semaphore:
                chunk_content = chunks[chunk_idx]
                result = None
                error = None

                if provider == "ollama":
                    if not HAS_HTTPX:
                        return {
                            "chunk_index": chunk_idx,
                            "error": "httpx_not_installed",
                            "message": "httpx required for Ollama calls"
                        }

                    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
                    try:
                        async with httpx.AsyncClient(timeout=180.0) as client:
                            response = await client.post(
                                f"{ollama_url}/api/generate",
                                json={
                                    "model": model,
                                    "prompt": f"{query}\n\nContext:\n{chunk_content}",
                                    "stream": False,
                                }
                            )
                            response.raise_for_status()
                            result = response.json().get("response", "")
                    except Exception as e:
                        error = str(e)

                elif provider == "claude-sdk":
                    if not HAS_CLAUDE_SDK:
                        return {
                            "chunk_index": chunk_idx,
                            "error": "claude_sdk_not_installed",
                            "message": "claude-agent-sdk required for claude-sdk provider"
                        }

                    try:
                        prompt = f"{query}\n\nContext:\n{chunk_content}"
                        options = ClaudeAgentOptions(max_turns=1)

                        messages = []
                        async for message in claude_query(prompt=prompt, options=options):
                            if hasattr(message, 'content'):
                                messages.append(message.content)

                        result = "\n".join(str(m) for m in messages) if messages else ""
                    except Exception as e:
                        error = str(e)

                if error:
                    return {
                        "chunk_index": chunk_idx,
                        "error": "provider_error",
                        "message": error
                    }

                return {
                    "chunk_index": chunk_idx,
                    "response": result,
                    "provider": provider,
                    "model": model
                }

        # Process all chunks in parallel (with concurrency limit)
        results = await asyncio.gather(*[process_chunk(idx) for idx in chunk_indices])

        # Count successes and failures
        successful = sum(1 for r in results if "response" in r)
        failed = len(results) - successful

        return [TextContent(
            type="text",
            text=json.dumps({
                "status": "completed",
                "total_chunks": len(chunk_indices),
                "successful": successful,
                "failed": failed,
                "concurrency": concurrency,
                "results": results
            }, indent=2)
        )]

    return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
