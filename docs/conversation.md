How could this paper be applied integrated into Claude code agents hooks plugin skill slash command https://arxiv.org/html/2512.24601v18:54 AM1 stepFetched Recursive Language Modelsarxiv.orgExamined recursive language model patterns and architected integration strategies.10sThis is a fascinating paper on Recursive Language Models (RLMs). Let me understand the key concepts:
Core Insight: RLMs treat long prompts as part of an external environment rather than feeding them directly into the model. The LLM can then programmatically examine, decompose, and recursively call itself over snippets of the prompt.
How it works:

Loads input prompt as a variable in a Python REPL environment
The LLM writes code to peek into, decompose, and invoke itself recursively
Uses sub-LLM calls to process chunks of context
Aggregates results programmatically

Key patterns observed:

Filtering input using code execution (regex, etc.)
Chunking and recursively sub-calling LMs
Answer verification through sub-LM calls
Passing recursive LM outputs through variables

Now, how could this be integrated into Claude Code agents, hooks, plugins, skills, or slash commands?
Integration approaches:

As a Claude Code Skill - Create a skill that implements the RLM pattern with:

A REPL environment setup
Context loading as a variable
Sub-LLM query function
Recursive decomposition patterns


As a Hook - Use hooks to:

Intercept long prompts and automatically apply RLM treatment
Pre-process context into chunks
Post-process aggregated results


As a Slash Command - /rlm command that:

Takes a file/context as input
Automatically applies recursive decomposition
Returns aggregated answer


As a Plugin/MCP Server - Build an MCP server that:

Exposes tools for context chunking
Provides sub-LLM query capabilities
Manages REPL state



Let me provide a comprehensive integration guide.This is a compelling paper on Recursive Language Models (RLMs) from MIT. The core insight: rather than feeding massive contexts directly into an LLM, treat the prompt as an external variable in a REPL environment where the model can programmatically inspect, chunk, and recursively sub-call itself over portions.
Here's how to integrate this into Claude Code:

1. As a Custom Skill
Create /mnt/skills/user/rlm/SKILL.md:
markdown# RLM Skill - Recursive Context Processing

## When to Use
- Context exceeds ~100K tokens
- Information-dense aggregation tasks (counting, comparing across documents)
- Multi-hop reasoning over large codebases or document sets

## Implementation Pattern
1. Load context as a Python variable (not in prompt)
2. Write code to inspect structure (print first N chars, count lines)
3. Chunk strategically based on structure
4. Use sub-LLM calls per chunk with targeted queries
5. Aggregate results programmatically
6. Final LLM call to synthesize answer
python# rlm_processor.py
import anthropic

client = anthropic.Anthropic()

def llm_query(prompt: str, context: str = "") -> str:
    """Sub-LLM call for recursive processing"""
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": f"{prompt}\n\nContext:\n{context[:400000]}"}]
    ).content[0].text

def rlm_process(context: str, query: str, chunk_size: int = 50000) -> str:
    """RLM pattern: chunk → sub-query → aggregate"""
    chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
    
    # Phase 1: Extract relevant info per chunk
    findings = []
    for i, chunk in enumerate(chunks):
        result = llm_query(f"Extract info relevant to: {query}", chunk)
        findings.append(result)
    
    # Phase 2: Aggregate
    return llm_query(f"Synthesize answer to: {query}", "\n---\n".join(findings))

2. As a Hook (Pre-processing)
In .claude/hooks/pre_tool.sh:
bash#!/bin/bash
# Intercept large file reads and apply RLM chunking

if [[ "$TOOL_NAME" == "view" ]]; then
    file_size=$(stat -f%z "$TOOL_ARGS_PATH" 2>/dev/null || stat -c%s "$TOOL_ARGS_PATH")
    if [[ $file_size -gt 500000 ]]; then
        echo "HOOK: Large file detected ($file_size bytes). Applying RLM chunking strategy."
        # Trigger RLM skill instead of direct view
    fi
fi

3. As a Slash Command
Add to .claude/commands/rlm.md:
markdown---
name: rlm
description: Process large context recursively
arguments:
  - name: file
    description: Path to large file/directory
  - name: query
    description: Question to answer
---

Process $ARGUMENTS.file using the Recursive Language Model pattern:

1. Load the file content as a variable (do not include in prompt)
2. Print structure info: `len(context)`, first 500 chars, line count
3. Determine chunking strategy based on structure
4. For each chunk, call sub-query: `llm_query(f"For query '{$ARGUMENTS.query}', extract relevant info from: {chunk}")`
5. Aggregate all chunk responses
6. Final synthesis call to answer the original query

Use the rlm_processor.py module from the RLM skill.

4. As an MCP Server
python# mcp_rlm_server.py
from mcp import Server, Tool
import anthropic

server = Server("rlm-processor")
contexts = {}  # State: loaded contexts

@server.tool("load_context")
def load_context(name: str, content: str) -> dict:
    """Load context as external variable"""
    contexts[name] = content
    return {"name": name, "length": len(content), "preview": content[:200]}

@server.tool("chunk_context") 
def chunk_context(name: str, strategy: str = "lines", size: int = 100) -> list:
    """Chunk loaded context"""
    ctx = contexts[name]
    if strategy == "lines":
        lines = ctx.split("\n")
        return ["\n".join(lines[i:i+size]) for i in range(0, len(lines), size)]
    return [ctx[i:i+size*1000] for i in range(0, len(ctx), size*1000)]

@server.tool("sub_llm_query")
def sub_llm_query(query: str, chunk: str) -> str:
    """Recursive sub-LLM call"""
    client = anthropic.Anthropic()
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": f"{query}\n\n{chunk}"}]
    ).content[0].text

Key Implementation Patterns from the Paper
PatternImplementationFilter via codeUse regex/string ops to narrow context before sub-callsSemantic chunkingSplit by structure (newlines, sections) not arbitrary char countSub-call for classificationOne LLM call per chunk for semantic tasks (labeling, extraction)Variable buffersStore sub-call outputs in Python variables, aggregate programmaticallyAnswer verificationFinal sub-call to verify synthesized answer against evidence
The paper shows RLMs handling 10M+ tokens with comparable cost to base models by selectively viewing context rather than ingesting everything. For your autonomous agent work, this is particularly relevant—agents can use RLM patterns to fill capability gaps when dealing with massive codebases or document corpuses.Rank these options by expected effectiveness of Claude using them autonomously9:02 AMEvaluated integration methods by Claude's autonomous decision-making capability.14sThe user wants me to rank the four integration options I proposed (Skill, Hook, Slash Command, MCP Server) by how effective they would be for Claude to use autonomously—meaning without human intervention, in agentic workflows like Claude Code.
Let me think through each:

MCP Server - This gives Claude explicit tools it can call programmatically. Claude can decide when to load context, chunk it, and make sub-calls. High autonomy potential because Claude can invoke these tools in any order based on reasoning.
Skill - Claude reads SKILL.md and follows the pattern. It provides guidance and reusable code, but requires Claude to remember to check the skill and apply it. Good for Claude Code's existing workflow.
Hook - Pre-processing hooks fire automatically based on conditions (file size). This is automatic but less flexible—Claude doesn't control when it fires. Good for guardrails but reduces Claude's agency in deciding how to process.
Slash Command - User-initiated. Requires the human to invoke /rlm. Lowest autonomy since it depends on human triggering.

For autonomous operation, the ranking should be:

MCP Server - Most autonomous. Claude can call tools at will, chain them, adapt strategy mid-task.
Skill - High autonomy but requires Claude to recognize when to apply it and follow the pattern correctly.
Hook - Automatic triggering is good, but less adaptive—fires on conditions, Claude doesn't control.
Slash Command - Requires human to initiate, lowest autonomy.

But wait—let me reconsider the hook. If the hook is designed well, it could automatically intercept large context situations and apply RLM without Claude needing to remember. That's actually powerful for autonomy in the sense of "works without Claude having to think about it." But it's also rigid.
For Claude using them autonomously (Claude making decisions), MCP > Skill > Hook > Slash Command.
The distinction matters here: if I'm measuring whether the system operates without Claude's active involvement, hooks could rank higher since they trigger automatically. But if I'm measuring Claude's ability to deliberately choose and adapt its approach, MCP and Skill take precedence. I'll go with the latter interpretation—Claude autonomously deciding when and how to apply RLM patterns.Ranking by Autonomous Effectiveness
RankOptionAutonomy ScoreRationale1MCP Server⭐⭐⭐⭐⭐Claude invokes tools at will—load_context, chunk_context, sub_llm_query—chaining them based on runtime reasoning. Full control over when/how to apply RLM strategy.2Skill⭐⭐⭐⭐Claude reads SKILL.md when it recognizes a large-context situation, then executes the pattern via bash/code tools. Requires Claude to remember to check the skill and correctly apply multi-step logic.3Hook⭐⭐⭐Fires automatically on conditions (file size > threshold), but Claude doesn't control when or how. Good guardrail, poor adaptability—can't adjust chunking strategy mid-task.4Slash Command⭐⭐Human must invoke /rlm. Zero autonomy—entirely dependent on user recognizing when RLM is needed.

Why MCP Wins for Autonomous Agents
┌─────────────────────────────────────────────────────────┐
│  Claude's Decision Loop (Autonomous)                    │
├─────────────────────────────────────────────────────────┤
│  1. Observe: "This codebase is 2M tokens"               │
│  2. Reason: "Direct ingestion will cause context rot"   │
│  3. Act: Call load_context → chunk_context → sub_query  │
│  4. Adapt: "Chunks too large, re-chunk at 25K"          │
│  5. Aggregate: Synthesize from sub-call outputs         │
└─────────────────────────────────────────────────────────┘
MCP tools let Claude adapt strategy mid-execution—the paper's key insight. The RLM trajectories show models:

Starting with regex filtering based on priors
Adjusting chunk sizes after observing structure
Verifying answers with additional sub-calls

Only MCP preserves this flexibility. Skills come close but rely on Claude executing a pre-defined script rather than dynamically composing tool calls.

Hybrid Recommendation
For your autonomous agent work, combine them:
MCP Server (primary)     → Claude's runtime toolkit for RLM patterns
     ↓
Skill (guidance)         → SKILL.md documents *when* to use RLM tools
     ↓  
Hook (safety net)        → Auto-intercept if Claude tries to ingest >500K directly
The hook becomes a fallback that fires if Claude forgets to use the MCP tools on massive contexts—preventing context rot without removing Claude's agency for smaller cases.What is the data store for the MCP|skill|hook9:14 AMExamined state persistence mechanisms across three architectural approaches.The user is asking about where/how data is stored across these three implementation options. This is a practical architectural question about state management.
Let me think through each:

MCP Server: The example I gave used an in-memory Python dict (contexts = {}). This is ephemeral - dies when server restarts. For persistence, you'd need to connect to actual storage.
Skill: Skills in Claude Code execute via bash/Python in the container. Data lives in the filesystem (/home/claude/) during the session, but the filesystem resets between tasks per the system prompt.
Hook: Hooks are shell scripts that run on tool invocation. They don't have persistent state themselves - they'd need to write to filesystem or call external services.

The user is likely thinking about:

Where does the loaded context variable live?
How do sub-call results get accumulated?
What persists across conversation turns vs. sessions?

This is a systems architecture question about state management for each approach.Data Store Comparison
OptionRuntime StatePersistenceScopeMCP ServerPython process memoryServer lifetime (ephemeral by default)Across all tool calls in sessionSkillContainer filesystem + Python variablesTask duration only (/home/claude resets)Single task executionHookNone inherentMust delegate to external storePer-invocation only

MCP Server
python# Default: in-memory dict (dies on restart)
contexts = {}

# Persistent options:
# 1. SQLite (local, simple)
import sqlite3
db = sqlite3.connect("/data/rlm_contexts.db")

# 2. Redis (if you need cross-instance sharing)
import redis
r = redis.Redis(host='localhost', port=6379)

# 3. Filesystem (your Mac Studio w/ Ollama)
import json
from pathlib import Path
STORE = Path("/mnt/rlm_cache")

def load_context(name: str, content: str):
    (STORE / f"{name}.txt").write_text(content)
    (STORE / f"{name}.meta.json").write_text(json.dumps({
        "length": len(content),
        "chunks": None  # populated later
    }))
Best for your setup: Filesystem on Mac Studio. Contexts persist across Claude Code sessions, Ollama can access same files for local inference comparison.

Skill
python# Skill executes in /home/claude (ephemeral container)
# State lives only during task execution

# Working directory approach:
WORK_DIR = Path("/home/claude/rlm_workspace")
WORK_DIR.mkdir(exist_ok=True)

# Store chunks and intermediate results
(WORK_DIR / "chunks").mkdir(exist_ok=True)
(WORK_DIR / "results").mkdir(exist_ok=True)

# For cross-session persistence, must copy to outputs:
# /mnt/user-data/outputs/rlm_cache/
Limitation: Container resets between tasks. If you want to cache a chunked codebase for reuse, you'd need to serialize to /mnt/user-data/outputs/ and reload.

Hook
bash#!/bin/bash
# Hooks are stateless - fire and forget

# Option 1: Write to temp file for downstream consumption
echo "$CONTEXT_METADATA" > /tmp/rlm_hook_state.json

# Option 2: Call external service
curl -X POST http://localhost:8080/rlm/register \
  -d "{\"file\": \"$TOOL_ARGS_PATH\", \"size\": $FILE_SIZE}"

# Option 3: Set environment variable (same shell session only)
export RLM_ACTIVE=1
```

**Reality**: Hooks aren't designed for state. They're triggers, not stores.

---

## Recommended Architecture for Your Setup
```
┌─────────────────────────────────────────────────────────────┐
│  Mac Studio M4 Max                                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   /data/rlm/                    ← Persistent store          │
│   ├── contexts/                 ← Raw loaded contexts       │
│   ├── chunks/                   ← Pre-chunked versions      │
│   ├── embeddings/               ← Optional: for retrieval   │
│   └── results/                  ← Cached sub-call outputs   │
│                                                             │
│   MCP Server ←──────────────────→ Filesystem                │
│       ↑                                                     │
│       │ tool calls                                          │
│       ↓                                                     │
│   Claude Code                                               │
│                                                             │
│   Ollama ←──────────────────────→ Same /data/rlm/           │
│   (local sub-calls)               (shared access)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
This lets you:

Cache chunked codebases once, reuse across sessions
Run sub-calls through Ollama locally (cost = $0)
Compare Ollama vs API sub-call quality
Persist RLM "trajectories" for debugging (like the paper's appendix examples)
