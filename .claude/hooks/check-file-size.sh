#!/usr/bin/env bash
# Hook script to suggest RLM tools for large file reads
# This is a PreToolUse hook for the Read tool
#
# Triggers when EITHER:
#   - File size > 25KB (configurable via RLM_SIZE_THRESHOLD)
#   - Estimated tokens > 10K (configurable via RLM_TOKEN_THRESHOLD)
#
# Token estimation: ~4 characters per token (conservative)
# Behavior: Suggests RLM tools but does NOT block the read

set -euo pipefail

# Read the hook input (tool name and arguments)
input=$(cat)

# Parse the file_path from the Read tool arguments
file_path=$(echo "$input" | jq -r '.tool_input.file_path // empty')

# If no file_path, allow the read to continue
if [[ -z "$file_path" ]]; then
  echo '{"decision": "approve"}'
  exit 0
fi

# Check if file exists
if [[ ! -f "$file_path" ]]; then
  # File doesn't exist yet - let Read tool handle the error
  echo '{"decision": "approve"}'
  exit 0
fi

# Get file size in bytes
file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo "0")

# Estimate token count (~4 chars per token)
# Claude's Read tool hard-fails at 25000 tokens, so warn earlier
estimated_tokens=$((file_size / 4))

# Thresholds (configurable via env vars)
size_threshold="${RLM_SIZE_THRESHOLD:-25600}"        # 25KB default
token_threshold="${RLM_TOKEN_THRESHOLD:-10000}"      # 10K tokens default

# Check if either threshold is exceeded
exceeds_size=$([[ "$file_size" -gt "$size_threshold" ]] && echo "yes" || echo "no")
exceeds_tokens=$([[ "$estimated_tokens" -gt "$token_threshold" ]] && echo "yes" || echo "no")

# If either threshold exceeded, suggest RLM tools
if [[ "$exceeds_size" == "yes" || "$exceeds_tokens" == "yes" ]]; then
  # Format file size for display
  if [[ "$file_size" -gt 1048576 ]]; then
    size_display="$(echo "scale=2; $file_size / 1048576" | bc) MB"
  elif [[ "$file_size" -gt 1024 ]]; then
    size_display="$(echo "scale=2; $file_size / 1024" | bc) KB"
  else
    size_display="${file_size} bytes"
  fi

  # Format token count for display
  if [[ "$estimated_tokens" -gt 1000 ]]; then
    token_display="~$(echo "scale=1; $estimated_tokens / 1000" | bc)K tokens"
  else
    token_display="~${estimated_tokens} tokens"
  fi

  # Get filename for display
  filename=$(basename "$file_path")

  # Build trigger reason
  if [[ "$exceeds_tokens" == "yes" && "$exceeds_size" == "yes" ]]; then
    trigger="FILE SIZE: ${size_display} AND TOKENS: ${token_display}"
  elif [[ "$exceeds_tokens" == "yes" ]]; then
    trigger="TOKEN COUNT: ${token_display} (${size_display})"
  else
    trigger="FILE SIZE: ${size_display} (${token_display})"
  fi

  message="${trigger} for ${filename} exceeds threshold.

Claude's Read tool HARD-FAILS at 25K tokens. This file will likely fail.

Per RFC 2119 keywords:
- You MUST use RLM tools for files >10K tokens (Read tool will fail otherwise)
- You SHOULD use rlm_load_context() to store this file as an external variable
- You SHOULD NOT load files >25KB directly into context window
- You MAY proceed with direct read ONLY for quick inspection (use limit parameter)

RECOMMENDED workflow:
  1. rlm_load_context(name, content) - REQUIRED first step
  2. rlm_inspect_context(name) - Get metadata without loading content
  3. rlm_chunk_context(name, 'lines', 200) - Split into processable pieces
  4. rlm_sub_query_batch(query, name, indices) - Process chunks in parallel

Pattern: Load -> Inspect -> Chunk -> Sub-query -> Aggregate"

  # Return JSON with message but allow read to continue (don't block)
  jq -n \
    --arg msg "$message" \
    '{
      "decision": "approve",
      "reason": $msg
    }'
else
  # File is small enough - proceed normally
  echo '{"decision": "approve"}'
fi
