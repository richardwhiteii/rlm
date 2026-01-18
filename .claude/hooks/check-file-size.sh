#!/usr/bin/env bash
# Hook script to suggest RLM tools for large file reads
# This is a PreToolUse hook for the Read tool
#
# Threshold: 25KB (configurable via RLM_SIZE_THRESHOLD env var)
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

# Threshold: 25KB = 25600 bytes (configurable via env var)
threshold="${RLM_SIZE_THRESHOLD:-25600}"

# If file is larger than threshold, suggest RLM tools
if [[ "$file_size" -gt "$threshold" ]]; then
  # Format file size for display
  if [[ "$file_size" -gt 1048576 ]]; then
    size_display="$(echo "scale=2; $file_size / 1048576" | bc) MB"
  elif [[ "$file_size" -gt 1024 ]]; then
    size_display="$(echo "scale=2; $file_size / 1024" | bc) KB"
  else
    size_display="${file_size} bytes"
  fi

  # Get filename for display
  filename=$(basename "$file_path")

  message="FILE SIZE: ${size_display} (${filename}) exceeds 25KB threshold.

Per RFC 2119 keywords:
- You SHOULD use rlm_load_context() to store this file as an external variable
- You SHOULD NOT load files >25KB directly into context window
- You MUST use rlm_chunk_context() before processing files >100KB
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
