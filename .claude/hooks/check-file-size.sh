#!/usr/bin/env bash
# Hook script to suggest RLM tools for large file reads
# This is a PreToolUse hook for the Read tool

set -euo pipefail

# Read the hook input (tool name and arguments)
input=$(cat)

# Parse the file_path from the Read tool arguments
file_path=$(echo "$input" | jq -r '.arguments.file_path // empty')

# If no file_path, allow the read to continue
if [[ -z "$file_path" ]]; then
  echo '{"continue": true}'
  exit 0
fi

# Check if file exists
if [[ ! -f "$file_path" ]]; then
  # File doesn't exist yet - let Read tool handle the error
  echo '{"continue": true}'
  exit 0
fi

# Get file size in bytes
file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null || echo "0")

# Threshold: 10KB = 10000 bytes
threshold=10000

# If file is larger than threshold, suggest RLM tools
if [[ "$file_size" -gt "$threshold" ]]; then
  # Format file size for display
  if [[ "$file_size" -gt 1048576 ]]; then
    size_display="$(echo "scale=2; $file_size / 1048576" | bc)MB"
  elif [[ "$file_size" -gt 1024 ]]; then
    size_display="$(echo "scale=2; $file_size / 1024" | bc)KB"
  else
    size_display="${file_size} bytes"
  fi

  message="📊 Large file detected: $file_path ($size_display)

Consider using RLM tools for efficient processing:
  1. rlm_load_context: Load file externally (keeps context clean)
  2. rlm_chunk_context: Break into manageable pieces
  3. rlm_sub_query_batch: Process chunks in parallel with sub-LLM calls

This approach prevents context bloat and enables processing files that would otherwise exceed token limits."

  # Return JSON with message but allow read to continue
  jq -n \
    --arg msg "$message" \
    '{
      "continue": true,
      "message": $msg
    }'
else
  # File is small enough - proceed normally
  echo '{"continue": true}'
fi
