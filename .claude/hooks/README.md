# RLM Hooks

## Large File Suggester Hook

**File:** `check-file-size.sh`

This PreToolUse hook intercepts `Read` tool calls and suggests using RLM tools when files exceed configurable thresholds.

### Behavior

Triggers when **EITHER** threshold is exceeded:
- **Size threshold:** 25KB (configurable via `RLM_SIZE_THRESHOLD` env var)
- **Token threshold:** 10K tokens (configurable via `RLM_TOKEN_THRESHOLD` env var)

Token estimation uses ~4 characters per token (conservative estimate).

**Why token threshold matters:** Claude's Read tool **hard-fails at 25K tokens**. The 10K token threshold warns early so you can use RLM before hitting the hard limit.

- **Action:** Suggests RLM tools but does NOT block the read
- **Output:** JSON with `decision: "approve"` and optional `reason` message

### Configuration

The hook is configured in `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Read",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/check-file-size.sh"
          }
        ]
      }
    ]
  }
}
```

### Testing

#### Manual Testing

1. **Create a large test file (>25KB):**
   ```bash
   python3 -c "print('x' * 150000)" > /tmp/large_file.txt
   ```

2. **Test the hook directly:**
   ```bash
   # Test with large file (should show suggestion)
   echo '{"tool_input": {"file_path": "/tmp/large_file.txt"}}' | .claude/hooks/check-file-size.sh

   # Expected output:
   # {
   #   "decision": "approve",
   #   "reason": "This file is 146.48 KB ..."
   # }
   ```

3. **Test with small file (should NOT show suggestion):**
   ```bash
   echo '{"tool_input": {"file_path": "/Users/richard/projects/fun/rlm/README.md"}}' | .claude/hooks/check-file-size.sh

   # Expected output:
   # {"decision": "approve"}
   ```

4. **Test edge cases:**
   ```bash
   # Non-existent file
   echo '{"tool_input": {"file_path": "/nonexistent.txt"}}' | .claude/hooks/check-file-size.sh
   # Output: {"decision": "approve"}

   # No file_path
   echo '{"tool_input": {}}' | .claude/hooks/check-file-size.sh
   # Output: {"decision": "approve"}
   ```

#### Customizing Thresholds

**Size threshold** (`RLM_SIZE_THRESHOLD` in bytes):

```bash
# Use 50KB size threshold
RLM_SIZE_THRESHOLD=51200 echo '{"tool_input": {"file_path": "somefile.txt"}}' | .claude/hooks/check-file-size.sh

# Use 1MB size threshold
RLM_SIZE_THRESHOLD=1048576 echo '{"tool_input": {"file_path": "somefile.txt"}}' | .claude/hooks/check-file-size.sh
```

**Token threshold** (`RLM_TOKEN_THRESHOLD` in tokens):

```bash
# Use 5K token threshold (more aggressive, catches smaller files)
RLM_TOKEN_THRESHOLD=5000 echo '{"tool_input": {"file_path": "somefile.txt"}}' | .claude/hooks/check-file-size.sh

# Use 20K token threshold (closer to hard limit)
RLM_TOKEN_THRESHOLD=20000 echo '{"tool_input": {"file_path": "somefile.txt"}}' | .claude/hooks/check-file-size.sh
```

**Both thresholds together:**

```bash
# Custom size (100KB) and token (15K) thresholds
RLM_SIZE_THRESHOLD=102400 RLM_TOKEN_THRESHOLD=15000 .claude/hooks/check-file-size.sh
```

### Integration Testing with Claude Code

1. Start Claude Code in the RLM project directory
2. Ask Claude to read a large file (>25KB)
3. Observe the suggestion message before the file is read
4. The read will still proceed (not blocked)

### File Structure

```
.claude/
  hooks/
    check-file-size.sh     # Hook script
    rlm-suggest.json       # Hook metadata (optional, for documentation)
    README.md              # This file
  settings.json            # Hook configuration
```

### Suggested RLM Workflow

When the hook fires, it suggests:

1. `rlm_load_context(name, content)` - Load as external variable
2. `rlm_inspect_context(name)` - Get structure without full content
3. `rlm_chunk_context(name, 'paragraphs')` - Split strategically
4. `rlm_sub_query_batch(query, name, indices)` - Process in parallel

This pattern prevents context bloat and enables processing files that exceed token limits.
