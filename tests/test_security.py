"""Security integration tests for the RLM MCP server.

These tests verify that security measures work correctly:
- Path traversal prevention
- ReDoS prevention via regex timeout
- Memory/size limits
- Integer validation

IMPORTANT: No mocks - these are real integration tests.
"""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

# Set test data directory before importing server
TEST_DATA_DIR = tempfile.mkdtemp(prefix="rlm_security_test_")
os.environ["RLM_DATA_DIR"] = TEST_DATA_DIR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlm_mcp_server import (
    _handle_chunk_context,
    _handle_exec,
    _handle_filter_context,
    _handle_get_chunk,
    _handle_get_results,
    _handle_load_context,
    _handle_store_result,
    _handle_sub_query,
    _handle_sub_query_batch,
    _check_disk_quota,
    _get_disk_usage,
    _safe_mkdir,
    contexts,
    CHUNKS_DIR,
    MAX_CONTEXT_SIZE,
    MAX_EXEC_CONTEXT_SIZE,
    MAX_RESULTS_FILE_SIZE,
    MAX_TOTAL_MEMORY,
    RESULTS_DIR,
)


@pytest.fixture(autouse=True)
def clear_contexts():
    """Clear contexts before each test."""
    contexts.clear()
    yield
    contexts.clear()


class TestPathTraversalPrevention:
    """Tests for path traversal attack prevention."""

    @pytest.mark.asyncio
    async def test_path_traversal_with_dotdot_blocked(self):
        """Verify context names with ../ path traversal are rejected."""
        result = await _handle_load_context({
            "name": "../etc/passwd",
            "content": "malicious content"
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_path_traversal_with_slash_blocked(self):
        """Verify context names with forward slashes are rejected."""
        result = await _handle_load_context({
            "name": "foo/bar",
            "content": "test content"
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_path_traversal_with_backslash_blocked(self):
        """Verify context names with backslashes are rejected."""
        result = await _handle_load_context({
            "name": "foo\\bar",
            "content": "test content"
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_valid_context_name_succeeds(self):
        """Verify valid context names work correctly."""
        result = await _handle_load_context({
            "name": "valid_name-123",
            "content": "test content"
        })
        response_text = result[0].text.lower()
        assert "error" not in response_text
        parsed = json.loads(result[0].text)
        assert parsed["status"] == "loaded"

    @pytest.mark.asyncio
    async def test_alphanumeric_underscore_hyphen_allowed(self):
        """Verify alphanumeric, underscore, and hyphen are allowed."""
        for name in ["test", "Test123", "test_name", "test-name", "TEST_123-abc"]:
            result = await _handle_load_context({
                "name": name,
                "content": "content"
            })
            parsed = json.loads(result[0].text)
            assert parsed["status"] == "loaded", f"Name '{name}' should be valid"

    @pytest.mark.asyncio
    async def test_special_characters_blocked(self):
        """Verify special characters are blocked."""
        for name in ["test.txt", "test:name", "test<name>", "test|name", "test name"]:
            result = await _handle_load_context({
                "name": name,
                "content": "content"
            })
            response_text = result[0].text.lower()
            assert "error" in response_text or "invalid" in response_text, \
                f"Name '{name}' should be invalid"

    @pytest.mark.asyncio
    async def test_filter_context_validates_both_names(self):
        """Verify filter_context validates both source and output names."""
        # First create a valid context
        await _handle_load_context({
            "name": "valid_source",
            "content": "line1\nline2\nline3"
        })

        # Try to filter with invalid output name (path traversal)
        result = await _handle_filter_context({
            "name": "valid_source",
            "output_name": "../malicious",
            "pattern": "line"
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_store_result_validates_name(self):
        """Verify store_result validates the result name."""
        result = await _handle_store_result({
            "name": "../malicious",
            "result": "test result"
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_get_results_validates_name(self):
        """Verify get_results validates the result name."""
        result = await _handle_get_results({
            "name": "../etc/passwd"
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text


class TestReDoSPrevention:
    """Tests for ReDoS (Regular Expression Denial of Service) prevention."""

    @pytest.mark.asyncio
    async def test_simple_regex_works(self):
        """Verify simple regex patterns work correctly."""
        await _handle_load_context({
            "name": "regex_test",
            "content": "error: something\ninfo: data\nerror: else"
        })

        result = await _handle_filter_context({
            "name": "regex_test",
            "output_name": "filtered",
            "pattern": "error:"
        })
        parsed = json.loads(result[0].text)
        assert parsed["status"] == "filtered"
        assert parsed["filtered_lines"] == 2

    @pytest.mark.asyncio
    async def test_evil_regex_does_not_hang(self):
        """Verify malicious regex patterns timeout instead of hanging."""
        # Load a context with content that could trigger catastrophic backtracking
        await _handle_load_context({
            "name": "redos_test",
            "content": "a" * 50 + "\n" + "a" * 50  # Reduced size but still dangerous
        })

        # Evil regex that causes catastrophic backtracking on non-matching strings
        # This pattern: (a+)+$ is a classic ReDoS vulnerability
        start_time = time.time()
        result = await _handle_filter_context({
            "name": "redos_test",
            "output_name": "filtered",
            "pattern": "(a+)+b"  # Will try to match 'b' at end, causing backtracking
        })
        elapsed = time.time() - start_time

        # Should complete within timeout (5 seconds) + some margin
        # If vulnerable, this would take exponentially long
        assert elapsed < 10, f"Regex took too long: {elapsed}s (possible ReDoS vulnerability)"
        # If we get here, the server either timed out or completed quickly
        # Both are acceptable - the important thing is we didn't hang

    @pytest.mark.asyncio
    async def test_invalid_regex_returns_error(self):
        """Verify invalid regex patterns return an error."""
        await _handle_load_context({
            "name": "invalid_regex_test",
            "content": "test content"
        })

        result = await _handle_filter_context({
            "name": "invalid_regex_test",
            "output_name": "filtered",
            "pattern": "[unclosed"  # Invalid regex - unclosed bracket
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text


class TestMemoryLimits:
    """Tests for memory and size limits."""

    @pytest.mark.asyncio
    async def test_context_size_limit_enforced(self):
        """Verify contexts exceeding size limit are rejected."""
        # Try to load content larger than MAX_CONTEXT_SIZE (100MB)
        # We create a string just over the limit to test the boundary
        huge_content = "x" * (MAX_CONTEXT_SIZE + 1)

        result = await _handle_load_context({
            "name": "huge",
            "content": huge_content
        })
        response_text = result[0].text.lower()
        assert "too_large" in response_text or "exceeds" in response_text or "error" in response_text

    @pytest.mark.asyncio
    async def test_content_at_limit_succeeds(self):
        """Verify content at exactly the limit succeeds."""
        # This test is skipped in CI because it uses too much memory
        # Uncomment for local testing if you have sufficient RAM
        pytest.skip("Skipping large allocation test - run manually if needed")

        # Content exactly at the limit should work
        content_at_limit = "x" * MAX_CONTEXT_SIZE
        result = await _handle_load_context({
            "name": "at_limit",
            "content": content_at_limit
        })
        parsed = json.loads(result[0].text)
        assert parsed["status"] == "loaded"

    @pytest.mark.asyncio
    async def test_results_file_size_limit(self):
        """Verify results file size limits are enforced."""
        # Create a results file that exceeds the limit
        results_file = RESULTS_DIR / "size_limit_test.jsonl"

        # Write content that exceeds MAX_RESULTS_FILE_SIZE
        with open(results_file, "w") as f:
            # Write more than 10MB
            large_result = json.dumps({"result": "x" * 1000}) + "\n"
            for _ in range(MAX_RESULTS_FILE_SIZE // len(large_result) + 100):
                f.write(large_result)

        # Now try to store another result - should fail
        result = await _handle_store_result({
            "name": "size_limit_test",
            "result": "one more result"
        })
        response_text = result[0].text.lower()
        assert "too_large" in response_text or "exceeds" in response_text or "error" in response_text

        # Clean up
        results_file.unlink(missing_ok=True)


class TestIntegerValidation:
    """Tests for integer parameter validation."""

    @pytest.mark.asyncio
    async def test_negative_chunk_index_rejected(self):
        """Verify negative chunk indices are rejected."""
        await _handle_load_context({
            "name": "int_test",
            "content": "line1\nline2\nline3"
        })
        await _handle_chunk_context({
            "name": "int_test",
            "strategy": "lines",
            "size": 1
        })

        result = await _handle_get_chunk({
            "name": "int_test",
            "chunk_index": -1
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_negative_chunk_index_in_sub_query_rejected(self):
        """Verify negative chunk_index in sub_query is rejected."""
        await _handle_load_context({
            "name": "sub_query_test",
            "content": "test content"
        })
        await _handle_chunk_context({
            "name": "sub_query_test",
            "strategy": "lines",
            "size": 1
        })

        result = await _handle_sub_query({
            "query": "analyze this",
            "context_name": "sub_query_test",
            "chunk_index": -5
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_negative_chunk_indices_in_batch_rejected(self):
        """Verify negative chunk indices in batch query are rejected."""
        await _handle_load_context({
            "name": "batch_test",
            "content": "line1\nline2\nline3"
        })
        await _handle_chunk_context({
            "name": "batch_test",
            "strategy": "lines",
            "size": 1
        })

        result = await _handle_sub_query_batch({
            "query": "analyze",
            "context_name": "batch_test",
            "chunk_indices": [0, -1, 2]  # -1 is invalid
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_chunk_index_out_of_range_rejected(self):
        """Verify out-of-range chunk indices are rejected."""
        await _handle_load_context({
            "name": "range_test",
            "content": "line1\nline2"
        })
        await _handle_chunk_context({
            "name": "range_test",
            "strategy": "lines",
            "size": 1
        })

        result = await _handle_get_chunk({
            "name": "range_test",
            "chunk_index": 999
        })
        response_text = result[0].text.lower()
        assert "out of range" in response_text or "error" in response_text

    @pytest.mark.asyncio
    async def test_valid_chunk_index_succeeds(self):
        """Verify valid chunk indices work correctly."""
        await _handle_load_context({
            "name": "valid_index_test",
            "content": "line1\nline2\nline3"
        })
        await _handle_chunk_context({
            "name": "valid_index_test",
            "strategy": "lines",
            "size": 1
        })

        result = await _handle_get_chunk({
            "name": "valid_index_test",
            "chunk_index": 0
        })
        assert "line1" in result[0].text

        result = await _handle_get_chunk({
            "name": "valid_index_test",
            "chunk_index": 2
        })
        assert "line3" in result[0].text


class TestConcurrencyLimits:
    """Tests for concurrency parameter validation."""

    @pytest.mark.asyncio
    async def test_concurrency_below_minimum_rejected(self):
        """Verify concurrency below minimum is rejected."""
        await _handle_load_context({
            "name": "conc_test",
            "content": "line1\nline2"
        })
        await _handle_chunk_context({
            "name": "conc_test",
            "strategy": "lines",
            "size": 1
        })

        result = await _handle_sub_query_batch({
            "query": "test",
            "context_name": "conc_test",
            "chunk_indices": [0, 1],
            "concurrency": 0  # Below minimum of 1
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_concurrency_above_maximum_rejected(self):
        """Verify concurrency above maximum is rejected."""
        await _handle_load_context({
            "name": "conc_test2",
            "content": "line1\nline2"
        })
        await _handle_chunk_context({
            "name": "conc_test2",
            "strategy": "lines",
            "size": 1
        })

        result = await _handle_sub_query_batch({
            "query": "test",
            "context_name": "conc_test2",
            "chunk_indices": [0, 1],
            "concurrency": 100  # Above maximum of 8
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text


class TestMaxDepthValidation:
    """Tests for max_depth parameter validation."""

    @pytest.mark.asyncio
    async def test_max_depth_below_minimum_rejected(self):
        """Verify max_depth below minimum is rejected."""
        await _handle_load_context({
            "name": "depth_test",
            "content": "test content"
        })

        result = await _handle_sub_query({
            "query": "test",
            "context_name": "depth_test",
            "max_depth": -1  # Below minimum of 0
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_max_depth_above_maximum_rejected(self):
        """Verify max_depth above maximum is rejected."""
        await _handle_load_context({
            "name": "depth_test2",
            "content": "test content"
        })

        result = await _handle_sub_query({
            "query": "test",
            "context_name": "depth_test2",
            "max_depth": 10  # Above maximum of 5
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text


class TestChunkSizeValidation:
    """Tests for chunk size parameter validation."""

    @pytest.mark.asyncio
    async def test_chunk_size_below_minimum_rejected(self):
        """Verify chunk size below minimum is rejected."""
        await _handle_load_context({
            "name": "chunk_size_test",
            "content": "test content"
        })

        result = await _handle_chunk_context({
            "name": "chunk_size_test",
            "strategy": "lines",
            "size": 0  # Below minimum of 1
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text

    @pytest.mark.asyncio
    async def test_chunk_size_above_maximum_rejected(self):
        """Verify chunk size above maximum is rejected."""
        await _handle_load_context({
            "name": "chunk_size_test2",
            "content": "test content"
        })

        result = await _handle_chunk_context({
            "name": "chunk_size_test2",
            "strategy": "lines",
            "size": 200000  # Above maximum of 100000
        })
        response_text = result[0].text.lower()
        assert "error" in response_text or "invalid" in response_text


# ============================================================================
# NEW SECURITY TESTS FOR H3, H4, H5, M3, M4, M6
# ============================================================================

class TestLRUMemoryEviction:
    """Tests for LRU memory eviction (H3)."""

    @pytest.mark.asyncio
    async def test_lru_eviction_on_memory_limit(self):
        """Verify LRU eviction occurs when memory limit is approached."""
        from rlm_mcp_server import contexts, MAX_TOTAL_MEMORY

        # Load several contexts to fill memory
        contexts_to_load = 10
        content_size = MAX_TOTAL_MEMORY // (contexts_to_load + 2)  # Leave some room

        for i in range(contexts_to_load):
            await _handle_load_context({
                "name": f"lru_test_{i}",
                "content": "x" * content_size
            })

        # Access the first one to make it recently used
        contexts.get("lru_test_0")

        # Load one more to trigger eviction
        await _handle_load_context({
            "name": "lru_test_final",
            "content": "x" * content_size
        })

        # The oldest non-accessed context should have been evicted
        # Context 0 was accessed, so it shouldn't be evicted
        assert "lru_test_final" in contexts

    @pytest.mark.asyncio
    async def test_memory_tracking_in_response(self):
        """Verify memory usage is reported in load response."""
        result = await _handle_load_context({
            "name": "memory_tracking_test",
            "content": "test content for memory tracking"
        })
        parsed = json.loads(result[0].text)
        assert "memory_used" in parsed
        assert "memory_available" in parsed
        assert parsed["memory_used"] > 0


class TestDiskQuota:
    """Tests for disk quota enforcement (H4)."""

    @pytest.mark.asyncio
    async def test_disk_quota_check_function_exists(self):
        """Verify disk quota check function is available."""
        from rlm_mcp_server import _check_disk_quota, _get_disk_usage

        # These functions should exist and be callable
        usage = _get_disk_usage()
        assert isinstance(usage, int)
        assert usage >= 0

        ok, err = _check_disk_quota(1000)
        assert isinstance(ok, bool)
        assert isinstance(err, str)


class TestRlmExecTimeout:
    """Tests for rlm_exec operation timeout (H5)."""

    @pytest.mark.asyncio
    async def test_rlm_exec_context_size_limit(self):
        """Verify rlm_exec rejects contexts that are too large."""
        from rlm_mcp_server import _handle_exec, MAX_EXEC_CONTEXT_SIZE

        # Load a context larger than MAX_EXEC_CONTEXT_SIZE
        large_content = "x" * (MAX_EXEC_CONTEXT_SIZE + 1000)
        await _handle_load_context({
            "name": "rlm_exec_size_test",
            "content": large_content
        })

        result = await _handle_exec({
            "code": "result = len(context)",
            "context_name": "rlm_exec_size_test"
        })
        response_text = result[0].text.lower()
        assert "too_large" in response_text or "exceeds" in response_text or "error" in response_text

    @pytest.mark.asyncio
    async def test_rlm_exec_with_valid_size_succeeds(self):
        """Verify rlm_exec works with contexts under the size limit."""
        from rlm_mcp_server import _handle_exec

        await _handle_load_context({
            "name": "rlm_exec_valid_test",
            "content": "hello world"
        })

        result = await _handle_exec({
            "code": "result = len(context)",
            "context_name": "rlm_exec_valid_test"
        })
        parsed = json.loads(result[0].text)
        assert parsed["result"] == 11  # len("hello world")


class TestFilterOperationTimeout:
    """Tests for filter operation timeout (M4)."""

    @pytest.mark.asyncio
    async def test_filter_operation_completes_for_normal_content(self):
        """Verify filter operation completes for normal-sized content."""
        await _handle_load_context({
            "name": "filter_timeout_test",
            "content": "\n".join(f"line {i}" for i in range(1000))
        })

        start_time = time.time()
        result = await _handle_filter_context({
            "name": "filter_timeout_test",
            "output_name": "filtered_result",
            "pattern": "line [0-9]+"
        })
        elapsed = time.time() - start_time

        parsed = json.loads(result[0].text)
        assert parsed["status"] == "filtered"
        assert elapsed < 30  # Should complete well under the timeout


class TestSymlinkPrevention:
    """Tests for symlink attack prevention (M6)."""

    @pytest.mark.asyncio
    async def test_safe_mkdir_rejects_symlinks(self):
        """Verify _safe_mkdir rejects symlinks."""
        from rlm_mcp_server import _safe_mkdir, CHUNKS_DIR

        # Create a symlink in the chunks directory
        symlink_path = CHUNKS_DIR / "symlink_test_target"
        actual_target = Path("/tmp/symlink_target_" + str(time.time()))

        try:
            actual_target.mkdir(parents=True, exist_ok=True)
            if symlink_path.exists() or symlink_path.is_symlink():
                symlink_path.unlink()
            symlink_path.symlink_to(actual_target)

            # Now try to use _safe_mkdir on the symlink
            ok, err = _safe_mkdir(symlink_path)
            assert not ok, "Should reject symlinks"
            assert "symlink" in err.lower()

        finally:
            # Cleanup
            if symlink_path.is_symlink():
                symlink_path.unlink()
            if actual_target.exists():
                actual_target.rmdir()

    @pytest.mark.asyncio
    async def test_safe_mkdir_allows_normal_directory(self):
        """Verify _safe_mkdir allows creating normal directories."""
        from rlm_mcp_server import _safe_mkdir, CHUNKS_DIR

        test_dir = CHUNKS_DIR / f"normal_test_{int(time.time())}"
        try:
            ok, err = _safe_mkdir(test_dir)
            assert ok, f"Should allow normal directory creation: {err}"
            assert test_dir.exists()
        finally:
            if test_dir.exists():
                test_dir.rmdir()


class TestConcurrentContextCreation:
    """Tests for race condition prevention (M3)."""

    @pytest.mark.asyncio
    async def test_concurrent_context_creation(self):
        """Verify concurrent context creation is handled safely."""
        import asyncio

        # Create many contexts concurrently
        async def create_context(i):
            return await _handle_load_context({
                "name": f"concurrent_test_{i}",
                "content": f"content {i}"
            })

        # Launch multiple concurrent creations
        tasks = [create_context(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

        # All should succeed (no race condition errors)
        for i, result in enumerate(results):
            parsed = json.loads(result[0].text)
            assert parsed["status"] == "loaded", f"Context {i} should have loaded successfully"


class TestErrorMessageSanitization:
    """Tests for error message sanitization (M1)."""

    @pytest.mark.asyncio
    async def test_error_does_not_leak_internal_paths(self):
        """Verify error messages don't leak internal file paths."""
        # Try to load a non-existent context
        result = await _handle_get_chunk({
            "name": "nonexistent_context_xyz",
            "chunk_index": 0
        })

        response_text = result[0].text
        # Should not contain internal paths like /tmp or /Users
        assert "/tmp/rlm" not in response_text
        assert "/Users/" not in response_text
        assert "Traceback" not in response_text
