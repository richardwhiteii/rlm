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
    _handle_filter_context,
    _handle_get_chunk,
    _handle_get_results,
    _handle_load_context,
    _handle_store_result,
    _handle_sub_query,
    _handle_sub_query_batch,
    contexts,
    MAX_CONTEXT_SIZE,
    MAX_RESULTS_FILE_SIZE,
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
