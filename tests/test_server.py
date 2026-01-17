"""Tests for the RLM MCP server."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Set test data directory before importing server
TEST_DATA_DIR = tempfile.mkdtemp(prefix="rlm_test_")
os.environ["RLM_DATA_DIR"] = TEST_DATA_DIR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlm_mcp_server import (
    _chunk_content,
    _error_response,
    _handle_chunk_context,
    _handle_filter_context,
    _handle_get_chunk,
    _handle_get_results,
    _handle_inspect_context,
    _handle_list_contexts,
    _handle_load_context,
    _handle_store_result,
    _handle_sub_query,
    _handle_sub_query_batch,
    _hash_content,
    _text_response,
    contexts,
)


@pytest.fixture(autouse=True)
def clear_contexts():
    """Clear contexts before each test."""
    contexts.clear()
    yield
    contexts.clear()


class TestPureFunctions:
    """Tests for pure helper functions."""

    def test_hash_content_deterministic(self):
        """Hash should be deterministic for same content."""
        content = "hello world"
        assert _hash_content(content) == _hash_content(content)

    def test_hash_content_different_for_different_content(self):
        """Different content should produce different hashes."""
        assert _hash_content("hello") != _hash_content("world")

    def test_hash_content_length(self):
        """Hash should be 12 characters (truncated SHA256)."""
        assert len(_hash_content("test")) == 12

    def test_text_response_with_string(self):
        """Text response should return string as-is."""
        result = _text_response("hello")
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_text_response_with_dict(self):
        """Text response should JSON-encode dicts."""
        result = _text_response({"key": "value"})
        assert len(result) == 1
        parsed = json.loads(result[0].text)
        assert parsed == {"key": "value"}

    def test_error_response_structure(self):
        """Error response should have error code and message."""
        result = _error_response("TEST_ERROR", "Something went wrong")
        parsed = json.loads(result[0].text)
        assert parsed["error"] == "TEST_ERROR"
        assert parsed["message"] == "Something went wrong"


class TestChunkContent:
    """Tests for the _chunk_content function."""

    def test_chunk_by_lines(self):
        """Chunking by lines should split on newlines."""
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = _chunk_content(content, "lines", 2)
        assert len(chunks) == 3
        assert chunks[0] == "line1\nline2"
        assert chunks[1] == "line3\nline4"
        assert chunks[2] == "line5"

    def test_chunk_by_chars(self):
        """Chunking by chars should split on character count."""
        content = "abcdefghij"
        chunks = _chunk_content(content, "chars", 3)
        assert len(chunks) == 4
        assert chunks[0] == "abc"
        assert chunks[1] == "def"
        assert chunks[2] == "ghi"
        assert chunks[3] == "j"

    def test_chunk_by_paragraphs(self):
        """Chunking by paragraphs should split on blank lines."""
        content = "para1\n\npara2\n\npara3\n\npara4"
        chunks = _chunk_content(content, "paragraphs", 2)
        assert len(chunks) == 2
        assert chunks[0] == "para1\n\npara2"
        assert chunks[1] == "para3\n\npara4"

    def test_chunk_empty_content(self):
        """Empty content should return single empty chunk."""
        chunks = _chunk_content("", "lines", 10)
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_chunk_unknown_strategy(self):
        """Unknown strategy should return empty list."""
        chunks = _chunk_content("test", "unknown", 10)
        assert chunks == []


class TestLoadContext:
    """Tests for context loading."""

    @pytest.mark.asyncio
    async def test_load_context_success(self):
        """Loading a context should store it and return metadata."""
        result = await _handle_load_context({
            "name": "test_ctx",
            "content": "hello\nworld",
        })
        parsed = json.loads(result[0].text)

        assert parsed["status"] == "loaded"
        assert parsed["name"] == "test_ctx"
        assert parsed["length"] == 11
        assert parsed["lines"] == 2
        assert "hash" in parsed

    @pytest.mark.asyncio
    async def test_load_context_stored_in_memory(self):
        """Loaded context should be accessible in memory."""
        await _handle_load_context({
            "name": "mem_test",
            "content": "test content",
        })

        assert "mem_test" in contexts
        assert contexts["mem_test"]["content"] == "test content"


class TestInspectContext:
    """Tests for context inspection."""

    @pytest.mark.asyncio
    async def test_inspect_context_success(self):
        """Inspecting a loaded context should return info."""
        await _handle_load_context({
            "name": "inspect_test",
            "content": "hello world " * 100,
        })

        result = await _handle_inspect_context({
            "name": "inspect_test",
            "preview_chars": 20,
        })
        parsed = json.loads(result[0].text)

        assert parsed["name"] == "inspect_test"
        assert len(parsed["preview"]) == 20
        assert parsed["has_chunks"] is False

    @pytest.mark.asyncio
    async def test_inspect_nonexistent_context(self):
        """Inspecting nonexistent context should return error."""
        result = await _handle_inspect_context({"name": "nonexistent"})
        assert "not found" in result[0].text


class TestChunkContext:
    """Tests for context chunking."""

    @pytest.mark.asyncio
    async def test_chunk_context_success(self):
        """Chunking a context should create chunks."""
        await _handle_load_context({
            "name": "chunk_test",
            "content": "\n".join([f"line{i}" for i in range(10)]),
        })

        result = await _handle_chunk_context({
            "name": "chunk_test",
            "strategy": "lines",
            "size": 3,
        })
        parsed = json.loads(result[0].text)

        assert parsed["status"] == "chunked"
        assert parsed["chunk_count"] == 4
        assert len(parsed["chunks"]) == 4

    @pytest.mark.asyncio
    async def test_chunk_nonexistent_context(self):
        """Chunking nonexistent context should return error."""
        result = await _handle_chunk_context({"name": "nonexistent"})
        assert "not found" in result[0].text


class TestGetChunk:
    """Tests for chunk retrieval."""

    @pytest.mark.asyncio
    async def test_get_chunk_success(self):
        """Getting a chunk should return its content."""
        await _handle_load_context({
            "name": "get_chunk_test",
            "content": "chunk0\nchunk0\nchunk1\nchunk1",
        })
        await _handle_chunk_context({
            "name": "get_chunk_test",
            "strategy": "lines",
            "size": 2,
        })

        result = await _handle_get_chunk({
            "name": "get_chunk_test",
            "chunk_index": 0,
        })
        assert "chunk0" in result[0].text

    @pytest.mark.asyncio
    async def test_get_chunk_out_of_range(self):
        """Getting chunk out of range should return error."""
        await _handle_load_context({
            "name": "range_test",
            "content": "test",
        })
        await _handle_chunk_context({
            "name": "range_test",
            "strategy": "lines",
            "size": 10,
        })

        result = await _handle_get_chunk({
            "name": "range_test",
            "chunk_index": 99,
        })
        assert "out of range" in result[0].text


class TestFilterContext:
    """Tests for context filtering."""

    @pytest.mark.asyncio
    async def test_filter_keep_mode(self):
        """Filter with keep mode should keep matching lines."""
        await _handle_load_context({
            "name": "filter_src",
            "content": "error: something\ninfo: data\nerror: else",
        })

        result = await _handle_filter_context({
            "name": "filter_src",
            "output_name": "errors_only",
            "pattern": "error:",
            "mode": "keep",
        })
        parsed = json.loads(result[0].text)

        assert parsed["filtered_lines"] == 2
        assert "errors_only" in contexts

    @pytest.mark.asyncio
    async def test_filter_remove_mode(self):
        """Filter with remove mode should remove matching lines."""
        await _handle_load_context({
            "name": "filter_src2",
            "content": "error: something\ninfo: data\nerror: else",
        })

        result = await _handle_filter_context({
            "name": "filter_src2",
            "output_name": "no_errors",
            "pattern": "error:",
            "mode": "remove",
        })
        parsed = json.loads(result[0].text)

        assert parsed["filtered_lines"] == 1


class TestStoreAndGetResults:
    """Tests for result storage and retrieval."""

    @pytest.mark.asyncio
    async def test_store_result(self):
        """Storing a result should succeed."""
        result = await _handle_store_result({
            "name": "test_results",
            "result": "found something",
            "metadata": {"chunk": 0},
        })
        assert "stored" in result[0].text

    @pytest.mark.asyncio
    async def test_get_results(self):
        """Getting stored results should return all results."""
        await _handle_store_result({
            "name": "multi_results",
            "result": "result1",
        })
        await _handle_store_result({
            "name": "multi_results",
            "result": "result2",
        })

        result = await _handle_get_results({"name": "multi_results"})
        parsed = json.loads(result[0].text)

        assert parsed["count"] == 2
        assert len(parsed["results"]) == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent_results(self):
        """Getting nonexistent results should return error."""
        result = await _handle_get_results({"name": "nonexistent"})
        assert "No results" in result[0].text


class TestListContexts:
    """Tests for listing contexts."""

    @pytest.mark.asyncio
    async def test_list_contexts_empty(self):
        """Listing with no contexts should return empty list."""
        result = await _handle_list_contexts({})
        parsed = json.loads(result[0].text)
        # May include disk-only contexts, so just check structure
        assert "contexts" in parsed

    @pytest.mark.asyncio
    async def test_list_contexts_with_data(self):
        """Listing should include loaded contexts."""
        await _handle_load_context({"name": "list_test1", "content": "a"})
        await _handle_load_context({"name": "list_test2", "content": "b"})

        result = await _handle_list_contexts({})
        parsed = json.loads(result[0].text)

        names = [c["name"] for c in parsed["contexts"]]
        assert "list_test1" in names
        assert "list_test2" in names


class TestSubQuery:
    """Tests for sub-query handler."""

    @pytest.mark.asyncio
    async def test_sub_query_context_not_found(self):
        """Sub-query on nonexistent context should error."""
        result = await _handle_sub_query({
            "query": "test",
            "context_name": "nonexistent",
        })
        parsed = json.loads(result[0].text)
        assert parsed["error"] == "context_not_found"

    @pytest.mark.asyncio
    async def test_sub_query_chunk_not_available(self):
        """Sub-query on non-chunked context with chunk_index should error."""
        await _handle_load_context({"name": "no_chunks", "content": "test"})

        result = await _handle_sub_query({
            "query": "test",
            "context_name": "no_chunks",
            "chunk_index": 0,
        })
        parsed = json.loads(result[0].text)
        assert parsed["error"] == "chunk_not_available"

    @pytest.mark.asyncio
    async def test_sub_query_with_mock_provider(self):
        """Sub-query should call provider and return response."""
        await _handle_load_context({"name": "query_test", "content": "test content"})

        with patch("rlm_mcp_server._make_provider_call", new_callable=AsyncMock) as mock:
            mock.return_value = ("mocked response", None)

            result = await _handle_sub_query({
                "query": "what is this?",
                "context_name": "query_test",
                "provider": "ollama",
            })
            parsed = json.loads(result[0].text)

            assert parsed["response"] == "mocked response"
            mock.assert_called_once()


class TestSubQueryBatch:
    """Tests for batch sub-query handler."""

    @pytest.mark.asyncio
    async def test_batch_query_context_not_found(self):
        """Batch query on nonexistent context should error."""
        result = await _handle_sub_query_batch({
            "query": "test",
            "context_name": "nonexistent",
            "chunk_indices": [0, 1],
        })
        parsed = json.loads(result[0].text)
        assert parsed["error"] == "context_not_found"

    @pytest.mark.asyncio
    async def test_batch_query_not_chunked(self):
        """Batch query on non-chunked context should error."""
        await _handle_load_context({"name": "not_chunked", "content": "test"})

        result = await _handle_sub_query_batch({
            "query": "test",
            "context_name": "not_chunked",
            "chunk_indices": [0],
        })
        parsed = json.loads(result[0].text)
        assert parsed["error"] == "context_not_chunked"

    @pytest.mark.asyncio
    async def test_batch_query_invalid_indices(self):
        """Batch query with invalid indices should error."""
        await _handle_load_context({
            "name": "batch_test",
            "content": "line1\nline2",
        })
        await _handle_chunk_context({
            "name": "batch_test",
            "strategy": "lines",
            "size": 1,
        })

        result = await _handle_sub_query_batch({
            "query": "test",
            "context_name": "batch_test",
            "chunk_indices": [0, 99],
        })
        parsed = json.loads(result[0].text)
        assert parsed["error"] == "invalid_chunk_indices"

    @pytest.mark.asyncio
    async def test_batch_query_with_mock_provider(self):
        """Batch query should process all chunks."""
        await _handle_load_context({
            "name": "batch_success",
            "content": "chunk1\nchunk2\nchunk3",
        })
        await _handle_chunk_context({
            "name": "batch_success",
            "strategy": "lines",
            "size": 1,
        })

        with patch("rlm_mcp_server._make_provider_call", new_callable=AsyncMock) as mock:
            mock.return_value = ("response", None)

            result = await _handle_sub_query_batch({
                "query": "analyze",
                "context_name": "batch_success",
                "chunk_indices": [0, 1, 2],
                "concurrency": 2,
            })
            parsed = json.loads(result[0].text)

            assert parsed["status"] == "completed"
            assert parsed["total_chunks"] == 3
            assert parsed["successful"] == 3
            assert parsed["failed"] == 0
            assert mock.call_count == 3
