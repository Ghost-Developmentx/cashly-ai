import pytest
from unittest.mock import AsyncMock
from contextlib import asynccontextmanager

from app.db.async_db.vector_ops import AsyncVectorOperations


def make_pool(mock_conn):
    @asynccontextmanager
    async def acquire():
        yield mock_conn

    pool = AsyncMock()
    pool.acquire = acquire
    return pool


@pytest.mark.asyncio
async def test_create_vector_extension():
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=0)
    mock_conn.execute = AsyncMock()
    pool = make_pool(mock_conn)
    ops = AsyncVectorOperations(pool)

    result = await ops.create_vector_extension()

    mock_conn.fetchval.assert_awaited_with(
        "SELECT COUNT(*) FROM pg_extension WHERE extname = 'vector'"
    )
    mock_conn.execute.assert_awaited_with("CREATE EXTENSION IF NOT EXISTS vector")
    assert result is True


@pytest.mark.asyncio
async def test_create_vector_index():
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    pool = make_pool(mock_conn)
    ops = AsyncVectorOperations(pool)

    result = await ops.create_vector_index(
        "conversation_embeddings", "embedding", index_type="ivfflat", lists=100
    )

    index_name = "idx_conversation_embeddings_embedding_vector"
    expected_query = f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON conversation_embeddings
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """

    executed_query = mock_conn.execute.await_args.args[0]
    assert executed_query.strip() == expected_query.strip()
    assert result is True


@pytest.mark.asyncio
async def test_insert_search_update():
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=42)
    mock_conn.fetch = AsyncMock(
        return_value=[{"id": 42, "embedding": [0.1, 0.2], "tag": "x"}]
    )
    mock_conn.execute = AsyncMock()
    pool = make_pool(mock_conn)
    ops = AsyncVectorOperations(pool)

    insert_id = await ops.insert_vector(
        "conversation_embeddings", [0.1, 0.2], {"tag": "x"}
    )
    assert insert_id == 42

    results = await ops.search_similar_vectors(
        "conversation_embeddings", [0.1, 0.2], limit=1, filters={"tag": "x"}
    )
    assert results == [{"id": 42, "tag": "x"}]

    updated = await ops.update_vector_metadata(
        "conversation_embeddings", 42, {"tag": "y"}
    )
    mock_conn.execute.assert_awaited()
    assert updated is True