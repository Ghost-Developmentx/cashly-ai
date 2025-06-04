"""
Async storage service for conversation embeddings.
Handles all embedding database operations asynchronously.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from app.db.async_db.vector_ops import AsyncVectorOperations
from app.db.async_db.connection import AsyncDatabaseConnection
from app.core.config import get_settings
from app.db.singleton_registry import registry

logger = logging.getLogger(__name__)


async def get_async_db_connection() -> AsyncDatabaseConnection:
    """
    Get the global asynchronous database connection instance.
    Uses a process-wide singleton registry to ensure true singleton behavior.
    """

    async def create_connection():
        settings = get_settings()
        return AsyncDatabaseConnection(settings)

    return await registry.get_or_create("async_db_connection", create_connection)


class AsyncEmbeddingStorage:
    """
    Handles asynchronous storage and retrieval of conversation embeddings as well as
    related operations. This class is designed to manage embeddings for conversation
    analysis and similarity searches in an efficient and scalable manner.
    """

    def __init__(self):
        self._db = None
        self._vector_ops: Optional[AsyncVectorOperations] = None

    async def _ensure_connection(self):
        """Ensure database connection using a shared singleton."""
        if self._db is None:
            self._db = await get_async_db_connection()

    async def get_vector_ops(self) -> AsyncVectorOperations:
        """Get or create a vector operations handler."""
        await self._ensure_connection()

        if self._vector_ops is None:
            pool = await self._db.get_asyncpg_pool()
            self._vector_ops = AsyncVectorOperations(pool)
        return self._vector_ops

    async def store_embedding(
        self,
        conversation_id: str,
        user_id: str,
        embedding: List[float],
        intent: str,
        assistant_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        success_indicator: bool = True,
    ) -> Optional[int]:
        try:
            vector_ops = await self.get_vector_ops()

            storage_metadata = {
                "conversation_id": conversation_id,
                "user_id": user_id,
                "intent": intent,
                "assistant_type": assistant_type,
                "conversation_metadata": metadata or {},
                "success_indicator": success_indicator,
                "message_count": metadata.get("message_count", 1) if metadata else 1,
                "created_at": datetime.now(),
            }

            embedding_id = await vector_ops.insert_vector(
                "conversation_embeddings", embedding, storage_metadata
            )

            if embedding_id:
                logger.info(
                    f"Stored embedding {embedding_id} for conversation {conversation_id}"
                )

            return embedding_id

        except Exception as e:
            logger.error(f"Failed to store embedding: {e}", exc_info=True)
            return None

    async def find_similar_conversations(
        self,
        embedding: List[float],
        user_id: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.8,
    ) -> List[Dict[str, Any]]:
        try:
            vector_ops = await self.get_vector_ops()

            filters = {"user_id": user_id} if user_id else {}

            results = await vector_ops.search_similar_vectors(
                "conversation_embeddings",
                embedding,
                limit=limit,
                threshold=similarity_threshold,
                filters=filters,
            )

            logger.debug(f"Found {len(results)} similar conversations")
            return results

        except Exception as e:
            logger.error(f"Failed to find similar conversations: {e}", exc_info=True)
            return []

    async def get_user_patterns(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        try:
            await self._ensure_connection()
            cutoff_date = datetime.now() - timedelta(days=days)

            pool = await self._db.get_asyncpg_pool()
            async with pool.acquire() as conn:
                intent_rows = await conn.fetch(
                    """
                    SELECT intent, COUNT(*) as count
                    FROM conversation_embeddings
                    WHERE user_id = $1 AND created_at >= $2
                    GROUP BY intent ORDER BY count DESC
                    """,
                    user_id,
                    cutoff_date,
                )

                assistant_rows = await conn.fetch(
                    """
                    SELECT assistant_type, COUNT(*) as count
                    FROM conversation_embeddings
                    WHERE user_id = $1 AND created_at >= $2
                    GROUP BY assistant_type ORDER BY count DESC
                    """,
                    user_id,
                    cutoff_date,
                )

            return {
                "intent_distribution": {
                    row["intent"]: row["count"] for row in intent_rows
                },
                "assistant_distribution": {
                    row["assistant_type"]: row["count"] for row in assistant_rows
                },
                "total_conversations": sum(row["count"] for row in intent_rows),
                "time_period_days": days,
                "unique_intents": len(intent_rows),
                "unique_assistants": len(assistant_rows),
            }

        except Exception as e:
            logger.error(f"Failed to get user patterns: {e}", exc_info=True)
            return {
                "error": str(e),
                "intent_distribution": {},
                "assistant_distribution": {},
                "total_conversations": 0,
            }

    async def cleanup_old_embeddings(self, days_to_keep: int = 90) -> int:
        try:
            await self._ensure_connection()
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            pool = await self._db.get_asyncpg_pool()
            async with pool.acquire() as conn:
                deleted = await conn.fetchval(
                    """
                    DELETE FROM conversation_embeddings
                    WHERE created_at < $1
                    RETURNING COUNT(*)
                    """,
                    cutoff_date,
                )

                logger.info(f"Cleaned up {deleted} old embeddings")
                return deleted or 0

        except Exception as e:
            logger.error(f"Failed to cleanup embeddings: {e}", exc_info=True)
            return 0

    @staticmethod
    async def close():
        """No-op: connection is managed globally."""
        logger.info("AsyncEmbeddingStorage.close() called (no-op)")
