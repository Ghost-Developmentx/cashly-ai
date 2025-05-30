"""
Async storage service for conversation embeddings.
Handles all embedding database operations asynchronously.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from db.async_db.connection import AsyncDatabaseConnection
from db.async_db.vector_ops import AsyncVectorOperations

logger = logging.getLogger(__name__)


class AsyncEmbeddingStorage:
    """Async storage for conversation embeddings."""

    def __init__(self, db_connection: Optional[AsyncDatabaseConnection] = None):
        self.db = db_connection or AsyncDatabaseConnection()
        self._vector_ops: Optional[AsyncVectorOperations] = None

    async def get_vector_ops(self) -> AsyncVectorOperations:
        """Get or create a vector operations handler."""
        if self._vector_ops is None:
            pool = await self.db.get_asyncpg_pool()
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
        """
        Store a conversation embedding asynchronously.

        Args:
            conversation_id: Unique conversation ID
            user_id: User identifier
            embedding: Embedding vector
            intent: Classified intent
            assistant_type: Assistant that handled query
            metadata: Additional metadata
            success_indicator: Success flag

        Returns:
            ID of stored embedding
        """
        try:
            vector_ops = await self.get_vector_ops()

            # Prepare metadata
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

            # Store using vector operations
            embedding_id = await vector_ops.insert_vector(
                "conversation_embeddings", embedding, storage_metadata
            )

            if embedding_id:
                logger.info(
                    f"Stored embedding {embedding_id} for "
                    f"conversation {conversation_id}"
                )

            return embedding_id

        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return None

    async def find_similar_conversations(
        self,
        embedding: List[float],
        user_id: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """
        Find similar conversations using vector similarity.

        Args:
            embedding: Query embedding
            user_id: Optional user filter
            limit: Maximum results
            similarity_threshold: Minimum similarity

        Returns:
            List of similar conversations
        """
        try:
            vector_ops = await self.get_vector_ops()

            # Prepare filters
            filters = {}
            if user_id:
                filters["user_id"] = user_id

            # Search similar vectors
            results = await vector_ops.search_similar_vectors(
                "conversation_embeddings",
                embedding,
                limit=limit,
                threshold=similarity_threshold,
                filters=filters,
            )

            logger.info(f"Found {len(results)} similar conversations")
            return results

        except Exception as e:
            logger.error(f"Failed to find similar conversations: {e}")
            return []

    async def get_user_patterns(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """
        Get conversation patterns for a user.

        Args:
            user_id: User identifier
            days: Days to look back

        Returns:
            Pattern analysis
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            async with self.db.get_session() as session:
                # Intent distribution query
                intent_query = """
                               SELECT intent, COUNT(*) as count
                               FROM conversation_embeddings
                               WHERE user_id = $1 AND created_at >= $2
                               GROUP BY intent
                               ORDER BY count DESC \
                               """

                # Execute query
                pool = await self.db.get_asyncpg_pool()
                async with pool.acquire() as conn:
                    intent_rows = await conn.fetch(intent_query, user_id, cutoff_date)

                    # Assistant distribution query
                    assistant_query = """
                                      SELECT assistant_type, COUNT(*) as count
                                      FROM conversation_embeddings
                                      WHERE user_id = $1 AND created_at >= $2
                                      GROUP BY assistant_type
                                      ORDER BY count DESC \
                                      """

                    assistant_rows = await conn.fetch(
                        assistant_query, user_id, cutoff_date
                    )

                # Format results
                intent_distribution = {
                    row["intent"]: row["count"] for row in intent_rows
                }

                assistant_distribution = {
                    row["assistant_type"]: row["count"] for row in assistant_rows
                }

                return {
                    "intent_distribution": intent_distribution,
                    "assistant_distribution": assistant_distribution,
                    "total_conversations": sum(intent_distribution.values()),
                    "time_period_days": days,
                    "unique_intents": len(intent_distribution),
                    "unique_assistants": len(assistant_distribution),
                }

        except Exception as e:
            logger.error(f"Failed to get user patterns: {e}")
            return {
                "error": str(e),
                "intent_distribution": {},
                "assistant_distribution": {},
                "total_conversations": 0,
            }

    async def cleanup_old_embeddings(self, days_to_keep: int = 90) -> int:
        """
        Clean up old embeddings.

        Args:
            days_to_keep: Keep embeddings from last N days

        Returns:
            Number of deleted records
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            pool = await self.db.get_asyncpg_pool()
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
            logger.error(f"Failed to cleanup embeddings: {e}")
            return 0

    async def close(self):
        """Close database connections."""
        await self.db.close()
