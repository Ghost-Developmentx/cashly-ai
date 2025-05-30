"""
Async vector similarity search service.
High-performance similarity search for embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from db.async_db.connection import AsyncDatabaseConnection
from db.async_db.vector_ops import AsyncVectorOperations

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a similarity search result."""

    conversation_id: str
    intent: str
    assistant_type: str
    similarity_score: float
    success_indicator: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AsyncVectorSearchService:
    """Async vector similarity search service."""

    def __init__(
        self,
        db_connection: Optional[AsyncDatabaseConnection] = None,
        default_limit: int = 5,
        min_similarity: float = 0.6,
    ):
        self.db = db_connection or AsyncDatabaseConnection()
        self.default_limit = default_limit
        self.min_similarity = min_similarity
        self._vector_ops: Optional[AsyncVectorOperations] = None

    async def get_vector_ops(self) -> AsyncVectorOperations:
        """Get or create a vector operations handler."""
        if self._vector_ops is None:
            pool = await self.db.get_asyncpg_pool()
            self._vector_ops = AsyncVectorOperations(pool)
        return self._vector_ops

    async def search_similar(
        self,
        embedding: List[float],
        user_id: Optional[str] = None,
        intent_filter: Optional[str] = None,
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        Search for similar conversation embeddings.

        Args:
            embedding: Query embedding vector
            user_id: Optional user filter
            intent_filter: Optional intent filter
            limit: Maximum results
            similarity_threshold: Minimum similarity

        Returns:
            List of search results
        """
        limit = limit or self.default_limit
        threshold = similarity_threshold or self.min_similarity

        try:
            # Build filters
            filters = {}
            if user_id:
                filters["user_id"] = user_id
            if intent_filter:
                filters["intent"] = intent_filter

            # Perform search
            vector_ops = await self.get_vector_ops()
            results = await vector_ops.search_similar_vectors(
                "conversation_embeddings",
                embedding,
                limit=limit,
                threshold=threshold,
                filters=filters,
            )

            # Convert to SearchResult objects
            search_results = []
            for row in results:
                search_results.append(
                    SearchResult(
                        conversation_id=row["conversation_id"],
                        intent=row["intent"],
                        assistant_type=row["assistant_type"],
                        similarity_score=float(row["similarity"]),
                        success_indicator=row["success_indicator"],
                        metadata=row.get("conversation_metadata", {}),
                    )
                )

            logger.info(
                f"Found {len(search_results)} similar conversations "
                f"(threshold: {threshold})"
            )
            return search_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}", exc_info=True)
            return []

    async def search_by_intent(
        self, embedding: List[float], intents: List[str], limit: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """
        Search for similar conversations grouped by intent.

        Args:
            embedding: Query embedding
            intents: List of intents to search
            limit: Max results per intent

        Returns:
            Dictionary mapping intents to results
        """
        results_by_intent = {}

        search_tasks = []

        for intent in intents:
            task = self.search_similar(
                embedding=embedding, intent_filter=intent, limit=limit
            )
            search_tasks.append((intent, task))

        # Wait for all searches
        for intent, task in search_tasks:
            results = await task
            results_by_intent[intent] = results

        return results_by_intent

    async def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics from database."""
        try:
            pool = await self.db.get_asyncpg_pool()
            async with pool.acquire() as conn:
                # Total embeddings
                total = await conn.fetchval(
                    "SELECT COUNT(*) FROM conversation_embeddings"
                )

                # By intent
                intent_stats = await conn.fetch(
                    """
                    SELECT intent, COUNT(*) as count
                    FROM conversation_embeddings
                    GROUP BY intent
                    """
                )

                # By assistant
                assistant_stats = await conn.fetch(
                    """
                    SELECT assistant_type, COUNT(*) as count
                    FROM conversation_embeddings
                    GROUP BY assistant_type
                    """
                )

                return {
                    "total_embeddings": total,
                    "by_intent": {row["intent"]: row["count"] for row in intent_stats},
                    "by_assistant": {
                        row["assistant_type"]: row["count"] for row in assistant_stats
                    },
                }

        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}

    async def close(self):
        """Close database connections."""
        await self.db.close()
