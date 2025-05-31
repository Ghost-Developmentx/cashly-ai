"""
Async vector similarity search service.
High-performance similarity search for embeddings.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

from db.async_db import AsyncDatabaseConnection
from db.init import get_async_db_connection
from db.singleton_registry import registry
from db.async_db.vector_ops import AsyncVectorOperations

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    conversation_id: str
    intent: str
    assistant_type: str
    similarity_score: float
    success_indicator: bool
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AsyncVectorSearchService:
    def __init__(self, default_limit: int = 5, min_similarity: float = 0.6):
        self.default_limit = default_limit
        self.min_similarity = min_similarity
        self._db: Optional[AsyncDatabaseConnection] = None
        self._vector_ops: Optional[AsyncVectorOperations] = None
        logger.info("✅ AsyncVectorSearchService created")

    async def _ensure_connection(self) -> None:
        """Ensure a database connection is established."""
        if self._db is None:
            self._db = await get_async_db_connection()
            logger.info("✅ AsyncVectorSearchService connected to shared DB")

    async def _get_vector_ops(self) -> AsyncVectorOperations:
        """Get or create a vector operations handler."""
        await self._ensure_connection()
        if self._vector_ops is None:
            pool = await self._db.get_asyncpg_pool()
            self._vector_ops = AsyncVectorOperations(pool)
        return self._vector_ops

    # ------------------------------- API calls -------------------------------- #
    async def search_similar(
        self,
        embedding: List[float],
        user_id: Optional[str] = None,
        intent_filter: Optional[str] = None,
        limit: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Search for similar vectors."""
        limit = limit or self.default_limit
        threshold = similarity_threshold or self.min_similarity

        filters: Dict[str, Any] = {}
        if user_id:
            filters["user_id"] = user_id
        if intent_filter:
            filters["intent"] = intent_filter

        try:
            vector_ops = await self._get_vector_ops()
            rows = await vector_ops.search_similar_vectors(
                "conversation_embeddings",
                embedding,
                limit=limit,
                threshold=threshold,
                filters=filters,
            )

            return [
                SearchResult(
                    conversation_id=row["conversation_id"],
                    intent=row["intent"],
                    assistant_type=row["assistant_type"],
                    similarity_score=float(row["similarity"]),
                    success_indicator=row["success_indicator"],
                    metadata=row.get("conversation_metadata", {}),
                )
                for row in rows
            ]
        except Exception as e:
            logger.error("Vector search failed: %s", e, exc_info=True)
            return []

    async def search_by_intent(
        self, embedding: List[float], intents: List[str], limit: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """Search by multiple intents."""
        results: Dict[str, List[SearchResult]] = {}
        for intent in intents:
            results[intent] = await self.search_similar(
                embedding=embedding, intent_filter=intent, limit=limit
            )
        return results

    async def get_statistics(self) -> Dict[str, Any]:
        """Get search statistics."""
        try:
            await self._ensure_connection()
            pool = await self._db.get_asyncpg_pool()
            async with pool.acquire() as conn:
                total = await conn.fetchval(
                    "SELECT COUNT(*) FROM conversation_embeddings"
                )
                intents = await conn.fetch(
                    "SELECT intent, COUNT(*) AS c FROM conversation_embeddings GROUP BY intent"
                )
                assistants = await conn.fetch(
                    "SELECT assistant_type, COUNT(*) AS c FROM conversation_embeddings GROUP BY assistant_type"
                )
            return {
                "total_embeddings": total,
                "by_intent": {r["intent"]: r["c"] for r in intents},
                "by_assistant": {r["assistant_type"]: r["c"] for r in assistants},
            }
        except Exception as e:
            logger.error("Statistics query failed: %s", e, exc_info=True)
            return {"error": str(e)}

    @classmethod
    async def get_instance(cls) -> "AsyncVectorSearchService":
        async def factory():
            return cls()

        return await registry.get_or_create("vector_search_service", factory)

    @staticmethod
    async def close():
        """No-op: pool is managed by shared connection."""
        logger.info("AsyncVectorSearchService.close() called (no-op)")
