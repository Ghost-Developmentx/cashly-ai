"""
Vector similarity search service for finding similar conversations.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from sqlalchemy import text
from db.init import get_db_connection

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
        return {
            "conversation_id": self.conversation_id,
            "intent": self.intent,
            "assistant_type": self.assistant_type,
            "similarity_score": self.similarity_score,
            "success_indicator": self.success_indicator,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """Handles vector similarity searches in PostgreSQL."""

    def __init__(self):
        self.db = get_db_connection()
        self.default_limit = 5
        self.min_similarity = 0.7

    def search_similar(
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
            limit: Maximum results (default: 5)
            similarity_threshold: Minimum similarity (default: 0.7)

        Returns:
            List of search results ordered by similarity
        """
        limit = limit or self.default_limit
        threshold = similarity_threshold or self.min_similarity

        try:
            with self.db.get_session() as session:
                # Build the query
                query = self._build_search_query(
                    user_id=user_id, intent_filter=intent_filter
                )

                # Execute search
                results = session.execute(
                    text(query),
                    {
                        "embedding": embedding,
                        "threshold": threshold,
                        "limit": limit,
                        "user_id": user_id,
                        "intent": intent_filter,
                    },
                ).fetchall()

                # Convert to SearchResult objects
                search_results = []
                for row in results:
                    search_results.append(
                        SearchResult(
                            conversation_id=row.conversation_id,
                            intent=row.intent,
                            assistant_type=row.assistant_type,
                            similarity_score=float(row.similarity),
                            success_indicator=row.success_indicator,
                            metadata=row.conversation_metadata or {},
                        )
                    )

                logger.info(f"Found {len(search_results)} similar conversations")
                return search_results

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def search_by_intent(
        self, embedding: List[float], intents: List[str], limit: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """
        Search for similar conversations grouped by intent.

        Args:
            embedding: Query embedding
            intents: List of intents to search
            limit: Max results per intent

        Returns:
            Dictionary mapping intents to search results
        """
        results_by_intent = {}

        for intent in intents:
            results = self.search_similar(
                embedding=embedding, intent_filter=intent, limit=limit
            )
            results_by_intent[intent] = results

        return results_by_intent

    @staticmethod
    def _build_search_query(
        user_id: Optional[str] = None, intent_filter: Optional[str] = None
    ) -> str:
        """Build the similarity search SQL query."""
        query = """
                SELECT
                    conversation_id,
                    intent,
                    assistant_type,
                    success_indicator,
                    conversation_metadata,
                    1 - (embedding <=> embedding::vector) as similarity
                FROM conversation_embeddings
                WHERE 1 - (embedding <=> embedding::vector) >= :threshold \
                """

        # Add filters
        if user_id:
            query += " AND user_id = :user_id"

        if intent_filter:
            query += " AND intent = :intent"

        # Order and limit
        query += " ORDER BY similarity DESC LIMIT :limit"

        return query
