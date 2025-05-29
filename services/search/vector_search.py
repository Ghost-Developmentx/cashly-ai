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
        self.min_similarity = 0.6

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
                # Build the query with FIXED SQL
                query = self._build_search_query(
                    user_id=user_id, intent_filter=intent_filter
                )

                # Execute search with proper parameter binding
                results = session.execute(
                    text(query),
                    {
                        "query_embedding": embedding,
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
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
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
        """Build the similarity search SQL query with FIXED comparison."""
        # FIXED: Compare with the query embedding parameter, not self-comparison
        query = """
    SELECT
        conversation_id,
        intent,
        assistant_type,
        success_indicator,
        conversation_metadata,
        1 - (embedding <=> (:query_embedding)::vector) as similarity
    FROM conversation_embeddings
    WHERE 1 - (embedding <=> (:query_embedding)::vector) >= :threshold
"""

        # Add filters
        if user_id:
            query += " AND user_id = :user_id"

        if intent_filter:
            query += " AND intent = :intent"

        # Order and limit
        query += " ORDER BY similarity DESC LIMIT :limit"

        return query

    def test_search(self, test_query: str = "Show me my invoices") -> Dict[str, Any]:
        """Test the vector search functionality."""
        try:
            from services.embeddings.openai_client import OpenAIEmbeddingClient

            # Generate embedding for test query
            client = OpenAIEmbeddingClient()
            embedding = client.create_embedding(test_query)

            if not embedding:
                return {"error": "Failed to generate embedding"}

            logger.info(f"Testing search with query: '{test_query}'")
            logger.info(f"Generated embedding with {len(embedding)} dimensions")

            # Search with different thresholds
            results_07 = self.search_similar(
                embedding, similarity_threshold=0.7, limit=10
            )
            results_05 = self.search_similar(
                embedding, similarity_threshold=0.5, limit=10
            )
            results_03 = self.search_similar(
                embedding, similarity_threshold=0.3, limit=10
            )

            # Get total count from database
            with self.db.get_session() as session:
                total_count = session.execute(
                    text("SELECT COUNT(*) FROM conversation_embeddings")
                ).scalar()

            return {
                "test_query": test_query,
                "embedding_dimensions": len(embedding),
                "total_embeddings_in_db": total_count,
                "results": {
                    "threshold_0.7": len(results_07),
                    "threshold_0.5": len(results_05),
                    "threshold_0.3": len(results_03),
                },
                "sample_results": (
                    [r.to_dict() for r in results_03[:3]] if results_03 else []
                ),
            }

        except Exception as e:
            logger.error(f"Test search failed: {e}")
            return {"error": str(e)}

    def debug_database_contents(self) -> Dict[str, Any]:
        """Debug what's actually in the database."""
        try:
            with self.db.get_session() as session:
                # Count total embeddings
                total_count = session.execute(
                    text("SELECT COUNT(*) FROM conversation_embeddings")
                ).scalar()

                # Count by intent
                intent_counts = session.execute(
                    text(
                        """
                         SELECT intent, COUNT(*) as count
                         FROM conversation_embeddings
                         GROUP BY intent
                         ORDER BY count DESC
                         """
                    )
                ).fetchall()

                # Sample a few records
                sample_records = session.execute(
                    text(
                        """
                         SELECT conversation_id, intent, assistant_type
                         FROM conversation_embeddings
                         LIMIT 5
                         """
                    )
                ).fetchall()

                return {
                    "total_embeddings": total_count,
                    "intent_counts": {row.intent: row.count for row in intent_counts},
                    "sample_records": [
                        {
                            "conversation_id": row.conversation_id,
                            "intent": row.intent,
                            "assistant_type": row.assistant_type,
                        }
                        for row in sample_records
                    ],
                }

        except Exception as e:
            logger.error(f"Database debug failed: {e}")
            return {"error": str(e)}
