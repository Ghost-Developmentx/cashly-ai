"""
Service for storing and retrieving embeddings from PostgreSQL.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta

from sqlalchemy import text

from models.conversation_embedding import ConversationEmbedding
from db.init import get_db_connection

logger = logging.getLogger(__name__)


class EmbeddingStorage:
    """Handles storage and retrieval of conversation embeddings."""

    def __init__(self):
        self.db = get_db_connection()

    def store_embedding(
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
        Store a conversation embedding.

        Args:
            conversation_id: Unique conversation identifier
            user_id: User identifier
            embedding: Embedding vector
            intent: Classified intent
            assistant_type: Assistant that handled the query
            metadata: Additional metadata
            success_indicator: Whether the interaction was successful

        Returns:
            ID of stored embedding or None if failed
        """
        try:
            with self.db.get_session() as session:
                embedding_record = ConversationEmbedding(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    embedding=embedding,
                    intent=intent,
                    assistant_type=assistant_type,
                    conversation_metadata=metadata or {},
                    success_indicator=success_indicator,
                    message_count=metadata.get("message_count", 1) if metadata else 1,
                )

                session.add(embedding_record)
                session.flush()

                embedding_id = embedding_record.id
                logger.info(
                    f"Stored embedding {embedding_id} for conversation {conversation_id}"
                )

                return embedding_id

        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")
            return None

    def find_similar_conversations(
        self,
        embedding: List[float],
        user_id: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """
        Find similar conversations using vector similarity.

        Args:
            embedding: Query embedding vector
            user_id: Optional user filter
            limit: Maximum results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar conversations with metadata
        """
        try:
            with self.db.get_session() as session:
                # Build a query with cosine similarity
                query = """
                        SELECT
                            id,
                            conversation_id,
                            intent,
                            assistant_type,
                            conversation_metadata,
                            success_indicator,
                            1 - (embedding <=> %s::vector) as similarity
                        FROM conversation_embeddings
                        WHERE 1 - (embedding <=> %s::vector) >= %s \
                        """

                params = [embedding, embedding, similarity_threshold]

                # Add user filter if provided
                if user_id:
                    query += " AND user_id = %s"
                    params.append(user_id)

                # Order by similarity and limit
                query += " ORDER BY similarity DESC LIMIT %s"
                params.append(limit)

                result = session.execute(text(query), params)

                similar_conversations = []
                for row in result:
                    similar_conversations.append(
                        {
                            "id": row[0],
                            "conversation_id": row[1],
                            "intent": row[2],
                            "assistant_type": row[3],
                            "metadata": row[4],
                            "success_indicator": row[5],
                            "similarity": float(row[6]),
                        }
                    )

                logger.info(f"Found {len(similar_conversations)} similar conversations")
                return similar_conversations

        except Exception as e:
            logger.error(f"Failed to find similar conversations: {e}")
            return []

    def get_user_conversation_patterns(
        self, user_id: str, days: int = 30
    ) -> Dict[str, Any]:
        """
        Get conversation patterns for a user.

        Args:
            user_id: User identifier
            days: Number of days to look back

        Returns:
            Dictionary with pattern analysis
        """
        try:
            with self.db.get_session() as session:
                cutoff_date = datetime.now() - timedelta(days=days)

                # Get intent distribution
                intent_query = """
                               SELECT intent, COUNT(*) as count
                               FROM conversation_embeddings
                               WHERE user_id = %s AND created_at >= %s
                               GROUP BY intent
                               ORDER BY count DESC \
                               """

                intent_result = session.execute(
                    text(intent_query), [user_id, cutoff_date]
                )

                intent_distribution = {row[0]: row[1] for row in intent_result}

                # Get assistant usage
                assistant_query = """
                                  SELECT assistant_type, COUNT(*) as count
                                  FROM conversation_embeddings
                                  WHERE user_id = %s AND created_at >= %s
                                  GROUP BY assistant_type
                                  ORDER BY count DESC \
                                  """

                assistant_result = session.execute(
                    text(assistant_query), [user_id, cutoff_date]
                )

                assistant_distribution = {row[0]: row[1] for row in assistant_result}

                return {
                    "intent_distribution": intent_distribution,
                    "assistant_distribution": assistant_distribution,
                    "total_conversations": sum(intent_distribution.values()),
                    "time_period_days": days,
                }

        except Exception as e:
            logger.error(f"Failed to get user patterns: {e}")
            return {}
