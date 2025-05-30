"""
SQLAlchemy model for conversation embeddings.
"""

from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy import Column, String, Integer, Boolean, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class ConversationEmbedding(Base):
    """
    Represents a database model for storing conversation embeddings.

    The ConversationEmbedding class is used to store metadata and embeddings
    related to a specific conversation. It provides attributes for storing
    information about the user, conversation, assistant type, intent, and
    various other metrics to facilitate analysis and processing of stored
    conversational data.

    Attributes
    ----------
    id : int
        The unique identifier for the conversation embedding record.
    conversation_id : str
        Identifier of the conversation associated with the embedding.
    user_id : str
        Identifier of the user associated with the conversation.
    intent : str
        Detected intent of the conversation.
    assistant_type : str
        Type of assistant that generated the embedding.
    embedding : Vector
        High-dimensional vector representing the conversation's embedding.
    message_count : int
        The number of messages associated with the conversation. Defaults to 1.
    success_indicator : bool
        Whether the conversation was successful. Defaults to True.
    created_at : datetime
        Timestamp indicating when the conversation embedding was created.
    conversation_metadata : JSON
        Additional metadata related to the conversation.
    """

    __tablename__ = "conversation_embeddings"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(255), nullable=False)
    user_id = Column(String(255), nullable=False)
    intent = Column(String(100), nullable=False)
    assistant_type = Column(String(100), nullable=False)
    embedding = Column(Vector(1536), nullable=False)
    message_count = Column(Integer, default=1)
    success_indicator = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)
    conversation_metadata = Column(JSON, default={})  # Changed from 'metadata'

    # Indexes for performance
    __table_args__ = (
        Index("idx_user_conversations", "user_id", "conversation_id"),
        Index("idx_intent", "intent"),
        Index("idx_assistant_type", "assistant_type"),
        Index("idx_created_at", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "intent": self.intent,
            "assistant_type": self.assistant_type,
            "message_count": self.message_count,
            "success_indicator": self.success_indicator,
            "created_at": self.created_at.isoformat(),
            "conversation_metadata": self.conversation_metadata,
        }
