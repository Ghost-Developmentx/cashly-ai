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
    """Model for storing conversation embeddings."""

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
