"""
Data models for conversation context.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Message role enumeration."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """Represents a single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from a dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=(
                datetime.fromisoformat(data["timestamp"])
                if data.get("timestamp")
                else None
            ),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationContext:
    """Complete conversation context with metadata."""

    conversation_id: str
    user_id: str
    messages: List[Message]
    current_intent: Optional[str] = None
    current_assistant: Optional[str] = None
    user_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        """Get total message count."""
        return len(self.messages)

    @property
    def user_messages(self) -> List[Message]:
        """Get only user messages."""
        return [m for m in self.messages if m.role == MessageRole.USER]

    @property
    def assistant_messages(self) -> List[Message]:
        """Get only assistant messages."""
        return [m for m in self.messages if m.role == MessageRole.ASSISTANT]

    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:] if self.messages else []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "current_intent": self.current_intent,
            "current_assistant": self.current_assistant,
            "user_context": self.user_context,
            "message_count": self.message_count,
        }
