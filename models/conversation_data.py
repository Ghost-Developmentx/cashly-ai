"""
Data models for conversation context.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """
    Enumeration of roles in a messaging system.

    The `MessageRole` class represents different roles that could exist in a
    messaging context. It inherits from `str` and `Enum` to enable comparison
    and string-based usage. This can be used in systems where messages are
    categorized or processed based on the role of the entity sending or
    associated with the message.

    Attributes
    ----------
    USER : MessageRole
        Represents a user in the messaging system.
    ASSISTANT : MessageRole
        Represents an assistant, such as an AI or chatbot, in the messaging
        system.
    SYSTEM : MessageRole
        Represents the system itself, often used for internal or system-level
        messages.
    """

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Message:
    """
    Represents a message object with role, content, timestamp, and metadata.

    This class is used to encapsulate data related to a message, such as its role,
    content, an optional timestamp, and associated metadata. It provides functionality
    to convert the message object to and from a dictionary representation.

    Attributes
    ----------
    role : MessageRole
        The role associated with the message, defining its purpose or source.
    content : str
        The textual content of the message.
    timestamp : Optional[datetime]
        The optional timestamp indicating when the message was created or sent.
    metadata : Dict[str, Any]
        Additional metadata or information related to the message.
    """

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
    """
    Represents the context of a conversation.

    This class stores information about an ongoing conversation, including the
    history of messages, the user's context, and any active intent or assistant.
    It acts as a state container for managing conversations and retrieving specific
    information such as user or assistant messages.

    Attributes
    ----------
    conversation_id : str
        Unique identifier for the conversation.
    user_id : str
        Unique identifier for the user participating in the conversation.
    messages : List[Message]
        List of messages exchanged during the conversation.
    current_intent : Optional[str]
        Current recognized user intent, if available.
    current_assistant : Optional[str]
        Current assistant assigned to handle the conversation, if applicable.
    user_context : Dict[str, Any]
        Contextual information related to the user, such as preferences or
        metadata.
    """

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
