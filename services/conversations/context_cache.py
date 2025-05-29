"""
Caches conversation contexts for performance.
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

from models.conversation_data import ConversationContext

logger = logging.getLogger(__name__)


class ContextCache:
    """In-memory cache for conversation contexts."""

    def __init__(self, ttl_minutes: int = 30, max_size: int = 1000):
        self.ttl_minutes = ttl_minutes
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}

    def get(self, conversation_id: str) -> Optional[ConversationContext]:
        """
        Get context from the cache.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Cached context or None
        """
        if conversation_id not in self._cache:
            return None

        # Check if expired
        if self._is_expired(conversation_id):
            self.remove(conversation_id)
            return None

        # Update access time
        self._access_times[conversation_id] = datetime.now()

        # Deserialize and return
        cached_data = self._cache[conversation_id]
        return self._deserialize_context(cached_data)

    def set(self, conversation_id: str, context: ConversationContext):
        """
        Store context in cache.

        Args:
            conversation_id: Conversation identifier
            context: Context to cache
        """
        # Check cache size
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        # Serialize and store
        self._cache[conversation_id] = self._serialize_context(context)
        self._access_times[conversation_id] = datetime.now()

        logger.debug(f"Cached context for conversation {conversation_id}")

    def remove(self, conversation_id: str):
        """Remove context from the cache."""
        if conversation_id in self._cache:
            del self._cache[conversation_id]
            del self._access_times[conversation_id]
            logger.debug(f"Removed cached context for {conversation_id}")

    def clear(self):
        """Clear all cached contexts."""
        self._cache.clear()
        self._access_times.clear()
        logger.info("Cleared context cache")

    def _is_expired(self, conversation_id: str) -> bool:
        """Check if the cached context is expired."""
        if conversation_id not in self._access_times:
            return True

        access_time = self._access_times[conversation_id]
        expiry_time = access_time + timedelta(minutes=self.ttl_minutes)

        return datetime.now() > expiry_time

    def _evict_oldest(self):
        """Evict oldest cached context."""
        if not self._access_times:
            return

        oldest_id = min(self._access_times, key=self._access_times.get)
        self.remove(oldest_id)
        logger.debug(f"Evicted oldest context: {oldest_id}")

    @staticmethod
    def _serialize_context(context: ConversationContext) -> Dict[str, Any]:
        """Serialize context for storage."""
        return context.to_dict()

    @staticmethod
    def _deserialize_context(data: Dict[str, Any]) -> ConversationContext:
        """Deserialize context from storage."""
        from models.conversation_data import Message

        # Reconstruct messages
        messages = [Message.from_dict(msg) for msg in data["messages"]]

        return ConversationContext(
            conversation_id=data["conversation_id"],
            user_id=data["user_id"],
            messages=messages,
            current_intent=data.get("current_intent"),
            current_assistant=data.get("current_assistant"),
            user_context=data.get("user_context", {}),
        )

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_minutes": self.ttl_minutes,
        }
