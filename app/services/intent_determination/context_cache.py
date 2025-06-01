"""
Async cache for conversation contexts.
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class AsyncContextCache:
    """
    Cache for asynchronously storing and retrieving context data with time-to-live (TTL) and
    maximum size constraints.

    This class provides a mechanism to store and manage context data associated with a
    conversation ID in an asynchronous environment. It ensures that cached data expires
    after a specified duration and automatically evicts the least recently used (LRU) data
    when the cache size exceeds the configured maximum.

    Attributes
    ----------
    ttl_minutes : int
        Time-to-live in minutes for each cached context. Cached data older than this duration
        is considered expired and will be removed upon retrieval attempts.
    max_size : int
        Maximum number of contexts that can be stored in the cache at any given time. Additional
        entries will trigger eviction of the least recently accessed data.
    _cache : Dict[str, Dict[str, Any]]
        Internal storage for cached contexts, indexed by conversation IDs.
    _access_times : Dict[str, datetime]
        Tracks the last access time for each cached context to facilitate TTL checks and LRU
        eviction.
    _lock : asyncio.Lock
        Asynchronous lock to ensure thread-safe access to internal cache structures.
    """

    def __init__(self, ttl_minutes: int = 30, max_size: int = 1000):
        self.ttl_minutes = ttl_minutes
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def get(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get context from cache.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Cached context or None
        """
        async with self._lock:
            if conversation_id not in self._cache:
                return None

            # Check if expired
            if self._is_expired(conversation_id):
                await self._remove(conversation_id)
                return None

            # Update access time
            self._access_times[conversation_id] = datetime.now()

            return self._cache[conversation_id].copy()

    async def set(self, conversation_id: str, context: Dict[str, Any]):
        """
        Store context in cache.

        Args:
            conversation_id: Conversation identifier
            context: Context to cache
        """
        async with self._lock:
            # Check cache size
            if len(self._cache) >= self.max_size:
                await self._evict_oldest()

            # Store context
            self._cache[conversation_id] = context.copy()
            self._access_times[conversation_id] = datetime.now()

            logger.debug(f"Cached context for conversation {conversation_id}")

    async def remove(self, conversation_id: str):
        """Remove context from cache."""
        async with self._lock:
            await self._remove(conversation_id)

    async def _remove(self, conversation_id: str):
        """Internal remove without lock."""
        if conversation_id in self._cache:
            del self._cache[conversation_id]
            del self._access_times[conversation_id]
            logger.debug(f"Removed cached context for {conversation_id}")

    def _is_expired(self, conversation_id: str) -> bool:
        """Check if the cached context is expired."""
        if conversation_id not in self._access_times:
            return True

        access_time = self._access_times[conversation_id]
        expiry_time = access_time + timedelta(minutes=self.ttl_minutes)

        return datetime.now() > expiry_time

    async def _evict_oldest(self):
        """Evict oldest cached context."""
        if not self._access_times:
            return

        oldest_id = min(self._access_times, key=self._access_times.get)
        await self._remove(oldest_id)
        logger.debug(f"Evicted oldest context: {oldest_id}")

    async def clear(self):
        """Clear all cached contexts."""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()
            logger.info("Cleared context cache")
