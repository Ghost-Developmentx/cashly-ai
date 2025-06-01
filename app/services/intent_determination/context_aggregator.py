"""
Async context aggregation for intent resolution.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

from .context_processor import ContextProcessor
from .context_cache import AsyncContextCache

logger = logging.getLogger(__name__)


class AsyncContextAggregator:
    """
    Manages and processes asynchronous conversation context aggregation.

    This class is responsible for managing user and conversation contexts,
    aggregating historical data, and preparing it for further processing.
    It integrates a context processor and an asynchronous cache to optimize
    repeated requests by utilizing cached results. Designed to efficiently
    handle user queries while maintaining system performance.

    Attributes
    ----------
    processor : ContextProcessor
        Instance of ContextProcessor used for processing conversational data.
    cache : AsyncContextCache
        Asynchronous cache instance used to store and retrieve conversation contexts.
    max_context_length : int
        Maximum length of the context that can be processed.
    max_messages : int
        Maximum number of messages allowed in a single conversation context.
    """

    def __init__(self):
        self.processor = ContextProcessor()
        self.cache = AsyncContextCache()
        self.max_context_length = 10000
        self.max_messages = 20

    async def process_conversation_async(
        self,
        conversation_id: str,
        user_id: str,
        conversation_history: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
        query: str = "",
    ) -> Dict[str, Any]:
        """
        Process conversation context asynchronously.
        """
        # Create a unique cache key that includes the query and message count
        cache_key = self._create_cache_key(
            conversation_id, len(conversation_history), query
        )

        # Check cache with the unique key
        cached_context = await self.cache.get(cache_key)
        if cached_context:
            logger.info(f"Using cached context for {cache_key}")
            return cached_context

        # Log what we're processing
        logger.info(
            f"Processing fresh context for conversation {conversation_id} "
            f"with {len(conversation_history)} messages"
        )

        # Process the context
        context_data = self._aggregate_context(
            conversation_history=conversation_history,
            user_id=user_id,
            conversation_id=conversation_id,
            user_context=user_context,
            query=query,
        )

        # Cache for future use with the unique key
        await self.cache.set(cache_key, context_data)

        return context_data

    @staticmethod
    def _create_cache_key(conversation_id: str, message_count: int, query: str) -> str:
        """Create a unique cache key based on the conversation state."""
        key_data = f"{conversation_id}:{message_count}:{query}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _aggregate_context(
        self,
        conversation_history: List[Dict[str, Any]],
        user_id: str,
        conversation_id: str,
        user_context: Optional[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        """Aggregate context from various sources."""
        # Extract key information
        key_info = self.processor.extract_key_information(
            conversation_history, user_context
        )

        # Prepare embedding text
        embedding_text = self.processor.prepare_embedding_text(
            conversation_history=conversation_history,
            user_context=user_context,
            current_query=query,
            max_length=self.max_context_length,
        )

        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "message_count": len(conversation_history),
            "embedding_text": embedding_text,
            "key_information": key_info,
            "user_context": user_context or {},
            "timestamp": datetime.now().isoformat(),
        }
