"""
Async context aggregation for intent resolution.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

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

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            conversation_history: Raw conversation history
            user_context: User context data
            query: Current query

        Returns:
            Processed conversation data
        """
        # Check cache first
        cached_context = await self.cache.get(conversation_id)
        if cached_context:
            logger.info(f"Using cached context for {conversation_id}")
            return cached_context

        # Process the context
        context_data = self._aggregate_context(
            conversation_history=conversation_history,
            user_id=user_id,
            conversation_id=conversation_id,
            user_context=user_context,
            query=query,
        )

        # Cache for future use
        await self.cache.set(conversation_id, context_data)

        return context_data

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
            "timestamp": datetime.utcnow().isoformat(),
        }
