"""
Main service for conversation context aggregation.
"""

import logging
from typing import Dict, List, Any, Optional

from .aggregator import ConversationAggregator
from .context_processor import ContextProcessor
from .context_cache import ContextCache
from models.conversation_data import ConversationContext

logger = logging.getLogger(__name__)


class AggregatorService:
    """Main service coordinating context aggregation."""

    def __init__(self):
        self.aggregator = ConversationAggregator()
        self.processor = ContextProcessor()
        self.cache = ContextCache()

    def process_conversation(
        self,
        conversation_id: str,
        user_id: str,
        conversation_history: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Process conversation and prepare for intent classification.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            conversation_history: Raw conversation history
            user_context: User context data
            use_cache: Whether to use caching

        Returns:
            Processed conversation data
        """
        # Check cache first
        if use_cache:
            cached_context = self.cache.get(conversation_id)
            if cached_context:
                logger.info(f"Using cached context for {conversation_id}")
                return self._prepare_response(cached_context)

        # Aggregate context
        context = self.aggregator.aggregate_context(
            conversation_history=conversation_history,
            user_id=user_id,
            conversation_id=conversation_id,
            user_context=user_context,
        )

        # Cache if enabled
        if use_cache:
            self.cache.set(conversation_id, context)

        return self._prepare_response(context)

    def _prepare_response(self, context: ConversationContext) -> Dict[str, Any]:
        """Prepare a response with processed context."""
        # Extract key information
        key_info = self.processor.extract_key_information(context)

        # Prepare embedding text
        embedding_text = self.processor.prepare_for_embedding(context)

        return {
            "conversation_id": context.conversation_id,
            "user_id": context.user_id,
            "message_count": context.message_count,
            "current_intent": context.current_intent,
            "current_assistant": context.current_assistant,
            "embedding_text": embedding_text,
            "key_information": key_info,
            "user_context": context.user_context,
        }

    def clear_cache(self):
        """Clear the context cache."""
        self.cache.clear()
        logger.info("Context cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return self.cache.get_stats()
