"""
Async intent classification service.
Main entry point for async intent classification.
"""

import logging
from typing import Dict, List, Optional, Any

from .classification_handler import AsyncClassificationHandler
from .routing_handler import AsyncRoutingHandler
from .fallback_handler import AsyncFallbackHandler
from ..intent_determination.intent_resolver import AsyncIntentResolver
from ..embeddings.async_embeddings import AsyncEmbeddingStorage

logger = logging.getLogger(__name__)


class AsyncIntentService:
    """
    AsyncIntentService provides an asynchronous interface for intent classification and
    routing strategies. It integrates functionalities such as classification using vectors,
    enhanced fallback classification, and learning from completed conversations. Additionally,
    it provides insights into classification performance statistics.

    Detailed description of the class, its purpose, and usage.

    Attributes
    ----------
    resolver : AsyncIntentResolver
        Handles intent resolution asynchronously.
    storage : AsyncEmbeddingStorage
        Storage for embeddings used in classification.
    classification_handler : AsyncClassificationHandler
        Handles classification using vector-based methods.
    routing_handler : AsyncRoutingHandler
        Manages routing strategies based on classification results.
    fallback_handler : AsyncFallbackHandler
        Handles fallback classification when other methods are insufficient.
    min_confidence_threshold : float
        Minimum confidence threshold to determine if classification results are valid.
    """

    def __init__(self):
        self.resolver = AsyncIntentResolver()
        self.storage = AsyncEmbeddingStorage()

        # Initialize handlers
        self.classification_handler = AsyncClassificationHandler(self.resolver)
        self.routing_handler = AsyncRoutingHandler()
        self.fallback_handler = AsyncFallbackHandler()

        self.min_confidence_threshold = 0.55

    async def classify_and_route(
        self,
        query: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Classify intent and determine routing strategy.

        Args:
            query: User's query text
            user_context: Optional user context
            conversation_history: Optional conversation history

        Returns:
            Classification and routing result
        """
        try:
            logger.info(f"🚀 Async classify_and_route called for: '{query}'")

            # Try vector classification first
            vector_result = await self.classification_handler.classify_with_vectors(
                query, user_context, conversation_history
            )

            # Check if confident enough
            if (
                vector_result
                and vector_result["confidence"] >= self.min_confidence_threshold
            ):
                logger.info(
                    f"✅ Vector classification successful: {vector_result['intent']} "
                    f"({vector_result['confidence']:.1%})"
                )
                return await self.routing_handler.format_response(
                    vector_result, query, method="vector_search"
                )

            # Fallback to enhanced keyword classification
            logger.info("🔄 Using enhanced fallback classification")
            fallback_result = await self.fallback_handler.classify_with_fallback(
                query, user_context
            )

            return await self.routing_handler.format_response(
                fallback_result, query, method="enhanced_fallback"
            )

        except Exception as e:
            logger.error(f"Intent classification failed: {e}", exc_info=True)
            return self.routing_handler.get_error_response(query)

    async def learn_from_conversation(
        self,
        conversation_id: str,
        user_id: str,
        conversation_history: List[Dict],
        final_intent: str,
        final_assistant: str,
        success_indicator: bool,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Learn from completed conversation asynchronously.

        Args:
            conversation_id: Unique conversation ID
            user_id: User identifier
            conversation_history: Full conversation
            final_intent: Final classified intent
            final_assistant: Assistant that handled it
            success_indicator: Was it successful?
            metadata: Additional metadata

        Returns:
            True if learning was successful
        """
        try:
            # Delegate to learner (implement AsyncIntentLearner separately)
            from .async_intent_learner import AsyncIntentLearner

            learner = AsyncIntentLearner(self.storage)

            return await learner.learn_from_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                conversation_history=conversation_history,
                final_intent=final_intent,
                final_assistant=final_assistant,
                success_indicator=success_indicator,
                metadata=metadata,
            )
        except Exception as e:
            logger.error(f"Failed to learn from conversation: {e}")
            return False

    async def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification performance statistics."""
        return {
            "min_confidence_threshold": self.min_confidence_threshold,
            "available_intents": list(self.routing_handler.assistant_routing.keys()),
            "fallback_active": True,
            "vector_db_status": "active",
        }
