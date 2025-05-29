"""
Integration service for context-aware intent classification.
"""

import logging
from typing import Dict, Any, List, Optional

from services.intent_determination.intent_resolver import IntentResolver
from services.embeddings.storage import EmbeddingStorage
from services.intent_classification.intent_learner import IntentLearner

logger = logging.getLogger(__name__)


class ContextAwareIntentService:
    """Main service integrating context-aware intent classification."""

    def __init__(self):
        self.resolver = IntentResolver()
        self.learner = IntentLearner()
        self.storage = EmbeddingStorage()

    def classify_with_context(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        user_id: str,
        conversation_id: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Classify intent using full context awareness.

        Args:
            query: User query
            conversation_history: Previous messages
            user_id: User identifier
            conversation_id: Conversation identifier
            user_context: Additional context

        Returns:
            Classification result with routing recommendation
        """
        # Resolve intent using similarity search
        resolution = self.resolver.resolve_intent(
            query=query,
            conversation_history=conversation_history,
            user_id=user_id,
            conversation_id=conversation_id,
            user_context=user_context,
        )

        # Format for compatibility with an existing system
        return {
            "classification": {
                "intent": resolution["intent"],
                "confidence": resolution["confidence"],
                "method": resolution["method"],
                "assistant_used": resolution["recommended_assistant"],
            },
            "routing": {
                "strategy": self._determine_routing_strategy(resolution),
                "primary_assistant": resolution["recommended_assistant"],
                "confidence": resolution["routing_confidence"],
            },
            "analysis": resolution["analysis"],
        }

    @staticmethod
    def _determine_routing_strategy(resolution: Dict[str, Any]) -> str:
        """Determine a routing strategy based on confidence."""
        confidence = resolution["confidence"]

        if confidence >= 0.85:
            return "direct_route"
        elif confidence >= 0.70:
            return "route_with_fallback"
        elif confidence >= 0.50:
            return "general_with_context"
        else:
            return "general_fallback"
