"""
Context-aware intent classification service.
This is the main entry point for all intent classification.
"""

import logging
from typing import Dict, List, Optional, Any

from services.intent_determination.intent_resolver import IntentResolver
from services.intent_classification.intent_learner import IntentLearner
from services.intent_classification.fallback_classifier import FallbackClassifier

logger = logging.getLogger(__name__)


class IntentService:
    """
    Main service for intent classification using context-aware embeddings.
    """

    def __init__(self):
        self.resolver = IntentResolver()
        self.learner = IntentLearner()
        self.fallback = FallbackClassifier()

        # Assistant routing mapping
        self.assistant_routing = {
            "transactions": "transaction_assistant",
            "accounts": "account_assistant",
            "invoices": "invoice_assistant",
            "bank_connection": "bank_connection_assistant",
            "payment_processing": "payment_processing_assistant",
            "forecasting": "forecasting_assistant",
            "budgets": "budget_assistant",
            "insights": "insights_assistant",
            "general": "general_assistant",
        }

    def classify_and_route(
        self,
        query: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Classify intent and determine routing strategy.

        Args:
            query: User's query text
            user_context: User profile and context
            conversation_history: Previous messages

        Returns:
            Classification and routing result
        """
        try:
            # Prepare context
            user_id = (
                user_context.get("user_id", "anonymous")
                if user_context
                else "anonymous"
            )
            conversation_id = (
                user_context.get("conversation_id", f"temp_{user_id}")
                if user_context
                else f"temp_{user_id}"
            )

            # Use context-aware resolution
            resolution = self.resolver.resolve_intent(
                query=query,
                conversation_history=conversation_history or [],
                user_id=user_id,
                conversation_id=conversation_id,
                user_context=user_context,
            )

            # Format response
            return self._format_response(resolution, query)

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Use fallback
            return self._use_fallback(query)

    def _format_response(
        self, resolution: Dict[str, Any], query: str
    ) -> Dict[str, Any]:
        """Format the resolution into the expected response structure."""
        intent = resolution["intent"]
        confidence = resolution["confidence"]

        return {
            "classification": {
                "intent": intent,
                "confidence": confidence,
                "method": resolution["method"],
                "assistant_used": resolution["recommended_assistant"],
            },
            "routing": {
                "strategy": self._determine_routing_strategy(confidence),
                "primary_assistant": resolution["recommended_assistant"],
                "confidence": resolution["routing_confidence"],
            },
            "should_route": confidence > 0.6 and intent != "general",
            "recommended_assistant": self.assistant_routing.get(
                intent, "general_assistant"
            ),
            "analysis": resolution.get("analysis", {}),
        }

    @staticmethod
    def _determine_routing_strategy(confidence: float) -> str:
        """Determine routing strategy based on confidence."""
        if confidence >= 0.85:
            return "direct_route"
        elif confidence >= 0.70:
            return "route_with_fallback"
        elif confidence >= 0.50:
            return "general_with_context"
        else:
            return "general_fallback"

    def _use_fallback(self, query: str) -> Dict[str, Any]:
        """Use fallback classification when embedding fails."""
        intent, confidence = self.fallback.classify(query)

        return {
            "classification": {
                "intent": intent,
                "confidence": confidence,
                "method": "fallback",
                "assistant_used": self.assistant_routing.get(
                    intent, "general_assistant"
                ),
            },
            "routing": {
                "strategy": "general_fallback",
                "primary_assistant": self.assistant_routing.get(
                    intent, "general_assistant"
                ),
                "confidence": confidence,
            },
            "should_route": False,
            "recommended_assistant": self.assistant_routing.get(
                intent, "general_assistant"
            ),
            "analysis": {"reason": "Embedding classification failed, using fallback"},
        }

    def learn_from_conversation(
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
        Learn from the completed conversation.

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
        return self.learner.learn_from_conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            conversation_history=conversation_history,
            final_intent=final_intent,
            final_assistant=final_assistant,
            success_indicator=success_indicator,
            metadata=metadata,
        )
