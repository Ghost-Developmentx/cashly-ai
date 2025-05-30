"""
Async intent classification service.
Main entry point for async intent classification.
"""

import logging
from typing import Dict, List, Optional, Any

from services.intent_determination.intent_resolver import AsyncIntentResolver
from services.intent_classification.intent_learner import IntentLearner
from services.intent_classification.fallback_classifier import FallbackClassifier
from services.embeddings.async_embeddings import AsyncEmbeddingStorage

logger = logging.getLogger(__name__)


class AsyncIntentService:
    """Async intent classification using embeddings."""

    def __init__(self):
        self.resolver = AsyncIntentResolver()
        self.learner = IntentLearner()
        self.fallback = FallbackClassifier()
        self.storage = AsyncEmbeddingStorage()
        self.min_confidence_threshold = 0.55

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
            "general": "transaction_assistant",
        }

    def classify_and_route(
        self,
        query: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Classify intent and determine routing strategy with enhanced fallback.
        """
        try:
            logger.info(f"🚀 classify_and_route called with:")
            logger.info(f"   Query: '{query}'")
            logger.info(f"   User context: {user_context is not None}")
            logger.info(f"   Conversation history: {conversation_history is not None}")

            # First attempt: Use vector-based resolution
            vector_result = self._try_vector_classification(
                query, user_context, conversation_history
            )

            # Check if vector classification is confident enough
            if (
                vector_result
                and vector_result["confidence"] >= self.min_confidence_threshold
            ):
                logger.info(
                    f"✅ Vector classification successful: {vector_result['intent']} ({vector_result['confidence']:.1%})"
                )
                return self._format_response(
                    vector_result, query, method="vector_search"
                )

            # Fallback: Use enhanced keyword-based classification
            logger.info(
                "🔄 Vector classification insufficient, using enhanced fallback"
            )
            fallback_result = self._use_enhanced_fallback(query, user_context)

            return self._format_response(
                fallback_result, query, method="enhanced_fallback"
            )

        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Ultimate fallback
            return self._use_simple_fallback(query)

    def _try_vector_classification(
        self,
        query: str,
        user_context: Optional[Dict],
        conversation_history: Optional[List[Dict]],
    ) -> Optional[Dict[str, Any]]:
        """Attempt vector-based classification with adjusted confidence threshold."""
        try:
            logger.info(f"🔍 Attempting vector classification for: '{query}'")

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

            logger.info(
                f"📋 Context: user_id={user_id}, conversation_id={conversation_id}"
            )
            logger.info(
                f"📋 Conversation history: {len(conversation_history) if conversation_history else 0} messages"
            )

            # Use context-aware resolution
            resolution = self.resolver.resolve_intent(
                query=query,
                conversation_history=conversation_history or [],
                user_id=user_id,
                conversation_id=conversation_id,
                user_context=user_context,
            )

            # Log the resolution details
            logger.info(f"📊 Vector resolution result:")
            logger.info(f"   Intent: {resolution['intent']}")
            logger.info(f"   Confidence: {resolution['confidence']:.3f}")
            logger.info(f"   Method: {resolution.get('method', 'unknown')}")

            # Check if we have analysis data
            if "analysis" in resolution:
                logger.info(f"   Analysis: {resolution['analysis']}")

            # ADJUSTED: Use a lower threshold and boost confidence from similarity
            raw_confidence = resolution["confidence"]

            # If we have good similarity results, boost confidence
            if "analysis" in resolution and "avg_similarity" in resolution.get(
                "analysis", {}
            ):
                avg_similarity = resolution["analysis"]["avg_similarity"]
                logger.info(f"   Avg similarity: {avg_similarity:.3f}")

                if avg_similarity >= 0.65:  # Good similarity
                    boosted_confidence = min(raw_confidence * 1.2, 0.95)  # Boost by 20%
                    logger.info(
                        f"   Boosted confidence from {raw_confidence:.3f} to {boosted_confidence:.3f} "
                        f"due to similarity {avg_similarity:.3f}"
                    )
                    resolution["confidence"] = boosted_confidence

            # Only return if we have reasonable confidence
            if resolution["confidence"] >= self.min_confidence_threshold:
                logger.info(
                    f"✅ Vector classification successful with confidence {resolution['confidence']:.3f}"
                )
                return resolution

            logger.info(
                f"❌ Vector classification confidence too low: {resolution['confidence']:.3f} < "
                f"{self.min_confidence_threshold:.3f}"
            )
            return None

        except Exception as e:
            logger.error(f"❌ Vector classification error: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _format_response(
        self, resolution: Dict[str, Any], query: str, method: str = "unknown"
    ) -> Dict[str, Any]:
        """Format the resolution into the expected response structure."""
        intent = resolution["intent"]
        confidence = resolution["confidence"]

        return {
            "classification": {
                "intent": intent,
                "confidence": confidence,
                "method": method,
                "assistant_used": resolution["recommended_assistant"],
            },
            "routing": {
                "strategy": self._determine_routing_strategy(confidence),
                "primary_assistant": resolution["recommended_assistant"],
                "confidence": resolution.get("routing_confidence", confidence),
            },
            "should_route": confidence > 0.4 and intent != "general",  # Lower threshold
            "recommended_assistant": self.assistant_routing.get(
                intent, "transaction_assistant"
            ),
            "analysis": resolution.get("analysis", {}),
        }

    @staticmethod
    def _determine_routing_strategy(confidence: float) -> str:
        """Determine routing strategy based on confidence."""
        if confidence >= 0.8:
            return "direct_route"
        elif confidence >= 0.6:
            return "route_with_fallback"
        elif confidence >= 0.4:
            return "general_with_context"
        else:
            return "general_fallback"

    def _use_enhanced_fallback(
        self, query: str, user_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Use enhanced fallback classification with context awareness."""
        # Get fallback classification
        intent, confidence = self.fallback.classify(query)

        # Apply context-based adjustments
        adjusted_confidence = self._apply_context_adjustments(
            intent, confidence, user_context, query
        )

        return {
            "intent": intent,
            "confidence": adjusted_confidence,
            "recommended_assistant": self.assistant_routing.get(
                intent, "transaction_assistant"
            ),
            "routing_confidence": adjusted_confidence,
            "method": "enhanced_fallback",
            "analysis": {
                "fallback_suggestions": self.fallback.get_intent_suggestions(query),
                "context_applied": user_context is not None,
                "reason": "Vector database empty or low confidence, using enhanced keyword matching",
            },
        }

    @staticmethod
    def _apply_context_adjustments(
        intent: str, confidence: float, user_context: Optional[Dict], query: str
    ) -> float:
        """Apply context-based confidence adjustments."""
        adjusted_confidence = confidence

        if not user_context:
            return adjusted_confidence

        # Boost confidence based on user context
        if intent == "accounts" and user_context.get("accounts"):
            adjusted_confidence = min(adjusted_confidence + 0.1, 0.9)

        if intent == "invoices" and user_context.get("stripe_connect", {}).get(
            "connected"
        ):
            adjusted_confidence = min(adjusted_confidence + 0.1, 0.9)

        if intent == "transactions" and user_context.get("transactions"):
            adjusted_confidence = min(adjusted_confidence + 0.1, 0.9)

        # Query-specific adjustments
        query_lower = query.lower()
        if "show" in query_lower or "list" in query_lower:
            adjusted_confidence = min(adjusted_confidence + 0.1, 0.9)

        return adjusted_confidence

    @staticmethod
    def _use_simple_fallback(query: str) -> Dict[str, Any]:
        """Ultimate simple fallback when everything else fails."""
        return {
            "classification": {
                "intent": "transactions",
                "confidence": 0.4,
                "method": "simple_fallback",
                "assistant_used": "transaction_assistant",
            },
            "routing": {
                "strategy": "general_fallback",
                "primary_assistant": "transaction_assistant",
                "confidence": 0.4,
            },
            "should_route": True,  # Always route to transaction assistant
            "recommended_assistant": "transaction_assistant",
            "analysis": {
                "reason": "All classification methods failed, using transaction assistant as fallback"
            },
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

    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification performance statistics."""
        return {
            "min_confidence_threshold": self.min_confidence_threshold,
            "available_intents": list(self.assistant_routing.keys()),
            "fallback_active": True,
            "vector_db_status": "checking...",  # Could add actual check here
        }
