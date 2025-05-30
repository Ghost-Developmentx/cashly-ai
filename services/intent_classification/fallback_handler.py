"""
Handles fallback classification when vector search fails.
"""

import logging
from typing import Dict, Optional, Any
from .fallback_classifier import FallbackClassifier

logger = logging.getLogger(__name__)


class AsyncFallbackHandler:
    """Handles fallback classification logic."""

    def __init__(self):
        self.fallback_classifier = FallbackClassifier()
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

    async def classify_with_fallback(
        self, query: str, user_context: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Use enhanced fallback classification.

        Args:
            query: User's query
            user_context: Optional user context

        Returns:
            Classification result
        """
        # Get fallback classification (sync call is okay here)
        intent, confidence = self.fallback_classifier.classify(query)

        # Apply context adjustments
        adjusted_confidence = self._apply_context_adjustments(
            intent, confidence, user_context, query
        )

        # Get suggestions for analysis
        suggestions = self.fallback_classifier.get_intent_suggestions(query)

        return {
            "intent": intent,
            "confidence": adjusted_confidence,
            "recommended_assistant": self.assistant_routing.get(
                intent, "transaction_assistant"
            ),
            "routing_confidence": adjusted_confidence,
            "method": "enhanced_fallback",
            "analysis": {
                "fallback_suggestions": suggestions,
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

        # Boost based on user context
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
