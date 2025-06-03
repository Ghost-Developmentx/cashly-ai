"""
Handles routing logic and response formatting.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AsyncRoutingHandler:
    """
    Asynchronous routing handler for assistant classification.

    This class provides functionality to handle the routing of various
    assistant classifications based on the intent and confidence levels of
    the provided query resolution. It also includes a fallback mechanism
    for handling errors when classification fails. This allows for a unified
    handling of various assistants, ensuring proper routing based on user
    inputs and circumstances.

    Attributes
    ----------
    assistant_routing : Dict[str, str]
        A dictionary mapping intent names to their corresponding assistant
        identifiers. It defines the routing strategy for each type of
        classification intent.
    """

    def __init__(self):
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

    async def format_response(
        self, resolution: Dict[str, Any], query: str, method: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Format the resolution into the expected response structure.

        Args:
            resolution: Classification resolution
            query: Original query
            method: Classification method used

        Returns:
            Formatted response
        """
        intent = resolution["intent"]
        confidence = resolution["confidence"]

        return {
            "classification": {
                "intent": intent,
                "confidence": confidence,
                "method": method,
                "assistant_used": resolution.get(
                    "recommended_assistant",
                    self.assistant_routing.get(intent, "transaction_assistant"),
                ),
            },
            "routing": {
                "strategy": self._determine_routing_strategy(confidence),
                "primary_assistant": resolution.get("recommended_assistant"),
                "confidence": resolution.get("routing_confidence", confidence),
            },
            "should_route": confidence > 0.4 and intent != "general",
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

    @staticmethod
    def get_error_response(query: str) -> Dict[str, Any]:
        """Get an error response when classification fails."""
        return {
            "classification": {
                "intent": "transactions",
                "confidence": 0.4,
                "method": "error_fallback",
                "assistant_used": "transaction_assistant",
            },
            "routing": {
                "strategy": "general_fallback",
                "primary_assistant": "transaction_assistant",
                "confidence": 0.4,
            },
            "should_route": True,
            "recommended_assistant": "transaction_assistant",
            "analysis": {
                "reason": "Classification failed, using transaction assistant as fallback"
            },
        }
