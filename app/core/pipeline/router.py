"""
Assistant router - selects the best assistant for a query.
Single responsibility: routing decisions only.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from ...schemas.assistant import AssistantType
from ...schemas.classification import Intent, ClassificationResult

logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    """Result of the routing decision."""
    assistant: AssistantType
    confidence: float
    reason: str
    should_reroute: bool = False
    alternative_assistant: Optional[AssistantType] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "assistant": self.assistant.value,
            "confidence": self.confidence,
            "reason": self.reason,
            "should_reroute": self.should_reroute,
            "alternative_assistant": self.alternative_assistant.value if self.alternative_assistant else None
        }

class AssistantRouter:
    """
    Routes queries to appropriate assistants based on classification.
    Handles cross-functional queries and rerouting logic.
    """

    def __init__(self):
        """Initialize router with routing rules."""
        self.routing_rules = self._build_routing_rules()
        self.confidence_threshold = 0.7

    @staticmethod
    def _build_routing_rules() -> Dict[str, Any]:
        """Build routing rules for special cases."""
        return {
            "cross_functional": {
                # Transaction + Account queries
                ("transaction", "balance"): AssistantType.TRANSACTION,
                ("spending", "account"): AssistantType.TRANSACTION,

                # Budget + Forecast queries
                ("budget", "forecast"): AssistantType.BUDGET,
                ("save", "predict"): AssistantType.BUDGET,

                # Invoice + Payment queries
                ("invoice", "stripe"): AssistantType.INVOICE,
                ("payment", "client"): AssistantType.PAYMENT_PROCESSING,
            },
            "confidence_boost": {
                # Boost confidence for certain combinations
                AssistantType.TRANSACTION: ["spent", "bought", "paid"],
                AssistantType.ACCOUNT: ["balance", "total", "funds"],
                AssistantType.FORECASTING: ["predict", "future", "will"],
            },
            "reroute_triggers": {
                # Keywords that might trigger rerouting
                "account": ["balance", "connect", "link"],
                "transaction": ["add", "create", "delete", "update"],
                "invoice": ["bill", "client", "send"],
            }
        }

    async def route(
            self,
            classification: ClassificationResult,
            query: str,
            user_context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Determine which assistant should handle the query.

        Args:
            classification: Classification result
            query: Original user query
            user_context: Optional user context

        Returns:
            RoutingDecision with selected assistant
        """
        # Start with classification suggestion
        selected_assistant = classification.suggested_assistant
        confidence = classification.confidence
        reason = f"Based on intent: {classification.intent.value}"

        # Check for cross-functional queries
        cross_functional_decision = self._check_cross_functional(query, classification)
        if cross_functional_decision:
            selected_assistant = cross_functional_decision[0]
            confidence = cross_functional_decision[1]
            reason = "Cross-functional query detected"

        # Apply confidence boosting rules
        boosted_confidence = self._apply_confidence_boost(
            selected_assistant,
            classification.keywords_found,
            confidence
        )
        if boosted_confidence > confidence:
            confidence = boosted_confidence
            reason += " (confidence boosted by keywords)"

        # Check if confidence is too low
        if confidence < self.confidence_threshold:
            # Consider rerouting or using default
            alternative = self._find_alternative_assistant(classification, user_context)
            if alternative:
                return RoutingDecision(
                    assistant=selected_assistant,
                    confidence=confidence,
                    reason=reason,
                    should_reroute=True,
                    alternative_assistant=alternative
                )

        # Create routing decision
        decision = RoutingDecision(
            assistant=selected_assistant,
            confidence=confidence,
            reason=reason
        )

        logger.info(f"Routed to {selected_assistant.value} (confidence: {confidence:.2f})")

        return decision

    def _check_cross_functional(
            self,
            query: str,
            classification: ClassificationResult
    ) -> Optional[Tuple[AssistantType, float]]:
        """Check if query matches cross-functional patterns."""
        query_lower = query.lower()

        for (keyword1, keyword2), assistant in self.routing_rules["cross_functional"].items():
            if keyword1 in query_lower and keyword2 in query_lower:
                # Cross-functional match found
                confidence = min(classification.confidence + 0.15, 0.95)
                return assistant, confidence

        return None

    def _apply_confidence_boost(
            self,
            assistant: AssistantType,
            keywords_found: List[str],
            current_confidence: float
    ) -> float:
        """Apply confidence boost based on keywords."""
        if assistant not in self.routing_rules["confidence_boost"]:
            return current_confidence

        boost_keywords = self.routing_rules["confidence_boost"][assistant]
        matches = sum(1 for keyword in keywords_found if keyword in boost_keywords)

        if matches > 0:
            # Boost confidence by 0.05 per matching keyword, max 0.15
            boost = min(matches * 0.05, 0.15)
            return min(current_confidence + boost, 0.95)

        return current_confidence

    @staticmethod
    def _find_alternative_assistant(
            classification: ClassificationResult,
            user_context: Optional[Dict[str, Any]]
    ) -> Optional[AssistantType]:
        """Find alternative assistant for low-confidence queries."""
        # Check if user has been using a specific assistant recently
        if user_context and "recent_assistant" in user_context:
            return AssistantType(user_context["recent_assistant"])

        # Default alternatives based on intent
        alternatives = {
            Intent.GENERAL: AssistantType.INSIGHTS,
            Intent.TRANSACTION_QUERY: AssistantType.ACCOUNT,
            Intent.ACCOUNT_BALANCE: AssistantType.TRANSACTION,
        }

        return alternatives.get(classification.intent)

    @staticmethod
    def should_reroute_response(
            assistant_response: str,
            current_assistant: AssistantType,
            original_query: str
    ) -> Optional[AssistantType]:
        """
        Check if response suggests rerouting to another assistant.

        Args:
            assistant_response: Response from current assistant
            current_assistant: Current assistant type
            original_query: Original user query

        Returns:
            Alternative assistant if rerouting needed, None otherwise
        """
        response_lower = assistant_response.lower()

        # Check for explicit rerouting phrases
        reroute_phrases = {
            "account assistant": AssistantType.ACCOUNT,
            "transaction assistant": AssistantType.TRANSACTION,
            "invoice assistant": AssistantType.INVOICE,
            "budget assistant": AssistantType.BUDGET,
            "forecast assistant": AssistantType.FORECASTING,
        }

        for phrase, target_assistant in reroute_phrases.items():
            if phrase in response_lower and target_assistant != current_assistant:
                logger.info(f"Rerouting suggested from {current_assistant.value} to {target_assistant.value}")
                return target_assistant

        # Check for implicit rerouting based on response content
        if current_assistant == AssistantType.TRANSACTION:
            if any(word in response_lower for word in ["balance", "account total", "funds available"]):
                return AssistantType.ACCOUNT
        elif current_assistant == AssistantType.ACCOUNT:
            if any(word in response_lower for word in ["transaction", "purchase", "spent"]):
                return AssistantType.TRANSACTION

        return None
