"""
Maps user intents to appropriate assistant types.
"""

import logging
from typing import Dict, Optional, List
from ..assistant_manager import AssistantType
from ..utils.constants import (
    INTENT_TO_ASSISTANT_MAPPING,
    ASSISTANT_KEYWORDS,
    INVOICE_CONTEXT_KEYWORDS,
    INVOICE_HISTORY_PHRASES,
    DEFAULT_RECENT_MESSAGES_COUNT,
    RoutingStrategy,
)

logger = logging.getLogger(__name__)


class IntentMapper:
    """Handles mapping of intents to appropriate assistant types."""

    def __init__(self):
        self.intent_mapping = self._convert_intent_mapping()
        self.assistant_keywords = ASSISTANT_KEYWORDS
        self.invoice_keywords = INVOICE_CONTEXT_KEYWORDS
        self.invoice_phrases = INVOICE_HISTORY_PHRASES

    def _convert_intent_mapping(self) -> Dict[str, AssistantType]:
        """Convert string mappings to AssistantType enums."""
        return {
            intent: AssistantType(assistant_str)
            for intent, assistant_str in INTENT_TO_ASSISTANT_MAPPING.items()
        }

    def get_assistant_for_intent(
        self,
        intent: str,
        routing_result: Dict,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> AssistantType:
        """
        Determine which assistant to use based on intent and query content.

        Args:
            intent: Classified intent from intent service
            routing_result: Full routing result with strategy
            query: User's query text
            conversation_history: Previous conversation messages

        Returns:
            AssistantType to handle the query
        """
        # Check for invoice context first
        if self.check_invoice_context(query, conversation_history):
            logger.info("📧 Detected invoice context - routing to Invoice Assistant")
            return AssistantType.INVOICE

        query_lower = query.lower()

        # Special handling for accounts intent
        if intent == "accounts":
            assistant_type = self._check_account_intent_keywords(query_lower)
            if assistant_type:
                return assistant_type
            return AssistantType.ACCOUNT

        # Handle routing strategies
        strategy = routing_result["routing"]["strategy"]

        if strategy in [
            RoutingStrategy.DIRECT_ROUTE.value,
            RoutingStrategy.ROUTE_WITH_FALLBACK.value,
        ]:
            return self.intent_mapping.get(intent, AssistantType.TRANSACTION)

        # Default fallback for general queries
        if strategy in [
            RoutingStrategy.GENERAL_WITH_CONTEXT.value,
            RoutingStrategy.GENERAL_FALLBACK.value,
        ]:
            return AssistantType.TRANSACTION

        return AssistantType.TRANSACTION

    def check_invoice_context(
        self, query: str, conversation_history: Optional[List[Dict]] = None
    ) -> bool:
        """
        Check if the query is related to a recent invoice action.

        Args:
            query: User's query text
            conversation_history: Previous conversation messages

        Returns:
            True if invoice context is detected
        """
        query_lower = query.lower()

        # Check for invoice-related keywords
        has_invoice_keyword = any(
            keyword in query_lower for keyword in self.invoice_keywords
        )

        if not has_invoice_keyword:
            return False

        # Check recent conversation history
        if conversation_history and len(conversation_history) > 1:
            recent_messages = conversation_history[-DEFAULT_RECENT_MESSAGES_COUNT:]

            for msg in recent_messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "").lower()
                    if any(phrase in content for phrase in self.invoice_phrases):
                        return True

        return False

    def _check_account_intent_keywords(
        self, query_lower: str
    ) -> Optional[AssistantType]:
        """
        Check for special keywords in account-related queries.

        Args:
            query_lower: Lowercase query text

        Returns:
            AssistantType if special keywords found, None otherwise
        """
        # Check bank connection keywords
        for keyword in self.assistant_keywords.get("bank_connection", []):
            if keyword in query_lower:
                logger.info(
                    f"🔗 Routing to Bank Connection Assistant due to keyword: '{keyword}'"
                )
                return AssistantType.BANK_CONNECTION

        # Check payment processing keywords
        for keyword in self.assistant_keywords.get("payment_processing", []):
            if keyword in query_lower:
                logger.info(
                    f"💳 Routing to Payment Processing Assistant due to keyword: '{keyword}'"
                )
                return AssistantType.PAYMENT_PROCESSING

        return None

    def get_default_assistant(self, intent: str) -> AssistantType:
        """
        Get the default assistant for a given intent.

        Args:
            intent: Intent classification

        Returns:
            Default AssistantType for the intent
        """
        return self.intent_mapping.get(intent, AssistantType.TRANSACTION)
