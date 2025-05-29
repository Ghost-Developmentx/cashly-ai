"""
Aggregates and processes conversation data from Rails.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from models.conversation_data import ConversationContext, Message, MessageRole

logger = logging.getLogger(__name__)


class ConversationAggregator:
    """Aggregates conversation data and extracts relevant context."""

    def __init__(self):
        self.max_context_length = 10000  # Max characters for context
        self.max_messages = 20  # Max messages to consider

    def aggregate_context(
        self,
        conversation_history: List[Dict[str, Any]],
        user_id: str,
        conversation_id: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> ConversationContext:
        """
        Aggregate conversation data into structured context.

        Args:
            conversation_history: Raw conversation history from Rails
            user_id: User identifier
            conversation_id: Conversation identifier
            user_context: Additional user context

        Returns:
            Structured conversation context
        """
        # Convert raw messages to Message objects
        messages = self._parse_messages(conversation_history)

        # Extract current state
        current_intent = self._extract_current_intent(messages)
        current_assistant = self._extract_current_assistant(messages)

        # Create context
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=messages,
            current_intent=current_intent,
            current_assistant=current_assistant,
            user_context=user_context or {},
        )

        logger.info(
            f"Aggregated context for conversation {conversation_id}: "
            f"{context.message_count} messages, intent={current_intent}"
        )

        return context

    def _parse_messages(self, conversation_history: List[Dict]) -> List[Message]:
        """Parse raw messages into Message objects."""
        messages = []

        for raw_msg in conversation_history:
            try:
                message = Message(
                    role=MessageRole(raw_msg.get("role", "user")),
                    content=raw_msg.get("content", ""),
                    timestamp=self._parse_timestamp(raw_msg.get("created_at")),
                    metadata=self._extract_message_metadata(raw_msg),
                )
                messages.append(message)
            except Exception as e:
                logger.warning(f"Failed to parse message: {e}")
                continue

        return messages

    @staticmethod
    def _parse_timestamp(timestamp_str: Optional[str]) -> Optional[datetime]:
        """Parse timestamp string."""
        if not timestamp_str:
            return None

        try:
            # Handle ISO format
            return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        except Exception:
            try:
                # Handle Rails default format
                return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
            except Exception:
                return None

    @staticmethod
    def _extract_message_metadata(raw_msg: Dict) -> Dict[str, Any]:
        """Extract metadata from raw message."""
        metadata = {}

        # Extract tool usage if present
        if "tools_used" in raw_msg:
            metadata["tools_used"] = raw_msg["tools_used"]

        # Extract feedback if present
        if "feedback_rating" in raw_msg:
            metadata["feedback"] = {
                "rating": raw_msg["feedback_rating"],
                "helpful": raw_msg.get("was_helpful", False),
            }

        return metadata

    @staticmethod
    def _extract_current_intent(messages: List[Message]) -> Optional[str]:
        """Extract current intent from the conversation."""
        # Look for intent in recent assistant messages
        for message in reversed(messages):
            if message.role == MessageRole.ASSISTANT:
                # Check metadata first
                if "intent" in message.metadata:
                    return message.metadata["intent"]

                # Try to extract from content patterns
                content_lower = message.content.lower()
                if "transaction" in content_lower:
                    return "transactions"
                elif "invoice" in content_lower:
                    return "invoices"
                elif "forecast" in content_lower:
                    return "forecasting"
                elif "budget" in content_lower:
                    return "budgets"

        return None

    def _extract_current_assistant(self, messages: List[Message]) -> Optional[str]:
        """Extract current assistant from conversation."""
        # This would ideally come from metadata
        # For now, infer from intent
        intent_to_assistant = {
            "transactions": "transaction_assistant",
            "invoices": "invoice_assistant",
            "forecasting": "forecasting_assistant",
            "budgets": "budget_assistant",
        }

        current_intent = self._extract_current_intent(messages)
        return intent_to_assistant.get(current_intent, "general_assistant")
