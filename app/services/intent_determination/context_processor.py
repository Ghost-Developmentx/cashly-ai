"""
Processes conversation context for intent determination.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ContextProcessor:
    """
    Processes and analyzes conversational history and context to extract useful information.

    The ContextProcessor is designed to analyze dialogue data in the form of conversation
    history combined with user-specific context. It aims to interpret relevant user queries,
    extract topics of discussion, identify performed actions, and summarize user-related data.
    Additionally, it prepares textual data optimized for embedding generation, which can be
    used in downstream tasks such as recommendation systems or response generation models.

    Attributes
    ----------
    topic_keywords : dict
        A dictionary where keys are high-level topics (e.g., 'transactions', 'accounts')
        and values are lists of keywords associated with each topic. Used for topic
        extraction in conversations.
    """

    def __init__(self):
        self.topic_keywords = {
            "transactions": ["transaction", "payment", "expense", "income"],
            "accounts": ["account", "bank", "balance", "connected"],
            "invoices": ["invoice", "bill", "client", "payment"],
            "forecasting": ["forecast", "predict", "future", "projection"],
            "budgets": ["budget", "limit", "spending", "allocation"],
            "insights": ["analyze", "trend", "pattern", "insight"],
        }

    def extract_key_information(
        self,
        conversation_history: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract key information from conversation and context.

        Args:
            conversation_history: Conversation messages
            user_context: User context data

        Returns:
            Dictionary with key information
        """
        # Extract from conversation
        user_queries = self._extract_user_queries(conversation_history)
        topics = self._extract_topics(conversation_history)
        actions = self._extract_actions(conversation_history)

        # Extract from user context
        context_info = self._extract_context_info(user_context)

        return {
            "user_queries": user_queries,
            "topics": topics,
            "actions_taken": actions,
            "user_context_summary": context_info,
            "message_count": len(conversation_history),
        }

    def prepare_embedding_text(
        self,
        conversation_history: List[Dict[str, Any]],
        user_context: Optional[Dict[str, Any]],
        current_query: str,
        max_length: int = 10000,
    ) -> str:
        """
        Prepare text for embedding generation.
        """
        parts = []

        # Add user context summary
        if user_context:
            context_summary = self._create_context_summary(user_context)
            if context_summary:
                parts.append(f"User Context: {context_summary}")

        # Add conversation flow
        if conversation_history:
            logger.info(
                f"Including {len(conversation_history)} messages in embedding context"
            )
            conversation_flow = self._create_conversation_flow(
                conversation_history[-10:]  # Last 10 messages
            )
            if conversation_flow:
                parts.append(f"Conversation:\n{conversation_flow}")

        # Add current query
        parts.append(f"Current Query: {current_query}")

        # Join and truncate
        text = "\n\n".join(parts)
        if len(text) > max_length:
            text = text[: max_length - 3] + "..."

        logger.debug(f"Prepared embedding text: {len(text)} characters")
        return text

    @staticmethod
    def _extract_user_queries(conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract all user queries from conversation."""
        queries = []
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content:
                    queries.append(content)
        return queries

    def _extract_topics(self, conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract main topics from conversation."""
        topics = set()

        for msg in conversation_history:
            content_lower = msg.get("content", "").lower()

            for topic, keywords in self.topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    topics.add(topic)

        return list(topics)

    @staticmethod
    def _extract_actions(
        conversation_history: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Extract actions taken in conversation."""
        actions = []

        for msg in conversation_history:
            if msg.get("role") == "assistant":
                content_lower = msg.get("content", "").lower()

                # Check for action indicators
                if "created" in content_lower:
                    actions.append({"type": "create", "inferred": True})
                elif "updated" in content_lower:
                    actions.append({"type": "update", "inferred": True})
                elif "deleted" in content_lower:
                    actions.append({"type": "delete", "inferred": True})
                elif "found" in content_lower or "showing" in content_lower:
                    actions.append({"type": "retrieve", "inferred": True})

        return actions

    @staticmethod
    def _extract_context_info(user_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract relevant information from user context."""
        if not user_context:
            return {}

        info = {}

        # Account information
        if "accounts" in user_context:
            accounts = user_context["accounts"]
            info["connected_accounts"] = len(accounts)
            info["total_balance"] = sum(acc.get("balance", 0) for acc in accounts)

        # Stripe status
        if "stripe_connect" in user_context:
            stripe = user_context["stripe_connect"]
            info["stripe_connected"] = stripe.get("connected", False)
            info["can_accept_payments"] = stripe.get("can_accept_payments", False)

        # Transaction count
        if "transactions" in user_context:
            info["transaction_count"] = len(user_context["transactions"])

        return info

    @staticmethod
    def _create_context_summary(user_context: Dict[str, Any]) -> str:
        """Create a summary of user context."""
        parts = []

        # Accounts
        accounts = user_context.get("accounts", [])
        if accounts:
            parts.append(f"{len(accounts)} connected accounts")

        # Stripe
        stripe = user_context.get("stripe_connect", {})
        if stripe.get("connected"):
            parts.append("Stripe connected")

        # Transactions
        transactions = user_context.get("transactions", [])
        if transactions:
            parts.append(f"{len(transactions)} transactions available")

        return ", ".join(parts) if parts else "New user"

    def _create_conversation_flow(self, messages: List[Dict[str, Any]]) -> str:
        """Create a conversation flow summary."""
        flow_parts = []
        total_length = 0
        max_flow_length = 2000  # Limit conversation flow size

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            # Skip if adding this would exceed limit
            if total_length + len(content) > max_flow_length:
                flow_parts.append("... (truncated earlier messages)")
                break

            if role == "user":
                # Truncate long messages
                if len(content) > 200:
                    content = content[:200] + "..."
                flow_parts.append(f"User: {content}")
                total_length += len(content) + 6  # "User: "
            elif role == "assistant":
                # Summarize assistant responses
                summary = self._summarize_assistant_response(content)
                flow_parts.append(f"Assistant: {summary}")
                total_length += len(summary) + 11  # "Assistant: "

        return "\n".join(flow_parts)

    @staticmethod
    def _summarize_assistant_response(content: str) -> str:
        """Create summary of assistant response."""
        content_lower = content.lower()

        if "created" in content_lower:
            return "Created resource"
        elif "updated" in content_lower:
            return "Updated resource"
        elif "deleted" in content_lower:
            return "Deleted resource"
        elif "found" in content_lower or "showing" in content_lower:
            return "Retrieved data"
        elif "forecast" in content_lower:
            return "Generated forecast"
        elif "budget" in content_lower:
            return "Provided budget information"
        else:
            # Return first sentence or truncated
            first_sentence = content.split(".")[0]
            if len(first_sentence) > 100:
                return first_sentence[:100] + "..."
            return first_sentence
