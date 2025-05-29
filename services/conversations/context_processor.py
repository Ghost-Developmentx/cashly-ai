"""
Processes conversation context for different use cases.
"""

import logging
from typing import List, Dict, Any

from models.conversation_data import ConversationContext, MessageRole

logger = logging.getLogger(__name__)


class ContextProcessor:
    """Processes conversation context for various purposes."""

    def __init__(self):
        self.summary_length = 500
        self.key_points_count = 5

    def prepare_for_embedding(self, context: ConversationContext) -> str:
        """
        Prepare context for embedding generation.

        Args:
            context: Conversation context

        Returns:
            String representation optimized for embedding
        """
        parts = []

        # Add user profile summary if available
        if context.user_context:
            profile_summary = self._create_user_profile_summary(context.user_context)
            if profile_summary:
                parts.append(f"User Profile: {profile_summary}")

        # Add conversation flow
        conversation_flow = self._create_conversation_flow(context.messages)
        parts.append(f"Conversation:\n{conversation_flow}")

        # Add extracted topics
        topics = self._extract_topics(context)
        if topics:
            parts.append(f"Topics: {', '.join(topics)}")

        return "\n\n".join(parts)

    def extract_key_information(self, context: ConversationContext) -> Dict[str, Any]:
        """
        Extract key information from the conversation.

        Args:
            context: Conversation context

        Returns:
            Dictionary with key information
        """
        return {
            "user_queries": self._extract_user_queries(context),
            "topics": self._extract_topics(context),
            "actions_taken": self._extract_actions(context),
            "sentiment": self._analyze_sentiment(context),
            "complexity": self._assess_complexity(context),
        }

    @staticmethod
    def _create_user_profile_summary(user_context: Dict[str, Any]) -> str:
        """Create summary of user profile."""
        parts = []

        # Account information
        accounts = user_context.get("accounts", [])
        if accounts:
            parts.append(f"{len(accounts)} connected accounts")

        # Financial status
        if "stripe_connect" in user_context:
            stripe = user_context["stripe_connect"]
            if stripe.get("connected"):
                parts.append("Stripe connected")

        # Activity level
        transactions = user_context.get("transactions", [])
        if transactions:
            parts.append(f"{len(transactions)} recent transactions")

        return ", ".join(parts) if parts else "New user"

    def _create_conversation_flow(self, messages: List[Any]) -> str:
        """Create a conversation flow summary."""
        flow_parts = []

        # Limit to recent messages
        recent_messages = messages[-10:] if len(messages) > 10 else messages

        for msg in recent_messages:
            if msg.role == MessageRole.USER:
                # Truncate long messages
                content = (
                    msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                )
                flow_parts.append(f"User: {content}")
            elif msg.role == MessageRole.ASSISTANT:
                # Summarize assistant responses
                summary = self._summarize_assistant_response(msg.content)
                flow_parts.append(f"Assistant: {summary}")

        return "\n".join(flow_parts)

    @staticmethod
    def _summarize_assistant_response(content: str) -> str:
        """Create summary of assistant response."""
        # Extract key actions or information
        if "created" in content.lower():
            return "Created resource"
        elif "found" in content.lower() or "showing" in content.lower():
            return "Retrieved data"
        elif "updated" in content.lower():
            return "Updated resource"
        elif "deleted" in content.lower():
            return "Deleted resource"
        elif "forecast" in content.lower():
            return "Generated forecast"
        elif "budget" in content.lower():
            return "Provided budget information"
        else:
            # Return first sentence or truncated content
            return content.split(".")[0][:100]

    @staticmethod
    def _extract_user_queries(context: ConversationContext) -> List[str]:
        """Extract all user queries."""
        return [msg.content for msg in context.user_messages]

    @staticmethod
    def _extract_topics(context: ConversationContext) -> List[str]:
        """Extract main topics from the conversation."""
        topics = set()

        # Define topic keywords
        topic_keywords = {
            "transactions": ["transaction", "payment", "expense", "income"],
            "accounts": ["account", "bank", "balance", "connect"],
            "invoices": ["invoice", "bill", "client", "payment"],
            "forecasting": ["forecast", "predict", "future", "projection"],
            "budgets": ["budget", "limit", "spending", "allocation"],
            "insights": ["analyze", "trend", "pattern", "insight"],
        }

        # Check all messages
        for msg in context.messages:
            content_lower = msg.content.lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    topics.add(topic)

        return list(topics)

    @staticmethod
    def _extract_actions(context: ConversationContext) -> List[Dict[str, Any]]:
        """Extract actions taken in conversation."""
        actions = []

        for msg in context.assistant_messages:
            # Check metadata for tool usage
            if "tools_used" in msg.metadata:
                for tool in msg.metadata["tools_used"]:
                    actions.append(
                        {
                            "type": "tool_call",
                            "tool": tool.get("name", "unknown"),
                            "success": tool.get("success", False),
                        }
                    )

            # Infer from content
            content_lower = msg.content.lower()
            if "created" in content_lower:
                actions.append({"type": "create", "inferred": True})
            elif "updated" in content_lower:
                actions.append({"type": "update", "inferred": True})
            elif "deleted" in content_lower:
                actions.append({"type": "delete", "inferred": True})

        return actions

    @staticmethod
    def _analyze_sentiment(context: ConversationContext) -> str:
        """Analyze conversation sentiment."""
        # Simple sentiment analysis based on keywords
        positive_words = ["thank", "great", "perfect", "excellent", "helpful"]
        negative_words = ["problem", "issue", "error", "wrong", "fail"]

        positive_count = 0
        negative_count = 0

        for msg in context.user_messages:
            content_lower = msg.content.lower()
            positive_count += sum(1 for word in positive_words if word in content_lower)
            negative_count += sum(1 for word in negative_words if word in content_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    @staticmethod
    def _assess_complexity(context: ConversationContext) -> str:
        """Assess conversation complexity."""
        # Based on message count and topics
        if context.message_count < 4:
            return "simple"
        elif context.message_count < 10:
            return "moderate"
        else:
            return "complex"
