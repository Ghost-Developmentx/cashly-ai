"""
Builds context from conversation history for embedding.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ConversationContextBuilder:
    """
    Handles building context and extracting metadata for conversation systems.

    The `ConversationContextBuilder` provides utilities to process conversation history,
    generate context strings for embedding, and extract meaningful metadata like intent,
    topics, and conversational characteristics. It is intended for usage in chatbot or
    assistant systems.

    Attributes
    ----------
    max_messages : int
        Maximum number of messages to include in the built context.
    max_tokens : int
        Maximum token limit for the context string.
    """

    def __init__(self, max_messages: int = 10, max_tokens: int = 4000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens

    def build_context(
        self, conversation_history: List[Dict[str, Any]], current_query: str
    ) -> str:
        """
        Build context string from conversation history.

        Args:
            conversation_history: List of message dictionaries
            current_query: Current user query

        Returns:
            Context string for embedding
        """
        if not conversation_history:
            return f"Current query: {current_query}"

        # Take recent messages
        recent_messages = conversation_history[-self.max_messages :]

        # Build context parts
        context_parts = []

        # Add conversation flow
        for msg in recent_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if role == "user":
                context_parts.append(f"User: {content}")
            elif role == "assistant":
                # Summarize assistant responses
                summary = self._summarize_assistant_response(content)
                context_parts.append(f"Assistant: {summary}")

        # Add current query
        context_parts.append(f"Current query: {current_query}")

        # Join and truncate if needed
        context = "\n".join(context_parts)
        return self._truncate_context(context)

    def extract_metadata(
        self,
        conversation_history: List[Dict[str, Any]],
        intent: str,
        assistant_type: str,
    ) -> Dict[str, Any]:
        """
        Extract metadata from conversation.

        Args:
            conversation_history: List of message dictionaries
            intent: Classified intent
            assistant_type: Assistant that handled the query

        Returns:
            Metadata dictionary
        """
        metadata = {
            "intent": intent,
            "assistant_type": assistant_type,
            "message_count": len(conversation_history),
            "timestamp": datetime.now().isoformat(),
        }

        # Extract key topics
        topics = self._extract_topics(conversation_history)
        if topics:
            metadata["topics"] = topics

        # Check for function calls
        function_calls = self._extract_function_calls(conversation_history)
        if function_calls:
            metadata["function_calls"] = function_calls

        return metadata

    @staticmethod
    def _summarize_assistant_response(content: str) -> str:
        """Summarize long assistant responses."""
        if len(content) < 200:
            return content

        # Take first and last parts
        return f"{content[:100]}... {content[-50:]}"

    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit token limits."""
        # Simple character-based truncation
        # In production, use tiktoken for accurate token counting
        if len(context) > self.max_tokens * 4:  # Rough estimate
            return context[: self.max_tokens * 4]
        return context

    @staticmethod
    def _extract_topics(conversation_history: List[Dict[str, Any]]) -> List[str]:
        """Extract key topics from conversation."""
        topics = set()

        # Simple keyword extraction
        keywords = {
            "transactions": ["transaction", "payment", "expense", "income"],
            "accounts": ["account", "bank", "balance"],
            "invoices": ["invoice", "bill", "client"],
            "forecasting": ["forecast", "predict", "future"],
            "budgets": ["budget", "limit", "spending"],
        }

        for msg in conversation_history:
            content = msg.get("content", "").lower()
            for topic, words in keywords.items():
                if any(word in content for word in words):
                    topics.add(topic)

        return list(topics)

    @staticmethod
    def _extract_function_calls(
        conversation_history: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract function calls from conversation."""
        function_calls = []

        # Look for patterns indicating function usage
        for msg in conversation_history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Simple pattern matching for function mentions
                if "created" in content or "updated" in content:
                    function_calls.append("data_modification")
                if "found" in content or "showing" in content:
                    function_calls.append("data_retrieval")

        return list(set(function_calls))

    @staticmethod
    def _extract_conversation_characteristics(
        conversation_history: List[Dict],
    ) -> List[str]:
        """Extract characteristics of the conversation."""
        characteristics = []

        # Check for multi-turn
        if len(conversation_history) > 2:
            characteristics.append("multi_turn")

        # Check for tool sequences
        all_tools = []
        for msg in conversation_history:
            if msg.get("role") == "assistant" and "tools_used" in msg:
                all_tools.extend([t["name"] for t in msg["tools_used"]])

        if len(all_tools) > 1:
            characteristics.append("multiple_tools")

        # Check for corrections or updates
        for msg in conversation_history:
            content_lower = msg.get("content", "").lower()
            if any(
                word in content_lower
                for word in ["update", "fix", "correct", "change", "actually"]
            ):
                characteristics.append("includes_corrections")
                break

        return characteristics
