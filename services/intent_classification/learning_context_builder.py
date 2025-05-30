"""
Builds context from conversation history for learning.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class LearningContextBuilder:
    """
    Provides tools to build conversation contexts for embedding from history and user queries.

    This class is designed to create context strings by processing a conversation history and
    a current query. It supports constraints like limiting the number of messages considered
    and truncating the resulting context string to a specified maximum length. The resulting
    context can then be used for embedding or further processing.

    Attributes
    ----------
    max_messages : int
        The maximum number of messages to take from the conversation history for context
        construction.
    max_context_length : int
        The maximum allowed length of the generated context string.
    """

    def __init__(self, max_messages: int = 10, max_context_length: int = 4000):
        self.max_messages = max_messages
        self.max_context_length = max_context_length

    async def build_context_async(
        self, conversation_history: List[Dict[str, Any]], current_query: str
    ) -> str:
        """
        Build context string from conversation history asynchronously.

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

    @staticmethod
    def _summarize_assistant_response(content: str) -> str:
        """Summarize long assistant responses."""
        if len(content) < 200:
            return content

        # Extract key information
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

    def _truncate_context(self, context: str) -> str:
        """Truncate context to fit within limits."""
        if len(context) <= self.max_context_length:
            return context

        # Truncate from the beginning to keep most recent context
        return "..." + context[-(self.max_context_length - 3) :]
