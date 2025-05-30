"""
Extracts metadata from conversations for learning.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class LearningMetadataExtractor:
    """
    Class for extracting metadata from conversational interactions.

    This class is designed to analyze conversation histories and extract metadata
    such as topics discussed, function calls made, and characteristics of the
    interaction. The extracted metadata can be useful for understanding user
    intent and behavior, improving conversational AI systems, or generating
    analytics.

    Attributes
    ----------
    topic_keywords : dict
        A dictionary mapping conversation topics to their associated keywords.
        Used to identify topics discussed in a conversation.
    """

    def __init__(self):
        self.topic_keywords = {
            "transactions": ["transaction", "payment", "expense", "income"],
            "accounts": ["account", "bank", "balance", "connect"],
            "invoices": ["invoice", "bill", "client", "payment"],
            "forecasting": ["forecast", "predict", "future", "projection"],
            "budgets": ["budget", "limit", "spending", "allocation"],
            "insights": ["analyze", "trend", "pattern", "insight"],
        }

    def extract_metadata(
        self,
        conversation_history: List[Dict[str, Any]],
        final_intent: str,
        final_assistant: str,
    ) -> Dict[str, Any]:
        """
        Extract metadata from conversation.

        Args:
            conversation_history: List of message dictionaries
            final_intent: Classified intent
            final_assistant: Assistant that handled the query

        Returns:
            Metadata dictionary
        """
        metadata = {
            "intent": final_intent,
            "assistant_type": final_assistant,
            "message_count": len(conversation_history),
            "timestamp": datetime.now().isoformat(),
        }

        # Extract topics
        topics = self._extract_topics(conversation_history)
        if topics:
            metadata["topics"] = ",".join(topics)

        # Extract function calls
        function_calls = self._extract_function_calls(conversation_history)
        if function_calls:
            metadata["function_calls"] = ",".join(function_calls)

        # Extract conversation characteristics
        characteristics = self._extract_characteristics(conversation_history)
        if characteristics:
            metadata["characteristics"] = ",".join(characteristics)

        return metadata

    def _extract_topics(self, conversation_history: List[Dict]) -> List[str]:
        """Extract main topics from the conversation."""
        topics = set()

        # Check all messages
        for msg in conversation_history:
            content_lower = msg.get("content", "").lower()

            for topic, keywords in self.topic_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    topics.add(topic)

        return list(topics)

    @staticmethod
    def _extract_function_calls(conversation_history: List[Dict]) -> List[str]:
        """Extract function calls from conversation."""
        function_calls = []

        for msg in conversation_history:
            if msg.get("role") == "assistant":
                # Check metadata for tool usage
                if "tools_used" in msg.get("metadata", {}):
                    for tool in msg["metadata"]["tools_used"]:
                        function_calls.append(tool.get("name", "unknown"))

                # Infer from content
                content_lower = msg.get("content", "").lower()
                if "created" in content_lower:
                    function_calls.append("create_operation")
                elif "updated" in content_lower:
                    function_calls.append("update_operation")
                elif "deleted" in content_lower:
                    function_calls.append("delete_operation")

        return list(set(function_calls))

    @staticmethod
    def _extract_characteristics(conversation_history: List[Dict]) -> List[str]:
        """Extract conversation characteristics."""
        characteristics = []

        # Multi-turn conversation
        if len(conversation_history) > 2:
            characteristics.append("multi_turn")

        # Check for corrections
        for msg in conversation_history:
            content_lower = msg.get("content", "").lower()
            if any(
                word in content_lower for word in ["update", "fix", "correct", "change"]
            ):
                characteristics.append("includes_corrections")
                break

        # Check for questions
        user_messages = [m for m in conversation_history if m.get("role") == "user"]
        if any("?" in msg.get("content", "") for msg in user_messages):
            characteristics.append("contains_questions")

        return characteristics
