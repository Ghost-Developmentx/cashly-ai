"""
Handles routing and re-routing logic for assistants.
"""

import logging
from typing import Dict, Optional, List
from ..assistant_manager import AssistantType, AssistantResponse
from ..utils.constants import (
    CROSS_ROUTING_PATTERNS,
    ROUTING_TRIGGER_PHRASES,
    ASSISTANT_MENTION_KEYWORDS,
    INTENT_TO_ASSISTANT_MAPPING,
    ResponseThresholds,
)

logger = logging.getLogger(__name__)


class AssistantRouter:
    """
    AssistantRouter is responsible for managing routing logic between
    different assistants based on predefined patterns, triggers, and other
    criteria.

    This class provides methods for determining if re-routing should occur,
    identifying the correct target assistant for handling a user query,
    and validating routing paths between assistants. Its purpose is to
    facilitate seamless collaboration between different assistants and
    efficiently manage query redirection when needed.

    Attributes
    ----------
    routing_patterns : Optional[Dict]
        A dictionary defining routing patterns between assistants. Patterns
        specify source assistants, target assistants, and their respective
        triggers for re-routing.
    trigger_phrases : list
        List of phrases or keywords that may trigger re-routing behavior.
    assistant_mentions : Dict[str, AssistantType]
        A mapping of specific assistant mentions to their respective
        AssistantType enums. Used to determine the referenced assistant
        within a query or response.
    """

    def __init__(self, routing_patterns: Optional[Dict] = None):
        self.routing_patterns = self._convert_routing_patterns(
            routing_patterns or CROSS_ROUTING_PATTERNS
        )
        self.trigger_phrases = ROUTING_TRIGGER_PHRASES
        self.assistant_mentions = self._convert_assistant_mentions()

    def _convert_routing_patterns(
        self, patterns: Dict
    ) -> Dict[AssistantType, Dict[str, AssistantType]]:
        """Convert string patterns to AssistantType enums."""
        converted = {}
        for source_str, targets in patterns.items():
            source_type = AssistantType(source_str)
            converted[source_type] = {
                pattern: AssistantType(target_str)
                for pattern, target_str in targets.items()
            }
        return converted

    def _convert_assistant_mentions(self) -> Dict[str, AssistantType]:
        """Convert string mentions to AssistantType enums."""
        return {
            mention: AssistantType(assistant_str)
            for mention, assistant_str in ASSISTANT_MENTION_KEYWORDS.items()
        }

    def should_reroute(
        self, response: AssistantResponse, current_assistant: AssistantType, query: str
    ) -> bool:
        """
        Determine if we should re-route to a different assistant.

        Args:
            response: Response from current assistant
            current_assistant: Currently active assistant
            query: Original user query

        Returns:
            True if re-routing should be attempted
        """
        # If we got function calls and a good response, DON'T re-route
        if (
            len(response.function_calls)
            >= ResponseThresholds.MIN_FUNCTION_CALLS_FOR_NO_REROUTE
            and response.success
        ):
            logger.info(
                "✅ Assistant executed functions successfully - no re-routing needed"
            )
            return False

        # If the response is substantial, DON'T re-route
        if (
            len(response.content) > ResponseThresholds.MIN_CONTENT_LENGTH_FOR_NO_REROUTE
            and response.success
        ):
            logger.info(
                "✅ Assistant provided substantial response - no re-routing needed"
            )
            return False

        # Only re-route if the assistant explicitly mentions other assistants
        response_lower = response.content.lower()

        for phrase in self.trigger_phrases:
            if phrase in response_lower:
                logger.info(f"🔄 Re-routing triggered by phrase: '{phrase}'")
                return True

        logger.info("✅ No re-routing needed - assistant should handle this query")
        return False

    def determine_correct_assistant(
        self,
        response: AssistantResponse,
        current_assistant: AssistantType,
        query: str,
        intent_service=None,
    ) -> Optional[AssistantType]:
        """
        Determine which assistant should handle the query.

        Args:
            response: Response from current assistant
            current_assistant: Currently active assistant
            query: Original user query
            intent_service: Optional intent service for re-classification

        Returns:
            AssistantType that should handle the query, or None
        """
        response_lower = response.content.lower()

        # Check response content for assistant mentions
        for mention, assistant_type in self.assistant_mentions.items():
            if f"{mention} assistant" in response_lower:
                return assistant_type

        # Check routing patterns
        if current_assistant in self.routing_patterns:
            patterns = self.routing_patterns[current_assistant]
            query_lower = query.lower()

            for pattern_key, target_assistant in patterns.items():
                if pattern_key in query_lower:
                    return target_assistant

        # Fallback: re-classify the query if intent service provided
        if intent_service:
            try:
                routing_result = intent_service.classify_and_route(query)
                new_intent = routing_result["classification"]["intent"]
                assistant_str = INTENT_TO_ASSISTANT_MAPPING.get(
                    new_intent, "transaction"
                )
                return AssistantType(assistant_str)
            except Exception as e:
                logger.error(f"Error re-classifying query: {e}")

        return None

    def get_routing_path(
        self, from_assistant: AssistantType, to_assistant: AssistantType
    ) -> Optional[List[AssistantType]]:
        """
        Get the routing path between two assistants.

        Args:
            from_assistant: Starting assistant
            to_assistant: Target assistant

        Returns:
            List of assistants in the routing path, or None if no path exists
        """
        # Direct route
        if from_assistant == to_assistant:
            return [from_assistant]

        # Check if direct route exists
        if from_assistant in self.routing_patterns:
            patterns = self.routing_patterns[from_assistant]
            for pattern, target in patterns.items():
                if target == to_assistant:
                    return [from_assistant, to_assistant]

        # TODO: Implement multi-hop routing if needed
        return None

    def can_route_between(
        self, from_assistant: AssistantType, to_assistant: AssistantType
    ) -> bool:
        """
        Check if routing is possible between two assistants.

        Args:
            from_assistant: Starting assistant
            to_assistant: Target assistant

        Returns:
            True if routing is possible
        """
        return self.get_routing_path(from_assistant, to_assistant) is not None
