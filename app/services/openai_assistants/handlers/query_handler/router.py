"""
Routing and rerouting logic for queries.
Handles decisions about which assistant should handle a query.
"""

import logging
from typing import Optional, Tuple
from .types import QueryContext, RoutingDecision
from ...assistant_manager import AssistantType, AssistantResponse

logger = logging.getLogger(__name__)

class RoutingHandler:
    """
    Handles routing decisions and attempts reroutes for assistant responses.

    The `RoutingHandler` class serves to evaluate the necessity of rerouting
    user queries between available assistants based on various criteria and
    performs rerouting operations when deemed appropriate. It integrates
    closely with a routing system, an assistant manager, and an intent
    service to make decisions and process rerouted queries. This enables
    dynamic and context-aware redirection of user queries to the most
    suitable assistant.

    Attributes
    ----------
    router : Router
        Manages routing rules and logic for determining if rerouting is needed.
    assistant_manager : AssistantManager
        Responsible for managing queries and interactions with assistants.
    intent_service : IntentService
        Provides intent analysis and related functionality for query context.
    """

    def __init__(self, router, assistant_manager, intent_service):
        self.router = router
        self.assistant_manager = assistant_manager
        self.intent_service = intent_service

    def should_attempt_reroute(
            self,
            response: AssistantResponse,
            current_assistant: AssistantType,
            context: QueryContext
    ) -> RoutingDecision:
        """
        Determine if rerouting should be attempted.

        Args:
            response: Assistant response
            current_assistant: Current assistant type
            context: Query context

        Returns:
            Routing decision
        """
        # Use router to check
        should_reroute = self.router.should_reroute(
            response,
            current_assistant,
            context.query
        )

        if not should_reroute:
            return RoutingDecision(
                should_reroute=False,
                target_assistant=None,
                confidence=1.0,
                reason="Response is satisfactory"
            )

        # Determine a target assistant
        target = self.router.determine_correct_assistant(
            response,
            current_assistant,
            context.query,
            self.intent_service
        )

        if target and target != current_assistant:
            return RoutingDecision(
                should_reroute=True,
                target_assistant=target,
                confidence=0.8,
                reason="Better assistant available"
            )

        return RoutingDecision(
            should_reroute=False,
            target_assistant=None,
            confidence=0.6,
            reason="No better alternative found"
        )

    async def attempt_reroute(
            self,
            response: AssistantResponse,
            decision: RoutingDecision,
            context: QueryContext
    ) -> Optional[Tuple[AssistantResponse, AssistantType]]:
        """
        Attempt to reroute to a better assistant.

        Args:
            response: Original response
            decision: Routing decision
            context: Query context

        Returns:
            New response and assistant if successful
        """
        if not decision.should_reroute or not decision.target_assistant:
            return None

        logger.info(
            f"🔄 Attempting reroute to {decision.target_assistant.value}"
        )

        try:
            # Process with a new assistant
            new_response = await self.assistant_manager.process_query(
                query=context.query,
                assistant_type=decision.target_assistant,
                user_id=context.user_id,
                user_context=context.user_context,
                conversation_history=context.conversation_history
            )

            # Check if the rerouted response is better
            if self._is_better_response(new_response, response):
                logger.info("✅ Rerouting successful")
                return new_response, decision.target_assistant
            else:
                logger.info("⚠️ Rerouted response not better, keeping original")
                return None

        except Exception as e:
            logger.error(f"Rerouting failed: {e}")
            return None

    @staticmethod
    def _is_better_response(
            new_response: AssistantResponse,
            original_response: AssistantResponse
    ) -> bool:
        """Check if the new response is better than the original."""
        # The new response should be successful
        if not new_response.success:
            return False

        # Should have function calls or substantial content
        has_functions = len(new_response.function_calls) > 0
        has_content = len(new_response.content) > 100

        # Compare to the original
        more_functions = (
                len(new_response.function_calls) >
                len(original_response.function_calls)
        )
        more_content = (
                len(new_response.content) >
                len(original_response.content) * 1.2
        )

        return (has_functions or has_content) and (more_functions or more_content)