"""
Main query processing handler that orchestrates all components.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from ..core.router import AssistantRouter
from ..core.intent_mapper import IntentMapper
from ..core.response_builder import ResponseBuilder
from ..processors.function_processor import FunctionProcessor
from ..assistant_manager import AssistantType, AssistantResponse

logger = logging.getLogger(__name__)


class QueryHandler:
    """Orchestrates query processing using all components."""

    def __init__(
        self,
        assistant_manager,
        intent_service,
        router: AssistantRouter,
        intent_mapper: IntentMapper,
        function_processor: FunctionProcessor,
    ):
        self.assistant_manager = assistant_manager
        self.intent_service = intent_service
        self.router = router
        self.intent_mapper = intent_mapper
        self.function_processor = function_processor

    async def process_query(
        self,
        query: str,
        user_id: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Process financial query with routing and re-routing capabilities.

        Args:
            query: User's query text
            user_id: User identifier
            user_context: User context including accounts, integrations, etc.
            conversation_history: Previous conversation messages

        Returns:
            Formatted response dictionary
        """
        try:
            # Step 1: Classify intent
            classification_result = self._classify_intent(
                query, user_context, conversation_history
            )

            # Step 2: Determine initial assistant
            initial_assistant = self._determine_initial_assistant(
                classification_result, query, conversation_history
            )

            # Step 3: Process with initial assistant
            assistant_response = await self._process_with_assistant(
                initial_assistant, query, user_id, user_context, conversation_history
            )

            # Step 4: Check for re-routing
            final_assistant = initial_assistant
            if self._should_attempt_reroute(
                assistant_response, initial_assistant, query
            ):
                rerouted = await self._attempt_reroute(
                    assistant_response,
                    initial_assistant,
                    query,
                    user_id,
                    user_context,
                    conversation_history,
                )
                if rerouted:
                    assistant_response, final_assistant = rerouted

            # Step 5: Process function calls to actions
            actions = self.function_processor.process_function_calls_to_actions(
                assistant_response.function_calls
            )

            # Step 6: Build final response
            return ResponseBuilder.build_response(
                assistant_response=assistant_response,
                actions=actions,
                classification=classification_result["classification"],
                routing_result=classification_result,
                query=query,
                user_id=user_id,
                final_assistant=final_assistant,
                initial_assistant=initial_assistant,
            )

        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            import traceback

            traceback.print_exc()

            return ResponseBuilder.build_error_response(e, query, user_id)

    def _classify_intent(
        self,
        query: str,
        user_context: Optional[Dict],
        conversation_history: Optional[List[Dict]],
    ) -> Dict[str, Any]:
        """Classify the intent of the query."""
        routing_result = self.intent_service.classify_and_route(
            query, user_context, conversation_history
        )

        classification = routing_result["classification"]
        logger.info(
            f"🎯 Intent: {classification['intent']} "
            f"(confidence: {classification['confidence']:.2%})"
        )

        return routing_result

    def _determine_initial_assistant(
        self,
        classification_result: Dict,
        query: str,
        conversation_history: Optional[List[Dict]],
    ) -> AssistantType:
        """Determine the initial assistant to use."""
        initial_assistant = self.intent_mapper.get_assistant_for_intent(
            classification_result["classification"]["intent"],
            classification_result,
            query,
            conversation_history,
        )

        logger.info(f"🤖 Initial assistant: {initial_assistant.value}")
        return initial_assistant

    async def _process_with_assistant(
        self,
        assistant_type: AssistantType,
        query: str,
        user_id: str,
        user_context: Optional[Dict],
        conversation_history: Optional[List[Dict]],
    ) -> AssistantResponse:
        """Process query with specified assistant."""
        assistant_response = await self.assistant_manager.process_query(
            query=query,
            assistant_type=assistant_type,
            user_id=user_id,
            user_context=user_context,
            conversation_history=conversation_history,
        )

        logger.info(f"🔧 Assistant response success: {assistant_response.success}")
        logger.info(
            f"🔧 Function calls count: {len(assistant_response.function_calls)}"
        )
        logger.info(f"🔧 Response content length: {len(assistant_response.content)}")

        return assistant_response

    def _should_attempt_reroute(
        self,
        assistant_response: AssistantResponse,
        current_assistant: AssistantType,
        query: str,
    ) -> bool:
        """Check if re-routing should be attempted."""
        return self.router.should_reroute(assistant_response, current_assistant, query)

    async def _attempt_reroute(
        self,
        assistant_response: AssistantResponse,
        initial_assistant: AssistantType,
        query: str,
        user_id: str,
        user_context: Optional[Dict],
        conversation_history: Optional[List[Dict]],
    ) -> Optional[Tuple[AssistantResponse, AssistantType]]:
        """
        Attempt to re-route to a more appropriate assistant.

        Returns:
            Tuple of (new_response, new_assistant) if successful, None otherwise
        """
        logger.info("🔄 Attempting seamless re-routing")

        correct_assistant = self.router.determine_correct_assistant(
            assistant_response, initial_assistant, query, self.intent_service
        )

        if correct_assistant and correct_assistant != initial_assistant:
            logger.info(
                f"🔄 Re-routing: {initial_assistant.value} -> {correct_assistant.value}"
            )

            rerouted_response = await self._process_with_assistant(
                correct_assistant, query, user_id, user_context, conversation_history
            )

            if rerouted_response.success and len(rerouted_response.function_calls) > 0:
                logger.info("✅ Re-routing successful")
                return rerouted_response, correct_assistant
            else:
                logger.info("⚠️ Re-routing didn't improve response, using original")

        return None
