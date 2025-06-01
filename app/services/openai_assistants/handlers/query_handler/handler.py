"""
Main query handler that orchestrates processing.
Coordinates all components for end-to-end query handling.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from .types import QueryContext, ProcessingResult
from .classifier import ClassificationHandler
from .router import RoutingHandler
from ...assistant_manager import AssistantType
from ...core.response_builder import ResponseBuilder

logger = logging.getLogger(__name__)

class QueryHandler:
    """
    Handles user query processing by integrating classification, routing, and
    assistant management systems.

    This class is designed to manage complete query lifecycle processing, from
    classification of user intents to routing the query to appropriate assistants,
    processing function calls, and generating a final response. It utilizes internal
    handlers for classification and routing, as well as an assistant management
    system to facilitate these tasks. This handler streamlines query handling for
    natural language processing or conversational AI systems.

    Attributes
    ----------
    assistant_manager : Any
        Manager responsible for handling assistant responses and interactions.
    function_processor : Any
        Processor responsible for transforming function calls into actionable
        outputs or actions.
    classifier : ClassificationHandler
        Handles classification of user queries and determines initial assistants.
    router : RoutingHandler
        Handles rerouting decisions and execution for queries that require
        alternate assistants.
    """

    def __init__(
            self,
            assistant_manager,
            intent_service,
            router,
            intent_mapper,
            function_processor
    ):
        """Initialize with required dependencies."""
        self.assistant_manager = assistant_manager
        self.function_processor = function_processor

        # Initialize sub-handlers
        self.classifier = ClassificationHandler(intent_service, intent_mapper)
        self.router = RoutingHandler(router, assistant_manager, intent_service)

        logger.info("Query handler initialized")

    async def process_query(
            self,
            query: str,
            user_id: str,
            user_context: Optional[Dict] = None,
            conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Process a query end-to-end.

        Args:
            query: User's query
            user_id: User identifier
            user_context: Optional user context
            conversation_history: Optional conversation history

        Returns:
            Formatted response dictionary
        """
        start_time = time.time()

        # Create query context
        context = QueryContext(
            query=query,
            user_id=user_id,
            user_context=user_context,
            conversation_history=conversation_history
        )

        try:
            # Process the query
            result = await self._process_query_async(context)

            # Build response
            response = ResponseBuilder.build_response(
                assistant_response=result.assistant_response,
                actions=result.actions,
                classification=result.classification["classification"],
                routing_result=result.routing_result,
                query=query,
                user_id=user_id,
                final_assistant=result.final_assistant,
                initial_assistant=result.initial_assistant
            )

            # Add processing time
            response["metadata"]["processing_time"] = time.time() - start_time

            return response

        except Exception as e:
            logger.error(f"Query processing failed: {e}", exc_info=True)
            return ResponseBuilder.build_error_response(e, query, user_id)

    async def _process_query_async(
            self,
            context: QueryContext
    ) -> ProcessingResult:
        """Core async query processing logic."""
        # Step 1: Classify intent
        classification_result = await self.classifier.classify_query(context)

        # Step 2: Determine initial assistant
        initial_assistant = self.classifier.determine_initial_assistant(
            classification_result,
            context
        )

        # Step 3: Process with initial assistant
        assistant_response = await self.assistant_manager.process_query(
            query=context.query,
            assistant_type=initial_assistant,
            user_id=context.user_id,
            user_context=context.user_context,
            conversation_history=context.conversation_history
        )

        # Step 4: Check for rerouting
        final_assistant = initial_assistant
        rerouted = False

        routing_decision = self.router.should_attempt_reroute(
            assistant_response,
            initial_assistant,
            context
        )

        if routing_decision.should_reroute:
            reroute_result = await self.router.attempt_reroute(
                assistant_response,
                routing_decision,
                context
            )

            if reroute_result:
                assistant_response, final_assistant = reroute_result
                rerouted = True

        # Step 5: Process function calls
        actions = self.function_processor.process_function_calls_to_actions(
            assistant_response.function_calls
        )

        # Return complete result
        return ProcessingResult(
            assistant_response=assistant_response,
            actions=actions,
            classification=classification_result,
            routing_result=classification_result,
            initial_assistant=initial_assistant,
            final_assistant=final_assistant,
            processing_time=0.0,  # Will be set by caller
            rerouted=rerouted
        )