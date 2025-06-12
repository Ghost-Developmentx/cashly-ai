"""
Main query pipeline orchestrator.
Coordinates the flow through all pipeline stages.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from ...schemas.classification import ClassificationResult
from app.core.assistants import UnifiedAssistantManager
from .classifier import QueryClassifier
from .router import AssistantRouter, RoutingDecision
from .executor import QueryExecutor, ExecutionResult
from .formatter import ResponseFormatter

logger = logging.getLogger(__name__)

class QueryPipeline:
    """
    Orchestrates query processing through the pipeline stages.
    Provides a clean, linear flow from query to response.
    """

    def __init__(
            self,
            assistant_manager: Optional[UnifiedAssistantManager] = None,
            enable_ml_classification: bool = False
    ):
        """
        Initialize a pipeline with components.

        Args:
            assistant_manager: Optional assistant manager instance
            enable_ml_classification: Whether to use ML for classification
        """
        # Initialize components
        self.classifier = QueryClassifier()
        self.router = AssistantRouter()
        self.executor = QueryExecutor(assistant_manager)
        self.formatter = ResponseFormatter()

        # Configuration
        self.enable_ml_classification = enable_ml_classification
        self.enable_rerouting = True
        self.max_reroute_attempts = 1

        logger.info("Query pipeline initialized")

    async def process_query(
            self,
            query: str,
            user_id: str,
            user_context: Optional[Dict[str, Any]] = None,
            conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the complete pipeline.

        Args:
            query: User's query
            user_id: User identifier
            user_context: Optional user context
            conversation_history: Optional conversation history

        Returns:
            Formatted response dictionary
        """
        start_time = time.time()

        try:
            # Stage 1: Classification
            classification = await self._classify(query, conversation_history)

            # Stage 2: Routing
            routing = await self._route(classification, query, user_context)

            # Stage 3: Execution
            execution_result = await self._execute(
                query=query,
                routing=routing,
                user_id=user_id,
                user_context=user_context,
                conversation_history=conversation_history
            )

            # Stage 4: Check for rerouting
            if self.enable_rerouting and execution_result.success:
                reroute_result = await self._check_and_handle_reroute(
                    execution_result=execution_result,
                    routing=routing,
                    query=query,
                    user_id=user_id,
                    user_context=user_context,
                    conversation_history=conversation_history
                )
                if reroute_result:
                    execution_result = reroute_result

            # Stage 5: Format response
            processing_time = time.time() - start_time

            response = self.formatter.format_success_response(
                execution_result=execution_result,
                classification=classification,
                routing=routing,
                query=query,
                user_id=user_id,
                processing_time=processing_time
            )

            # Log pipeline metrics
            self._log_pipeline_metrics(
                classification=classification,
                routing=routing,
                execution_result=execution_result,
                processing_time=processing_time
            )

            return response

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

            # Format error response
            return self.formatter.format_error_response(
                error=e,
                query=query,
                user_id=user_id,
                classification=classification if 'classification' in locals() else None,
                routing=routing if 'routing' in locals() else None
            )

    async def _classify(
            self,
            query: str,
            conversation_history: Optional[List[Dict[str, Any]]]
    ) -> ClassificationResult:
        """Stage 1: Classify the query."""
        logger.debug(f"Classifying query: {query[:50]}...")

        if self.enable_ml_classification:
            return await self.classifier.classify_with_ml(query, conversation_history)
        else:
            return await self.classifier.classify(query, conversation_history)

    async def _route(
            self,
            classification: ClassificationResult,
            query: str,
            user_context: Optional[Dict[str, Any]]
    ) -> RoutingDecision:
        """Stage 2: Route to appropriate assistant."""
        logger.debug(f"Routing based on intent: {classification.intent.value}")

        return await self.router.route(classification, query, user_context)

    async def _execute(
            self,
            query: str,
            routing: RoutingDecision,
            user_id: str,
            user_context: Optional[Dict[str, Any]],
            conversation_history: Optional[List[Dict[str, Any]]]
    ) -> ExecutionResult:
        """Stage 3: Execute with selected assistant."""
        logger.debug(f"Executing with assistant: {routing.assistant.value}")

        # Use fallback if routing suggests it
        if routing.should_reroute and routing.alternative_assistant:
            return await self.executor.execute_with_fallback(
                query=query,
                primary_assistant=routing.assistant,
                fallback_assistant=routing.alternative_assistant,
                user_id=user_id,
                user_context=user_context,
                conversation_history=conversation_history
            )
        else:
            return await self.executor.execute_with_retry(
                query=query,
                assistant_type=routing.assistant,
                user_id=user_id,
                user_context=user_context,
                conversation_history=conversation_history
            )

    async def _check_and_handle_reroute(
            self,
            execution_result: ExecutionResult,
            routing: RoutingDecision,
            query: str,
            user_id: str,
            user_context: Optional[Dict[str, Any]],
            conversation_history: Optional[List[Dict[str, Any]]],
            attempt: int = 0
    ) -> Optional[ExecutionResult]:
        """Check if response suggests rerouting and handle it."""
        if attempt >= self.max_reroute_attempts:
            return None

        # Check if response suggests rerouting
        reroute_assistant = self.router.should_reroute_response(
            assistant_response=execution_result.assistant_response.content,
            current_assistant=execution_result.assistant_response.assistant_type,
            original_query=query
        )

        if not reroute_assistant:
            return None

        logger.info(f"Rerouting from {routing.assistant.value} to {reroute_assistant.value}")

        # Execute with new assistant
        new_result = await self.executor.execute(
            query=query,
            assistant_type=reroute_assistant,
            user_id=user_id,
            user_context=user_context,
            conversation_history=conversation_history
        )

        # Add reroute metadata
        if new_result.assistant_response.metadata is None:
            new_result.assistant_response.metadata = {}

        new_result.assistant_response.metadata.update({
            "rerouted": True,
            "rerouted_from": routing.assistant.value,
            "reroute_reason": "Response suggested alternative assistant"
        })

        return new_result

    @staticmethod
    def _log_pipeline_metrics(
            classification: ClassificationResult,
            routing: RoutingDecision,
            execution_result: ExecutionResult,
            processing_time: float
    ):
        """Log pipeline performance metrics."""
        metrics = {
            "intent": classification.intent.value,
            "classification_confidence": round(classification.confidence, 3),
            "routing_confidence": round(routing.confidence, 3),
            "assistant_used": execution_result.assistant_response.assistant_type.value,
            "execution_time": round(execution_result.execution_time, 3),
            "total_time": round(processing_time, 3),
            "tool_calls": execution_result.tool_calls_count,
            "success": execution_result.success
        }

        logger.info(f"Pipeline metrics: {metrics}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of all pipeline components.

        Returns:
            Health status of each component
        """
        health = {
            "status": "healthy",
            "components": {
                "classifier": {"status": "healthy"},
                "router": {"status": "healthy"},
                "executor": {"status": "unknown"},
                "formatter": {"status": "healthy"}
            }
        }

        # Check executor (which depends on assistant manager)
        try:
            manager_health = await self.executor.assistant_manager.health_check()
            health["components"]["executor"] = {
                "status": manager_health.get("status", "unknown"),
                "assistant_manager": manager_health
            }
        except Exception as e:
            health["components"]["executor"] = {
                "status": "error",
                "error": str(e)
            }
            health["status"] = "degraded"

        return health
