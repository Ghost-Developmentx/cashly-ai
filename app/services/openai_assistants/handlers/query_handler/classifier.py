"""
Classification handling for queries.
Manages intent classification and initial routing.
"""

import logging
from typing import Dict, Any
from .types import QueryContext
from ...assistant_manager import AssistantType

logger = logging.getLogger(__name__)

class ClassificationHandler:
    """
    Handles classification of user queries and mapping to corresponding assistants.

    This class is responsible for processing user queries, classifying their intents,
    and routing them to the appropriate assistant type based on classification results
    and business logic. It leverages an intent service and an intent mapper for
    classification and fallback mechanisms.

    Attributes
    ----------
    intent_service : Any
        Service utility for running the intent classification and routing.
    intent_mapper : Any
        Mapper utility for determining assistants based on intents and fallback
        logic.
    """

    def __init__(self, intent_service, intent_mapper):
        self.intent_service = intent_service
        self.intent_mapper = intent_mapper

    async def classify_query(
            self,
            context: QueryContext
    ) -> Dict[str, Any]:
        """
        Classify the intent of a query.

        Args:
            context: Query context

        Returns:
            Classification result
        """
        logger.info(f"🎯 Classifying query for user {context.user_id}")

        try:
            # Run classification
            routing_result = await self._run_classification(context)

            # Log results
            classification = routing_result["classification"]
            logger.info(
                f"🎯 Intent: {classification['intent']} "
                f"(confidence: {classification['confidence']:.2%})"
            )

            return routing_result

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return self._create_fallback_classification()

    async def _run_classification(
            self,
            context: QueryContext
    ) -> Dict[str, Any]:
        """Run async classification."""
        # Intent service might be sync, so we handle both cases
        import asyncio
        import inspect

        if inspect.iscoroutinefunction(self.intent_service.classify_and_route):
            return await self.intent_service.classify_and_route(
                context.query,
                context.user_context,
                context.conversation_history
            )
        else:
            # Run sync method in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self.intent_service.classify_and_route,
                context.query,
                context.user_context,
                context.conversation_history
            )

    def determine_initial_assistant(
            self,
            classification_result: Dict[str, Any],
            context: QueryContext
    ) -> AssistantType:
        """
        Determine an initial assistant based on classification.

        Args:
            classification_result: Classification results
            context: Query context

        Returns:
            Initial assistant type
        """
        # Try to get a recommended assistant
        recommended = classification_result.get(
            "recommended_assistant",
            "transaction_assistant"
        )

        try:
            # Remove the '_ assistant' suffix if present
            assistant_name = recommended.replace("_assistant", "")
            return AssistantType(assistant_name)
        except ValueError:
            # Fallback to intent mapper
            return self._fallback_to_intent_mapper(
                classification_result,
                context
            )

    def _fallback_to_intent_mapper(
            self,
            classification_result: Dict[str, Any],
            context: QueryContext
    ) -> AssistantType:
        """Use intent mapper as a fallback."""
        intent = classification_result["classification"]["intent"]

        return self.intent_mapper.get_assistant_for_intent(
            intent,
            classification_result,
            context.query,
            context.conversation_history
        )

    @staticmethod
    def _create_fallback_classification() -> Dict[str, Any]:
        """Create a fallback classification for errors."""
        return {
            "classification": {
                "intent": "general",
                "confidence": 0.0,
                "method": "error_fallback"
            },
            "routing": {
                "strategy": "error",
                "primary_assistant": "transaction_assistant",
                "confidence": 0.0
            },
            "recommended_assistant": "transaction_assistant"
        }