"""
Main OpenAI Integration Service.
Orchestrates async query processing and component coordination.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from .config import IntegrationConfig
from .analytics import IntegrationAnalytics
from ..handlers.query_handler import QueryHandler

logger = logging.getLogger(__name__)


class OpenAIIntegrationService:
    """
    Service for integrating and managing OpenAI-based operations.

    This class serves as the primary interface for the OpenAI Integration,
    enabling processing of user queries, conversation management, health
    checks, and analytics tracking. It initializes required components such
    as configuration, analytics, and query handlers to facilitate secured
    and robust interaction with the OpenAI assistant.

    Attributes
    ----------
    config : IntegrationConfig
        Configuration object for managing dependencies and settings.
    analytics : IntegrationAnalytics
        Handles analytics tracking related to query processing outcomes.
    query_handler : QueryHandler
        Manages processing of user queries with the OpenAI assistant and its
        related functionality.
    """

    def __init__(self):
        """Initialize service with all components."""
        # Initialize configuration and components
        self.config = IntegrationConfig()

        # Initialize analytics
        self.analytics = IntegrationAnalytics(
            self.config.intent_service, self.config.intent_mapper
        )

        # Initialize query handler with dependencies
        self.query_handler = QueryHandler(
            assistant_manager=self.config.assistant_manager,
            intent_service=self.config.intent_service,
            router=self.config.router,
            intent_mapper=self.config.intent_mapper,
            function_processor=self.config.function_processor,
        )

        logger.info("✅ OpenAI Integration Service initialized")

    async def process_financial_query(
        self,
        query: str,
        user_id: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Process a financial query asynchronously.

        Args:
            query: User's query text
            user_id: User identifier
            user_context: User context data
            conversation_history: Previous messages

        Returns:
            Formatted response dictionary
        """
        start_time = time.time()

        try:
            # Process query through handler
            response = await self.query_handler.process_query(
                query=query,
                user_id=user_id,
                user_context=user_context,
                conversation_history=conversation_history,
            )

            # Track analytics
            response_time = time.time() - start_time
            self.analytics.track_query(
                user_id=user_id,
                query=query,
                intent=response.get("classification", {}).get("intent", "unknown"),
                assistant_used=response.get("classification", {}).get(
                    "assistant_used", "unknown"
                ),
                success=response.get("success", False),
                response_time=response_time,
            )

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            return self._create_error_response(e, query, user_id)

    def clear_conversation(self, user_id: str):
        """Clear conversation history for a user."""
        self.config.assistant_manager.clear_thread(user_id)
        logger.info(f"🗑️ Cleared conversation for user {user_id}")

    async def get_conversation_history(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history for a user."""
        return await self.config.assistant_manager.get_conversation_history(
            user_id, limit
        )

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        health_results = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        # Check configuration
        config_validation = await self.config.validate()
        health_results["components"]["configuration"] = config_validation
        if not config_validation["is_valid"]:
            health_results["status"] = "degraded"

        # Check assistant manager
        try:
            assistant_health = await self.config.assistant_manager.health_check()
            health_results["components"]["assistant_manager"] = assistant_health
            if assistant_health["status"] != "healthy":
                health_results["status"] = "degraded"
        except Exception as e:
            health_results["components"]["assistant_manager"] = {
                "status": "error",
                "error": str(e),
            }
            health_results["status"] = "unhealthy"

        # Summary
        health_results["summary"] = {
            "available_assistants": self._get_available_assistants(),
            "missing_assistants": self._get_missing_assistants(),
        }

        return health_results

    async def get_analytics(self, user_id: str, recent_queries: List[str]) -> Dict[str, Any]:
        """Get analytics for user queries."""
        return await self.analytics.get_analytics(user_id, recent_queries)

    def _get_available_assistants(self) -> int:
        """Get count of available assistants."""
        validation = self.config.assistant_config.validate()
        return len(validation["configured_assistants"])

    def _get_missing_assistants(self) -> List[str]:
        """Get list of missing assistants."""
        validation = self.config.assistant_config.validate()
        return validation["missing_assistants"]

    @staticmethod
    def _create_error_response(
        error: Exception, query: str, user_id: str
    ) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "message": "I encountered an error processing your request.",
            "response_text": "I encountered an error processing your request.",
            "actions": [],
            "tool_results": [],
            "classification": {
                "intent": "general",
                "confidence": 0.0,
                "assistant_used": "general",
                "method": "error",
            },
            "routing": {"strategy": "error"},
            "success": False,
            "error": str(error),
            "metadata": {
                "user_id": user_id,
                "query_length": len(query),
                "error_type": type(error).__name__,
                "timestamp": datetime.now().isoformat(),
            },
        }
