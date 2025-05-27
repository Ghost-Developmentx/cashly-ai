"""
Simplified OpenAI Integration Service using composed components.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .assistant_manager import AssistantManager, AssistantType
from ..intent_classification.intent_service import IntentService

from .core.router import AssistantRouter
from .core.intent_mapper import IntentMapper
from .core.response_builder import ResponseBuilder
from .processors.function_processor import FunctionProcessor
from .handlers.query_handler import QueryHandler
from .utils.constants import CROSS_ROUTING_PATTERNS, INTENT_TO_ASSISTANT_MAPPING

logger = logging.getLogger(__name__)


class OpenAIIntegrationService:
    """
    Simplified integration service using composed components.

    This service orchestrates:
    - Intent classification
    - Assistant routing
    - Query processing
    - Response formatting
    """

    def __init__(self):
        # Initialize core managers
        self.assistant_manager = AssistantManager()
        self.intent_service = IntentService()

        # Initialize components with dependency injection
        self.router = AssistantRouter(CROSS_ROUTING_PATTERNS)
        self.intent_mapper = IntentMapper()
        self.function_processor = FunctionProcessor()

        # Initialize the main query handler
        self.query_handler = QueryHandler(
            assistant_manager=self.assistant_manager,
            intent_service=self.intent_service,
            router=self.router,
            intent_mapper=self.intent_mapper,
            function_processor=self.function_processor,
        )

        # Setup tool executor
        self._setup_tool_executor()

        logger.info("✅ OpenAI Integration Service initialized")

    def _setup_tool_executor(self):
        """Set up the tool executor from existing Fin service."""
        try:
            from services.fin.tool_registry import ToolRegistry

            tool_registry = ToolRegistry()

            def tool_executor_wrapper(tool_name, tool_args, **kwargs):
                """Wrapper that calls tool registry with correct signature."""
                user_id = kwargs.get("user_id", "unknown")
                user_context = kwargs.get("user_context", {})
                transactions = kwargs.get("transactions", [])

                try:
                    return tool_registry.execute(
                        tool_name,
                        tool_args,
                        user_id=user_id,
                        transactions=transactions,
                        user_context=user_context,
                    )
                except Exception as e:
                    logger.error(f"Tool execution failed for {tool_name}: {e}")
                    return {"error": f"Tool execution failed: {str(e)}"}

            self.assistant_manager.set_tool_executor(tool_executor_wrapper)
            logger.info("✅ Tool executor connected successfully")

        except Exception as e:
            logger.error(f"❌ Failed to connect tool executor: {e}")

    async def process_financial_query(
        self,
        query: str,
        user_id: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Process a financial query.

        This is the main entry point that delegates to the query handler.

        Args:
            query: User's query text
            user_id: User identifier
            user_context: User context (accounts, integrations, etc.)
            conversation_history: Previous conversation messages

        Returns:
            Formatted response dictionary
        """
        return await self.query_handler.process_query(
            query=query,
            user_id=user_id,
            user_context=user_context,
            conversation_history=conversation_history,
        )

    def clear_conversation(self, user_id: str):
        """
        Clear conversation history for a user.

        Args:
            user_id: User identifier
        """
        self.assistant_manager.clear_thread(user_id)
        logger.info(f"🗑️ Cleared conversation for user {user_id}")

    def get_conversation_history(
        self, user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.

        Args:
            user_id: User identifier
            limit: Maximum number of messages to return

        Returns:
            List of conversation messages
        """
        return self.assistant_manager.get_conversation_history(user_id, limit)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform comprehensive health check.

        Returns:
            Health status of all components
        """
        health_components = {}
        overall_status = "healthy"

        # Check assistant manager
        try:
            assistant_health = self.assistant_manager.health_check()
            health_components["assistant_manager"] = assistant_health
            if assistant_health["status"] != "healthy":
                overall_status = "degraded"
        except Exception as e:
            health_components["assistant_manager"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            overall_status = "unhealthy"

        # Check intent service
        try:
            test_result = self.intent_service.classify_and_route("test query")
            health_components["intent_service"] = {
                "status": "healthy",
                "test_intent": test_result["classification"]["intent"],
            }
        except Exception as e:
            health_components["intent_service"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            overall_status = "unhealthy"

        # Check components
        health_components["components"] = {
            "router": {"status": "healthy" if self.router else "not_initialized"},
            "intent_mapper": {
                "status": "healthy" if self.intent_mapper else "not_initialized"
            },
            "function_processor": {
                "status": "healthy" if self.function_processor else "not_initialized"
            },
            "query_handler": {
                "status": "healthy" if self.query_handler else "not_initialized"
            },
        }

        # Summary
        return {
            "status": overall_status,
            "components": health_components,
            "available_assistants": self._get_available_assistants(),
            "missing_assistants": self._get_missing_assistants(),
            "timestamp": self._get_timestamp(),
        }

    def get_analytics(self, user_id: str, recent_queries: List[str]) -> Dict[str, Any]:
        """
        Get analytics for recent queries.

        Args:
            user_id: User identifier
            recent_queries: List of recent query texts

        Returns:
            Analytics data
        """
        try:
            # Get intent analytics
            intent_analytics = self.intent_service.get_routing_analytics(recent_queries)

            # Calculate assistant usage
            assistant_usage = {}
            for query in recent_queries:
                routing = self.intent_service.classify_and_route(query)
                intent = routing["classification"]["intent"]
                assistant = self.intent_mapper.get_default_assistant(intent)
                assistant_usage[assistant.value] = (
                    assistant_usage.get(assistant.value, 0) + 1
                )

            return {
                "intent_analytics": intent_analytics,
                "assistant_usage": assistant_usage,
                "total_queries": len(recent_queries),
                "user_id": user_id,
            }

        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "total_queries": len(recent_queries),
            }

    def _get_available_assistants(self) -> int:
        """Get count of available assistants."""
        return len(
            [aid for aid in self.assistant_manager.assistant_ids.values() if aid]
        )

    def _get_missing_assistants(self) -> List[str]:
        """Get list of missing assistants."""
        return [
            atype.value
            for atype, aid in self.assistant_manager.assistant_ids.items()
            if not aid
        ]

    @staticmethod
    def _get_timestamp() -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()
