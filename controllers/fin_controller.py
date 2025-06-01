""" "
Controller for Fin conversational AI endpoints.
Handles natural language query processing and OpenAI Assistant integration.
"""

from typing import Dict, Any, Tuple
from datetime import datetime
from flask import g
from controllers.base_controller import BaseController
from services.openai_assistants import OpenAIIntegrationService
from middleware.async_manager import run_async


class FinController(BaseController):
    """Controller for Fin conversational AI operations"""

    def __init__(self):
        super().__init__()
        self.openai_service = OpenAIIntegrationService()

    def process_query(self) -> Tuple[Dict[str, Any], int]:
        """
        Process a natural language query using OpenAI Assistants.

        Expected JSON input:
        {
            "user_id": "user_123",
            "query": "Show me my transactions",
            "transactions": [...],
            "conversation_history": [...],
            "user_context": {...}
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id", "query"])

            # Extract parameters
            user_id = data.get("user_id")
            query = data.get("query")
            transactions = data.get("transactions", [])
            conversation_history = data.get("conversation_history", [])
            user_context = data.get("user_context", {})

            # Add conversation_id to user_context if available
            if conversation_history and not user_context.get("conversation_id"):
                user_context["conversation_id"] = (
                    f"conv_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )

            # Ensure user_id is in context
            if "user_id" not in user_context:
                user_context["user_id"] = user_id

            # Log request details
            self.logger.info(f"📥 OpenAI Fin query from user {user_id}: {query}")
            self.logger.info(
                f"📥 User context: {len(user_context.get('accounts', []))} accounts"
            )
            self.logger.info(f"📥 Transactions: {len(transactions)}")
            self.logger.info(
                f"📥 Conversation history: {len(conversation_history)} messages"
            )

            # Add transactions to user_context
            if user_context and transactions:
                user_context["transactions"] = transactions
                self.logger.info(
                    f"📥 Added {len(transactions)} transactions to user_context"
                )

            # Process query using async manager
            result = self._process_query_async(
                query=query,
                user_id=user_id,
                user_context=user_context,
                conversation_history=conversation_history,
            )

            # Log response details
            self.logger.info("📤 OpenAI response generated successfully")
            self.logger.info(f"📤 Intent: {result['classification']['intent']}")
            self.logger.info(
                f"📤 Assistant: {result['classification']['assistant_used']}"
            )
            self.logger.info(f"📤 Actions: {len(result.get('actions', []))}")

            # Log action details
            if result.get("actions"):
                for i, action in enumerate(result["actions"]):
                    self.logger.info(f"📤 Action {i}: {action.get('type')}")

            return self.success_response(result)

        return self.handle_request(_handle)

    @run_async
    async def _process_query_async(
        self,
        query: str,
        user_id: str,
        user_context: Dict[str, Any],
        conversation_history: list,
    ) -> Dict[str, Any]:
        """Process query asynchronously using the centralized event loop."""
        return await self.openai_service.process_financial_query(
            query=query,
            user_id=user_id,
            user_context=user_context,
            conversation_history=conversation_history,
        )

    def health_check(self) -> Tuple[Dict[str, Any], int]:
        """
        Health check for the OpenAI Assistants system.

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            health = self._health_check_async()

            response_data = {
                "status": health["status"],
                "components": health["components"],
                "timestamp": datetime.now().isoformat(),
            }

            # Add async manager status
            if hasattr(g, "async_manager"):
                response_data["async_manager"] = "healthy"
            else:
                response_data["async_manager"] = "unavailable"

            status_code = 200 if health["status"] == "healthy" else 503
            return response_data, status_code

        return self.handle_request(_handle)

    @run_async
    async def _health_check_async(self) -> Dict[str, Any]:
        """Run health check asynchronously."""
        return await self.openai_service.health_check()

    def get_analytics(self) -> Tuple[Dict[str, Any], int]:
        """
        Get analytics for recent queries and assistant usage.

        Expected JSON input:
        {
            "user_id": "user_123",
            "recent_queries": [...]
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id"])

            user_id = data.get("user_id")
            recent_queries = data.get("recent_queries", [])

            analytics = self.openai_service.get_analytics(user_id, recent_queries)

            return self.success_response(analytics)

        return self.handle_request(_handle)
