"""
Controller for Fin conversational AI endpoints.
Handles natural language query processing and OpenAI Assistant integration.
"""

import asyncio
from typing import Dict, Any, Tuple
from datetime import datetime
from controllers.base_controller import BaseController
from services.openai_assistants import OpenAIIntegrationService


class FinController(BaseController):
    """Controller for Fin conversational AI operations"""

    def __init__(self):
        super().__init__()
        self.openai_service = OpenAIIntegrationService()

    def process_query(self) -> Tuple[Dict[str, Any], int]:
        """
        Process a natural language query using OpenAI Assistants with intent classification

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

            # Log request details with debugging info
            self.logger.info(f"📥 OpenAI Fin query from user {user_id}: {query}")
            self.logger.info(
                f"📥 User context: {len(user_context.get('accounts', []))} accounts"
            )
            self.logger.info(f"📥 Transactions: {len(transactions)}")

            # Add transactions to user_context for assistant access
            if user_context and transactions:
                user_context["transactions"] = transactions
                self.logger.info(
                    f"📥 Added {len(transactions)} transactions to user_context"
                )

            # Process the query asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.openai_service.process_financial_query(
                        query=query,
                        user_id=user_id,
                        user_context=user_context,
                        conversation_history=conversation_history,
                    )
                )
            finally:
                loop.close()

            # Log response details
            self.logger.info("📤 OpenAI response generated successfully")
            self.logger.info(f"📤 Intent: {result['classification']['intent']}")
            self.logger.info(
                f"📤 Assistant: {result['classification']['assistant_used']}"
            )
            self.logger.info(f"📤 Actions: {len(result.get('actions', []))}")

            # Log detailed action information for debugging
            if result.get("actions"):
                for i, action in enumerate(result["actions"]):
                    self.logger.info(f"📤 Action {i}: {action.get('type')}")

            return self.success_response(result)

        return self.handle_request(_handle)

    def health_check(self) -> Tuple[Dict[str, Any], int]:
        """
        Health check for the OpenAI Assistants system

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            health = self.openai_service.health_check()

            response_data = {
                "status": health["status"],
                "components": health["components"],
                "available_assistants": health["available_assistants"],
                "missing_assistants": health["missing_assistants"],
                "timestamp": datetime.now().isoformat(),
            }

            # Return the appropriate status code based on health
            status_code = 200 if health["status"] == "healthy" else 503

            return response_data, status_code

        return self.handle_request(_handle)

    def get_analytics(self) -> Tuple[Dict[str, Any], int]:
        """
        Get analytics for recent queries and assistant usage

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
