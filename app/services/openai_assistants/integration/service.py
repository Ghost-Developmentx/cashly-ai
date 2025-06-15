"""
Main OpenAI Integration Service.
Uses the new QueryPipeline for all query processing.
"""

import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from app.core.pipeline import QueryPipeline
from app.core.assistants import UnifiedAssistantManager

logger = logging.getLogger(__name__)



class OpenAIIntegrationService:
    """
    Service for integrating and managing OpenAI-based operations.

    This is now a thin wrapper around the QueryPipeline that handles:
    - Analytics tracking
    - Conversation management
    - Health checks
    """

    def __init__(self):
        """Initialize service with pipeline and assistant manager."""
        # Initialize core components
        self.assistant_manager = UnifiedAssistantManager()

        # Initialize pipeline with ML classification enabled
        self.pipeline = QueryPipeline(
            assistant_manager=self.assistant_manager,
            enable_ml_classification=True
        )

        # Simple analytics tracking (consider moving to a separate service)
        self._query_metrics = []

        logger.info("✅ OpenAI Integration Service initialized with QueryPipeline")

    async def process_financial_query(
        self,
        query: str,
        user_id: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Process a financial query using the pipeline.

        Args:
            query: User's query text
            user_id: User identifier
            user_context: User context data
            conversation_history: Previous messages

        Returns:
            Formatted response dictionary from pipeline
        """
        start_time = time.time()

        try:
            # Process through pipeline
            response = await self.pipeline.process_query(
                query=query,
                user_id=user_id,
                user_context=user_context,
                conversation_history=conversation_history,
            )

            # Track metrics
            response_time = time.time() - start_time
            self._track_query_metrics(
                user_id=user_id,
                query=query,
                response=response,
                response_time=response_time
            )

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            # The pipeline already handles errors, but this is a fallback
            return self._create_error_response(e, query, user_id)

    def clear_conversation(self, user_id: str):
        """Clear conversation history for a user."""
        success = self.assistant_manager.clear_user_thread(user_id)
        if success:
            logger.info(f"🗑️ Cleared conversation for user {user_id}")
        else:
            logger.warning(f"No conversation found for user {user_id}")

    @staticmethod
    async def get_conversation_history(
            user_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history for a user.

        Note: This is a simplified implementation. In production,
        you'd retrieve actual thread messages from OpenAI.
        """
        # TODO: Implement actual conversation history retrieval
        # For now, return empty list
        logger.warning("get_conversation_history not fully implemented")
        return []

    async def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        health_results = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        # Check pipeline health
        try:
            pipeline_health = await self.pipeline.health_check()
            health_results["components"]["pipeline"] = pipeline_health
            if pipeline_health["status"] != "healthy":
                health_results["status"] = "degraded"
        except Exception as e:
            health_results["components"]["pipeline"] = {
                "status": "error",
                "error": str(e)
            }
            health_results["status"] = "unhealthy"

        # Check assistant manager
        try:
            assistant_health = await self.assistant_manager.health_check()
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
        validation = await self.assistant_manager.validate_all_assistants()
        health_results["summary"] = {
            "available_assistants": [
                t for t, c in validation["assistants"].items()
                if c.get("has_id", False)
            ],
            "missing_assistants": [
                t for t, c in validation["assistants"].items()
                if not c.get("has_id", False)
            ],
        }

        return health_results

    async def get_analytics(self, user_id: str, recent_queries: List[str]) -> Dict[str, Any]:
        """
        Get analytics for user queries.

        This is a simplified version. In production, you'd use a proper
        analytics service or database.
        """
        # Filter metrics for this user
        user_metrics = [m for m in self._query_metrics if m["user_id"] == user_id]

        # Calculate statistics
        if not user_metrics:
            return {
                "user_id": user_id,
                "intent_analytics": {},
                "assistant_usage": {},
                "performance_metrics": {
                    "total_queries": 0,
                    "avg_response_time": 0,
                    "success_rate": 0
                },
                "total_queries": 0,
            }

        # Intent distribution
        intent_counts = {}
        assistant_counts = {}
        success_count = 0
        total_response_time = 0

        for metric in user_metrics:
            # Count intents
            intent = metric.get("intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

            # Count assistants
            assistant = metric.get("assistant", "unknown")
            assistant_counts[assistant] = assistant_counts.get(assistant, 0) + 1

            # Track success and response time
            if metric.get("success", False):
                success_count += 1
            total_response_time += metric.get("response_time", 0)

        total_queries = len(user_metrics)

        return {
            "user_id": user_id,
            "intent_analytics": {
                "distribution": intent_counts,
                "total_classified": sum(intent_counts.values())
            },
            "assistant_usage": assistant_counts,
            "performance_metrics": {
                "total_queries": total_queries,
                "avg_response_time": total_response_time / total_queries if total_queries > 0 else 0,
                "success_rate": (success_count / total_queries * 100) if total_queries > 0 else 0
            },
            "total_queries": total_queries,
        }

    def _track_query_metrics(
            self,
            user_id: str,
            query: str,
            response: Dict[str, Any],
            response_time: float
    ):
        """Track query metrics for analytics."""
        metric = {
            "user_id": user_id,
            "query": query[:100],  # Truncate for storage
            "intent": response.get("classification", {}).get("intent", "unknown"),
            "assistant": response.get("classification", {}).get("assistant_used", "unknown"),
            "success": response.get("success", False),
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        }

        self._query_metrics.append(metric)

        # Keep only last 1000 metrics in memory (simple cleanup)
        if len(self._query_metrics) > 1000:
            self._query_metrics = self._query_metrics[-1000:]

    @staticmethod
    def _create_error_response(
            error: Exception, query: str, user_id: str
    ) -> Dict[str, Any]:
        """Create standardized error response as fallback."""
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
                "rerouted": False
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
