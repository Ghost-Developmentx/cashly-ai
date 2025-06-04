"""
Analytics functionality for the integration service.
Tracks usage patterns and performance metrics.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class IntegrationAnalytics:
    """
    Handles tracking and generation of analytics for user queries.

    This class is designed to manage the tracking of user queries, classify intents,
    and generate analytics. It provides methods for tracking queries, computing intent
    distribution, calculating assistant usage, and assessing performance metrics. The
    data collected is cached for optimization and efficiency purposes.

    Attributes
    ----------
    intent_service : Any
        Service to classify intents and route queries.
    intent_mapper : Any
        Mapper to resolve default assistants for specific intents.
    _query_cache : defaultdict
        In-memory cache to store tracked queries per user.
    """

    def __init__(self, intent_service, intent_mapper):
        self.intent_service = intent_service
        self.intent_mapper = intent_mapper
        self._query_cache = defaultdict(list)

    def track_query(
        self,
        user_id: str,
        query: str,
        intent: str,
        assistant_used: str,
        success: bool,
        response_time: float,
    ):
        """Track a query for analytics."""
        self._query_cache[user_id].append(
            {
                "query": query,
                "intent": intent,
                "assistant": assistant_used,
                "success": success,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep only the last 100 queries per user in memory
        if len(self._query_cache[user_id]) > 100:
            self._query_cache[user_id] = self._query_cache[user_id][-100:]

    def get_analytics(self, user_id: str, recent_queries: List[str]) -> Dict[str, Any]:
        """
        Generate analytics for user queries.

        Args:
            user_id: User identifier
            recent_queries: List of recent query texts

        Returns:
            Analytics data
        """
        try:
            # Get intent analytics
            intent_analytics = self._analyze_intents(recent_queries)

            # Calculate assistant usage
            assistant_usage = self._calculate_assistant_usage(recent_queries, user_id)

            # Get performance metrics
            performance_metrics = self._get_performance_metrics(user_id)

            return {
                "intent_analytics": intent_analytics,
                "assistant_usage": assistant_usage,
                "performance_metrics": performance_metrics,
                "total_queries": len(recent_queries),
                "user_id": user_id,
                "generated_at": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error generating analytics: {e}")
            return {
                "error": str(e),
                "user_id": user_id,
                "total_queries": len(recent_queries),
            }

    async def _analyze_intents(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze intent distribution in queries."""
        if not queries:
            return {"message": "No queries to analyze"}

        intent_counts = defaultdict(int)
        confidence_sum = defaultdict(float)

        for query in queries:
            try:
                result = await self.intent_service.classify_and_route(query)
                intent = result["classification"]["intent"]
                confidence = result["classification"]["confidence"]

                intent_counts[intent] += 1
                confidence_sum[intent] += confidence
            except Exception as e:
                logger.warning(f"Failed to classify query: {e}")

        # Calculate averages
        intent_stats = {}
        for intent, count in intent_counts.items():
            intent_stats[intent] = {
                "count": count,
                "percentage": (count / len(queries)) * 100,
                "avg_confidence": confidence_sum[intent] / count,
            }

        return {
            "intent_distribution": intent_stats,
            "total_classified": sum(intent_counts.values()),
            "unique_intents": len(intent_counts),
        }

    async def _calculate_assistant_usage(
        self, queries: List[str], user_id: str
    ) -> Dict[str, int]:
        """Calculate assistant usage statistics."""
        assistant_usage = defaultdict(int)

        # From recent queries
        for query in queries:
            try:
                routing = await self.intent_service.classify_and_route(query)
                intent = routing["classification"]["intent"]
                assistant = self.intent_mapper.get_default_assistant(intent)
                assistant_usage[assistant.value] += 1
            except Exception as e:
                logger.warning(f"Failed to route query: {e}")

        # From tracked queries
        if user_id in self._query_cache:
            for tracked_query in self._query_cache[user_id]:
                assistant = tracked_query.get("assistant")
                if assistant:
                    assistant_usage[assistant] += 1

        return dict(assistant_usage)

    def _get_performance_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get performance metrics for a user."""
        if user_id not in self._query_cache:
            return {"message": "No performance data available"}

        queries = self._query_cache[user_id]
        if not queries:
            return {"message": "No queries tracked"}

        # Calculate metrics
        total_queries = len(queries)
        successful_queries = sum(1 for q in queries if q.get("success", False))
        response_times = [
            q.get("response_time", 0) for q in queries if "response_time" in q
        ]

        metrics = {
            "total_queries": total_queries,
            "success_rate": (
                (successful_queries / total_queries * 100) if total_queries > 0 else 0
            ),
            "avg_response_time": (
                sum(response_times) / len(response_times) if response_times else 0
            ),
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
        }

        return metrics
