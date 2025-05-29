"""
Intelligent routing decisions based on historical performance.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

from services.search.vector_search import SearchResult

logger = logging.getLogger(__name__)


class RoutingIntelligence:
    """Makes intelligent routing decisions based on historical data."""

    def __init__(self):
        self.min_success_rate = 0.7
        self.min_sample_size = 3

    def recommend_assistant(
        self,
        intent: str,
        search_results: List[SearchResult],
        user_preferences: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Recommend the best assistant for the intent.

        Args:
            intent: Determined intent
            search_results: Historical similar conversations
            user_preferences: User-specific preferences

        Returns:
            Routing recommendation with reasoning
        """
        # Analyze assistant performance
        assistant_performance = self._analyze_assistant_performance(
            search_results, intent
        )

        # Apply user preferences
        if user_preferences:
            assistant_performance = self._apply_user_preferences(
                assistant_performance, user_preferences
            )

        # Select best assistant
        recommendation = self._select_best_assistant(assistant_performance, intent)

        return recommendation

    @staticmethod
    def _analyze_assistant_performance(
        search_results: List[SearchResult], target_intent: str
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze performance of each assistant."""
        performance = defaultdict(
            lambda: {
                "success_count": 0,
                "total_count": 0,
                "avg_similarity": 0.0,
                "similarities": [],
            }
        )

        for result in search_results:
            # Only consider results matching target intent
            if result.intent != target_intent:
                continue

            assistant = result.assistant_type
            performance[assistant]["total_count"] += 1
            performance[assistant]["similarities"].append(result.similarity_score)

            if result.success_indicator:
                performance[assistant]["success_count"] += 1

        # Calculate success rates and averages
        for assistant, data in performance.items():
            if data["total_count"] > 0:
                data["success_rate"] = data["success_count"] / data["total_count"]
                data["avg_similarity"] = sum(data["similarities"]) / len(
                    data["similarities"]
                )
            else:
                data["success_rate"] = 0.0
                data["avg_similarity"] = 0.0

            # Remove similarities list to clean up
            del data["similarities"]

        return dict(performance)

    @staticmethod
    def _apply_user_preferences(
        performance: Dict[str, Dict[str, Any]], preferences: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Apply user-specific preferences to performance data."""
        # Example: User might prefer certain assistants
        preferred_assistants = preferences.get("preferred_assistants", [])

        for assistant in preferred_assistants:
            if assistant in performance:
                # Boost the success rate slightly for preferred assistants
                performance[assistant]["preference_boost"] = 0.1

        return performance

    def _select_best_assistant(
        self, performance: Dict[str, Dict[str, Any]], intent: str
    ) -> Dict[str, Any]:
        """Select the best assistant based on performance."""
        # Default assistant mapping
        default_assistants = {
            "transactions": "transaction_assistant",
            "invoices": "invoice_assistant",
            "forecasting": "forecasting_assistant",
            "budgets": "budget_assistant",
            "accounts": "account_assistant",
            "general": "general_assistant",
        }

        if not performance:
            # No historical data, use default
            return {
                "assistant": default_assistants.get(intent, "general_assistant"),
                "confidence": 0.5,
                "reasoning": "No historical data available",
                "method": "default_mapping",
            }

        # Score each assistant
        scored_assistants = []
        for assistant, perf in performance.items():
            # Calculate score
            score = self._calculate_assistant_score(perf)

            scored_assistants.append(
                {
                    "assistant": assistant,
                    "score": score,
                    "success_rate": perf["success_rate"],
                    "sample_size": perf["total_count"],
                }
            )

        # Sort by score
        scored_assistants.sort(key=lambda x: x["score"], reverse=True)

        # Select best with sufficient data
        for candidate in scored_assistants:
            if (
                candidate["sample_size"] >= self.min_sample_size
                and candidate["success_rate"] >= self.min_success_rate
            ):
                return {
                    "assistant": candidate["assistant"],
                    "confidence": candidate["score"],
                    "success_rate": candidate["success_rate"],
                    "sample_size": candidate["sample_size"],
                    "reasoning": (
                        f"Historical success rate: {candidate['success_rate']:.1%} "
                        f"from {candidate['sample_size']} similar conversations"
                    ),
                    "method": "performance_based",
                }

        # Fallback to highest score regardless of thresholds
        if scored_assistants:
            best = scored_assistants[0]
            return {
                "assistant": best["assistant"],
                "confidence": best["score"] * 0.8,  # Reduce confidence
                "success_rate": best["success_rate"],
                "sample_size": best["sample_size"],
                "reasoning": "Best available option (below optimal thresholds)",
                "method": "best_available",
            }

        # Ultimate fallback
        return {
            "assistant": default_assistants.get(intent, "general_assistant"),
            "confidence": 0.4,
            "reasoning": "Using default mapping",
            "method": "fallback",
        }

    @staticmethod
    def _calculate_assistant_score(performance: Dict[str, Any]) -> float:
        """Calculate overall score for an assistant."""
        success_rate = performance["success_rate"]
        sample_size = performance["total_count"]
        avg_similarity = performance["avg_similarity"]
        preference_boost = performance.get("preference_boost", 0.0)

        # Weight factors
        success_weight = 0.5
        similarity_weight = 0.3
        sample_weight = 0.2

        # Calculate sample size factor (0 to 1)
        sample_factor = min(sample_size / 10, 1.0)

        # Calculate base score
        score = (
            success_rate * success_weight
            + avg_similarity * similarity_weight
            + sample_factor * sample_weight
        )

        # Apply preference boost
        score = min(score + preference_boost, 1.0)

        return score
