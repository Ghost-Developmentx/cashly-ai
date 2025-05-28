"""
Refactored Fin Learning Service for OpenAI Assistants.
Orchestrates multiple specialized learning services for comprehensive AI improvement.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from services.learning.intent_learning_service import IntentLearningService
from services.learning.assistant_performance_service import AssistantPerformanceService
from services.learning.conversation_analytics_service import (
    ConversationAnalyticsService,
)

logger = logging.getLogger(__name__)


class FinLearningService:
    """
    Refactored learning service that orchestrates specialized learning components.
    Focused on OpenAI Assistant performance improvement and user experience optimization.
    """

    def __init__(self):
        self.intent_learning = IntentLearningService()
        self.performance_service = AssistantPerformanceService()
        self.conversation_analytics = ConversationAnalyticsService()

    def process_learning_dataset(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a comprehensive learning dataset to improve OpenAI Assistant performance.

        Args:
            dataset: Dictionary containing various types of learning data:
                - conversations: List of conversation data
                - assistant_logs: List of assistant interaction logs
                - function_calls: List of function call logs
                - user_feedback: List of user feedback data

        Returns:
            Comprehensive learning results and improvement recommendations
        """
        logger.info("Processing comprehensive learning dataset for OpenAI Assistants")

        # Extract different data types
        conversations = dataset.get("conversations", [])
        helpful_conversations = dataset.get("helpful_conversations", [])
        unhelpful_conversations = dataset.get("unhelpful_conversations", [])
        assistant_logs = dataset.get("assistant_logs", [])
        function_calls = dataset.get("function_calls", [])
        user_feedback = dataset.get("user_feedback", [])

        # Combine helpful and unhelpful conversations
        all_conversations = (
            helpful_conversations + unhelpful_conversations + conversations
        )

        # Mark conversation success/failure
        for conv in helpful_conversations:
            conv["success"] = True
        for conv in unhelpful_conversations:
            conv["success"] = False

        logger.info(
            f"Processing {len(all_conversations)} conversations, {len(assistant_logs)} assistant logs"
        )

        # Process different aspects of learning
        results = {
            "processing_timestamp": datetime.now().isoformat(),
            "dataset_summary": self._summarize_dataset(dataset),
            "learning_results": {},
        }

        # Intent and routing learning
        if all_conversations:
            logger.info("Analyzing intent and routing patterns...")
            intent_results = self.intent_learning.learn_from_conversations(
                all_conversations
            )
            results["learning_results"]["intent_learning"] = intent_results

            if assistant_logs:
                assistant_interaction_results = (
                    self.intent_learning.learn_from_assistant_interactions(
                        assistant_logs
                    )
                )
                results["learning_results"][
                    "assistant_interactions"
                ] = assistant_interaction_results

        # Performance analysis
        if assistant_logs:
            logger.info("Analyzing assistant performance...")
            performance_results = (
                self.performance_service.analyze_assistant_performance(assistant_logs)
            )
            results["learning_results"]["performance_analysis"] = performance_results

        if function_calls:
            logger.info("Analyzing function call performance...")
            function_results = self.performance_service.track_function_call_performance(
                function_calls
            )
            results["learning_results"]["function_performance"] = function_results

        # Conversation quality analysis
        if all_conversations:
            logger.info("Analyzing conversation quality...")
            quality_results = self.conversation_analytics.analyze_conversation_quality(
                all_conversations
            )
            results["learning_results"]["conversation_quality"] = quality_results

            engagement_results = (
                self.conversation_analytics.analyze_user_engagement_patterns(
                    all_conversations
                )
            )
            results["learning_results"]["user_engagement"] = engagement_results

            pain_points = self.conversation_analytics.identify_conversation_pain_points(
                all_conversations
            )
            results["learning_results"]["pain_points"] = pain_points

        # Generate comprehensive recommendations
        recommendations = self._generate_comprehensive_recommendations(
            results["learning_results"]
        )
        results["recommendations"] = recommendations

        # Calculate overall improvement score
        improvement_score = self._calculate_improvement_potential(
            results["learning_results"]
        )
        results["improvement_potential"] = improvement_score

        logger.info(
            f"Learning analysis complete. Generated {len(recommendations)} recommendations"
        )

        return results

    def analyze_assistant_effectiveness(
        self, assistant_logs: List[Dict[str, Any]], time_window: str = "7d"
    ) -> Dict[str, Any]:
        """
        Focused analysis of OpenAI Assistant effectiveness.

        Args:
            assistant_logs: List of assistant interaction logs
            time_window: Time window for analysis

        Returns:
            Assistant effectiveness analysis
        """
        logger.info(f"Analyzing assistant effectiveness over {time_window}")

        # Performance analysis
        performance_results = self.performance_service.analyze_assistant_performance(
            assistant_logs, time_window
        )

        # Routing effectiveness
        routing_results = self.performance_service.monitor_routing_effectiveness(
            assistant_logs
        )

        # Extract conversations from assistant logs for quality analysis
        conversations = []
        for log in assistant_logs:
            if "conversation" in log:
                conversations.append(log["conversation"])

        quality_results = None
        if conversations:
            quality_results = self.conversation_analytics.analyze_conversation_quality(
                conversations
            )

        return {
            "time_window": time_window,
            "performance_analysis": performance_results,
            "routing_effectiveness": routing_results,
            "conversation_quality": quality_results,
            "overall_effectiveness_score": self._calculate_effectiveness_score(
                performance_results, routing_results, quality_results
            ),
        }

    def generate_improvement_plan(
        self, analysis_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a prioritized improvement plan based on analysis results.

        Args:
            analysis_results: Results from learning analysis

        Returns:
            Structured improvement plan with priorities and timelines
        """
        logger.info("Generating improvement plan from analysis results")

        # Extract all recommendations
        all_recommendations = []

        # Collect recommendations from different analysis components
        learning_results = analysis_results.get("learning_results", {})

        for component, results in learning_results.items():
            if isinstance(results, dict) and "recommendations" in results:
                component_recs = results["recommendations"]
                if isinstance(component_recs, list):
                    for rec in component_recs:
                        all_recommendations.append(
                            {
                                "recommendation": rec,
                                "component": component,
                                "priority": self._assess_recommendation_priority(
                                    rec, component
                                ),
                            }
                        )

        # Prioritize recommendations
        high_priority = [r for r in all_recommendations if r["priority"] == "high"]
        medium_priority = [r for r in all_recommendations if r["priority"] == "medium"]
        low_priority = [r for r in all_recommendations if r["priority"] == "low"]

        # Create improvement plan
        improvement_plan = {
            "plan_created": datetime.now().isoformat(),
            "total_recommendations": len(all_recommendations),
            "immediate_actions": high_priority[:5],  # Top 5 high priority items
            "short_term_goals": medium_priority[:10],  # Top 10 medium priority items
            "long_term_improvements": low_priority,
            "estimated_timeline": self._estimate_implementation_timeline(
                all_recommendations
            ),
            "success_metrics": self._define_success_metrics(analysis_results),
        }

        return improvement_plan

    @staticmethod
    def _summarize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the learning dataset."""
        summary = {
            "total_conversations": 0,
            "helpful_conversations": len(dataset.get("helpful_conversations", [])),
            "unhelpful_conversations": len(dataset.get("unhelpful_conversations", [])),
            "assistant_logs": len(dataset.get("assistant_logs", [])),
            "function_calls": len(dataset.get("function_calls", [])),
            "user_feedback": len(dataset.get("user_feedback", [])),
        }

        summary["total_conversations"] = (
            summary["helpful_conversations"]
            + summary["unhelpful_conversations"]
            + len(dataset.get("conversations", []))
        )

        return summary

    def _generate_comprehensive_recommendations(
        self, learning_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive recommendations from all learning components."""
        recommendations = []

        # Priority mapping for different types of recommendations
        priority_keywords = {
            "high": ["urgent", "critical", "immediate", "broken", "failing"],
            "medium": ["improve", "optimize", "enhance", "review"],
            "low": ["consider", "suggest", "monitor", "track"],
        }

        # Collect recommendations from all components
        for component, results in learning_results.items():
            if isinstance(results, dict):
                # Direct recommendations
                if "recommendations" in results:
                    component_recs = results["recommendations"]
                    if isinstance(component_recs, list):
                        for rec in component_recs:
                            recommendations.append(
                                {
                                    "recommendation": rec,
                                    "component": component,
                                    "category": self._categorize_recommendation(rec),
                                    "priority": self._assess_recommendation_priority(
                                        rec, component
                                    ),
                                }
                            )

                # Nested recommendations
                for key, value in results.items():
                    if isinstance(value, dict) and "recommendations" in value:
                        nested_recs = value["recommendations"]
                        if isinstance(nested_recs, list):
                            for rec in nested_recs:
                                recommendations.append(
                                    {
                                        "recommendation": rec,
                                        "component": f"{component}.{key}",
                                        "category": self._categorize_recommendation(
                                            rec
                                        ),
                                        "priority": self._assess_recommendation_priority(
                                            rec, component
                                        ),
                                    }
                                )

        # Remove duplicates
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            rec_text = rec["recommendation"].lower()
            if rec_text not in seen:
                seen.add(rec_text)
                unique_recommendations.append(rec)

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        unique_recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return unique_recommendations

    @staticmethod
    def _categorize_recommendation(recommendation: str) -> str:
        """Categorize a recommendation by type."""
        rec_lower = recommendation.lower()

        if any(word in rec_lower for word in ["intent", "classification", "routing"]):
            return "intent_routing"
        elif any(word in rec_lower for word in ["function", "call", "api"]):
            return "function_calls"
        elif any(
            word in rec_lower for word in ["assistant", "performance", "response"]
        ):
            return "assistant_performance"
        elif any(
            word in rec_lower for word in ["conversation", "user", "satisfaction"]
        ):
            return "user_experience"
        elif any(word in rec_lower for word in ["error", "failure", "timeout"]):
            return "error_handling"
        else:
            return "general"

    @staticmethod
    def _assess_recommendation_priority(recommendation: str, component: str) -> str:
        """Assess the priority of a recommendation."""
        rec_lower = recommendation.lower()

        # High priority indicators
        if any(
            word in rec_lower
            for word in ["urgent", "critical", "broken", "failing", "error rate"]
        ):
            return "high"

        # Component-based priority
        if component in ["performance_analysis", "function_performance"]:
            if any(word in rec_lower for word in ["low success", "timeout", "failure"]):
                return "high"

        # Medium priority indicators
        if any(
            word in rec_lower for word in ["improve", "optimize", "review", "update"]
        ):
            return "medium"

        # Default to low priority
        return "low"

    @staticmethod
    def _calculate_improvement_potential(
        learning_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate the potential for improvement based on analysis results."""
        scores = []

        # Performance improvement potential
        if "performance_analysis" in learning_results:
            perf_data = learning_results["performance_analysis"]
            overall_metrics = perf_data.get("overall_metrics", {})
            success_rate = overall_metrics.get("overall_success_rate", 1.0)

            # More room for improvement with lower success rates
            perf_potential = (1.0 - success_rate) * 100
            scores.append(("performance", perf_potential))

        # Conversation quality improvement potential
        if "conversation_quality" in learning_results:
            quality_data = learning_results["conversation_quality"]
            quality_score = quality_data.get("quality_score", 1.0)

            quality_potential = (1.0 - quality_score) * 100
            scores.append(("conversation_quality", quality_potential))

        # User engagement improvement potential
        if "user_engagement" in learning_results:
            engagement_data = learning_results["user_engagement"]
            engagement_score = engagement_data.get("engagement_score", 1.0)

            engagement_potential = (1.0 - engagement_score) * 100
            scores.append(("user_engagement", engagement_potential))

        # Calculate overall improvement potential
        if scores:
            avg_potential = sum(score[1] for score in scores) / len(scores)
            max_potential = max(score[1] for score in scores)

            return {
                "overall_potential": avg_potential,
                "max_potential": max_potential,
                "component_potentials": dict(scores),
                "improvement_level": (
                    "high"
                    if avg_potential > 30
                    else "medium" if avg_potential > 15 else "low"
                ),
            }

        return {"overall_potential": 0, "improvement_level": "unknown"}

    @staticmethod
    def _calculate_effectiveness_score(
        performance_results: Dict[str, Any],
        routing_results: Dict[str, Any],
        quality_results: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate overall effectiveness score."""
        scores = []

        # Performance score
        if performance_results:
            overall_metrics = performance_results.get("overall_metrics", {})
            success_rate = overall_metrics.get("overall_success_rate", 0)
            scores.append(success_rate * 0.4)  # 40% weight

        # Routing effectiveness score
        if routing_results:
            direct_success = routing_results.get("direct_routes", {}).get(
                "success_rate", 0
            )
            reroute_success = routing_results.get("rerouted_interactions", {}).get(
                "success_rate", 0
            )

            # Weight direct routing more heavily
            routing_score = (direct_success * 0.7) + (reroute_success * 0.3)
            scores.append(routing_score * 0.3)  # 30% weight

        # Conversation quality score
        if quality_results:
            quality_score = quality_results.get("quality_score", 0)
            scores.append(quality_score * 0.3)  # 30% weight

        return sum(scores) if scores else 0.0

    @staticmethod
    def _estimate_implementation_timeline(
        recommendations: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Estimate implementation timeline for recommendations."""
        high_priority_count = sum(1 for r in recommendations if r["priority"] == "high")
        medium_priority_count = sum(
            1 for r in recommendations if r["priority"] == "medium"
        )
        low_priority_count = sum(1 for r in recommendations if r["priority"] == "low")

        # Rough estimation based on priority and complexity
        immediate_timeline = f"{high_priority_count * 2}-{high_priority_count * 3} days"
        short_term_timeline = (
            f"{medium_priority_count * 1}-{medium_priority_count * 2} weeks"
        )
        long_term_timeline = f"{low_priority_count * 2}-{low_priority_count * 4} weeks"

        return {
            "immediate_actions": immediate_timeline,
            "short_term_goals": short_term_timeline,
            "long_term_improvements": long_term_timeline,
            "total_estimated_duration": f"{2 + medium_priority_count + low_priority_count * 2}-{4 + medium_priority_count * 2 + low_priority_count * 4} weeks",
        }

    @staticmethod
    def _define_success_metrics(
        analysis_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Define success metrics for tracking improvement."""
        metrics = [
            {
                "metric": "Overall Success Rate",
                "current_value": None,
                "target_improvement": "5-10%",
                "measurement": "percentage of successful interactions",
            },
            {
                "metric": "Average Response Time",
                "current_value": None,
                "target_improvement": "10-20% reduction",
                "measurement": "seconds",
            },
            {
                "metric": "Function Call Success Rate",
                "current_value": None,
                "target_improvement": "5% increase",
                "measurement": "percentage of successful function calls",
            },
            {
                "metric": "User Satisfaction Score",
                "current_value": None,
                "target_improvement": "0.5 point increase",
                "measurement": "1-5 scale",
            },
            {
                "metric": "Conversation Completion Rate",
                "current_value": None,
                "target_improvement": "5% increase",
                "measurement": "percentage of completed conversations",
            },
        ]

        # Populate current values if available
        learning_results = analysis_results.get("learning_results", {})

        if "performance_analysis" in learning_results:
            perf_data = learning_results["performance_analysis"]
            overall_metrics = perf_data.get("overall_metrics", {})

            for metric in metrics:
                if metric["metric"] == "Overall Success Rate":
                    metric["current_value"] = (
                        f"{overall_metrics.get('overall_success_rate', 0) * 100:.1f}%"
                    )
                elif metric["metric"] == "Average Response Time":
                    metric["current_value"] = (
                        f"{overall_metrics.get('avg_response_time', 0):.2f}s"
                    )

        if "conversation_quality" in learning_results:
            quality_data = learning_results["conversation_quality"]

            for metric in metrics:
                if metric["metric"] == "Conversation Completion Rate":
                    metric["current_value"] = (
                        f"{quality_data.get('completion_rate', 0) * 100:.1f}%"
                    )
                elif metric["metric"] == "User Satisfaction Score":
                    satisfaction = quality_data.get("satisfaction_analysis", {}).get(
                        "avg_satisfaction"
                    )
                    if satisfaction is not None:
                        metric["current_value"] = f"{satisfaction:.2f}"

        return metrics
