"""
Intent Learning Service for OpenAI Assistants.
Handles learning from user interactions to improve intent classification.
"""

import logging
from typing import Dict, List, Any
from collections import defaultdict

from services.intent_classification.intent_service import IntentService
from services.intent_classification.conversation_data_processor import (
    ConversationDataProcessor,
)

logger = logging.getLogger(__name__)


class IntentLearningService:
    """
    Service for learning and improving intent classification from user interactions.
    Focused on OpenAI Assistant routing and intent understanding.
    """

    def __init__(self):
        self.intent_service = IntentService()
        self.data_processor = ConversationDataProcessor()

    def learn_from_conversations(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Learn from successful and unsuccessful conversation patterns.

        Args:
            conversations: List of conversation data with success/failure indicators

        Returns:
            Learning results and improvements made
        """
        logger.info(
            f"Processing {len(conversations)} conversations for intent learning"
        )

        # Separate successful and unsuccessful conversations
        successful = [c for c in conversations if c.get("success", False)]
        unsuccessful = [c for c in conversations if not c.get("success", False)]

        logger.info(
            f"Found {len(successful)} successful, {len(unsuccessful)} unsuccessful conversations"
        )

        # Extract intent patterns from successful conversations
        successful_patterns = self._extract_intent_patterns(successful)

        # Analyze failure patterns to improve routing
        failure_analysis = self._analyze_routing_failures(unsuccessful)

        # Update an intent classification model if we have enough data
        training_results = None
        if len(successful) >= 10:
            training_results = self._retrain_intent_classifier(successful)

        return {
            "successful_patterns": successful_patterns,
            "failure_analysis": failure_analysis,
            "training_results": training_results,
            "recommendations": self._generate_improvement_recommendations(
                successful_patterns, failure_analysis
            ),
        }

    def learn_from_assistant_interactions(
        self, assistant_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Learn from OpenAI Assistant interactions and function calls.

        Args:
            assistant_logs: List of assistant interaction logs

        Returns:
            Analysis of assistant performance and routing effectiveness
        """
        logger.info(f"Analyzing {len(assistant_logs)} assistant interactions")

        # Group by assistant type
        assistant_performance = defaultdict(list)

        for log in assistant_logs:
            assistant_type = log.get("assistant_type", "unknown")
            assistant_performance[assistant_type].append(log)

        # Analyze each assistant's performance
        performance_analysis = {}

        for assistant_type, logs in assistant_performance.items():
            analysis = self._analyze_assistant_performance(assistant_type, logs)
            performance_analysis[assistant_type] = analysis

        # Identify cross-assistant routing patterns
        routing_patterns = self._analyze_routing_patterns(assistant_logs)

        return {
            "assistant_performance": performance_analysis,
            "routing_patterns": routing_patterns,
            "optimization_suggestions": self._generate_routing_optimizations(
                performance_analysis, routing_patterns
            ),
        }

    def _extract_intent_patterns(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract successful intent patterns from conversations."""
        patterns = defaultdict(list)

        for conv in conversations:
            messages = conv.get("messages", [])
            final_assistant = conv.get("final_assistant")
            intent = conv.get("intent")

            # Extract user queries that led to successful outcomes
            user_queries = [
                msg.get("content", "") for msg in messages if msg.get("role") == "user"
            ]

            if user_queries and intent and final_assistant:
                patterns[intent].extend(
                    [
                        {
                            "query": query,
                            "assistant": final_assistant,
                            "success_metrics": conv.get("success_metrics", {}),
                        }
                        for query in user_queries
                    ]
                )

        # Summarize patterns
        pattern_summary = {}
        for intent, examples in patterns.items():
            pattern_summary[intent] = {
                "count": len(examples),
                "common_phrases": self._extract_common_phrases(examples),
                "preferred_assistant": self._get_most_common_assistant(examples),
            }

        return pattern_summary

    def _analyze_routing_failures(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze failed conversations to identify routing issues."""
        failure_types = defaultdict(list)

        for conv in conversations:
            failure_reason = conv.get("failure_reason", "unknown")
            initial_intent = conv.get("initial_intent")
            final_assistant = conv.get("final_assistant")

            failure_types[failure_reason].append(
                {
                    "initial_intent": initial_intent,
                    "final_assistant": final_assistant,
                    "user_query": self._extract_first_user_query(conv),
                    "reroute_count": conv.get("reroute_count", 0),
                }
            )

        # Analyze common failure patterns
        analysis = {}
        for failure_type, examples in failure_types.items():
            analysis[failure_type] = {
                "count": len(examples),
                "common_intents": self._get_common_values(examples, "initial_intent"),
                "problematic_assistants": self._get_common_values(
                    examples, "final_assistant"
                ),
                "avg_reroutes": sum(e.get("reroute_count", 0) for e in examples)
                / len(examples),
            }

        return analysis

    def _retrain_intent_classifier(
        self, successful_conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Retrain the intent classifier with successful conversation examples."""
        try:
            # Convert conversations to training format
            training_data = self.data_processor.extract_from_fin_conversations(
                successful_conversations
            )

            # Train the intent classifier
            results = self.intent_service.train_with_new_data(
                conversations=successful_conversations,
                tool_usage_stats={},  # Could be extracted from conversations if needed
            )

            logger.info(
                f"Intent classifier retrained with {len(training_data)} examples"
            )
            return results

        except Exception as e:
            logger.error(f"Failed to retrain intent classifier: {e}")
            return {"status": "failed", "error": str(e)}

    def _analyze_assistant_performance(
        self, assistant_type: str, logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance metrics for a specific assistant."""
        total_interactions = len(logs)
        successful_interactions = sum(1 for log in logs if log.get("success", False))

        # Calculate function call success rates
        function_calls = []
        for log in logs:
            function_calls.extend(log.get("function_calls", []))

        successful_functions = sum(
            1 for fc in function_calls if fc.get("success", False)
        )
        total_functions = len(function_calls)

        # Calculate average response time
        response_times = [
            log.get("response_time", 0) for log in logs if log.get("response_time")
        ]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )

        # Identify common failure reasons
        failures = [log for log in logs if not log.get("success", False)]
        failure_reasons = defaultdict(int)
        for failure in failures:
            reason = failure.get("failure_reason", "unknown")
            failure_reasons[reason] += 1

        return {
            "total_interactions": total_interactions,
            "success_rate": (
                successful_interactions / total_interactions
                if total_interactions > 0
                else 0
            ),
            "function_success_rate": (
                successful_functions / total_functions if total_functions > 0 else 0
            ),
            "avg_response_time": avg_response_time,
            "common_failures": dict(failure_reasons),
            "recommendations": self._generate_assistant_recommendations(
                assistant_type, logs
            ),
        }

    @staticmethod
    def _analyze_routing_patterns(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze routing patterns across assistant interactions."""
        routing_data = []

        for log in logs:
            initial_assistant = log.get("initial_assistant")
            final_assistant = log.get("final_assistant")
            reroute_count = log.get("reroute_count", 0)
            success = log.get("success", False)

            if initial_assistant and final_assistant:
                routing_data.append(
                    {
                        "initial": initial_assistant,
                        "final": final_assistant,
                        "rerouted": initial_assistant != final_assistant,
                        "reroute_count": reroute_count,
                        "success": success,
                    }
                )

        # Calculate routing statistics
        total_queries = len(routing_data)
        rerouted_queries = sum(1 for r in routing_data if r["rerouted"])
        successful_reroutes = sum(
            1 for r in routing_data if r["rerouted"] and r["success"]
        )

        # Find the most common routing paths
        routing_paths = defaultdict(int)
        for r in routing_data:
            if r["rerouted"]:
                path = f"{r['initial']} -> {r['final']}"
                routing_paths[path] += 1

        return {
            "total_queries": total_queries,
            "reroute_rate": (
                rerouted_queries / total_queries if total_queries > 0 else 0
            ),
            "reroute_success_rate": (
                successful_reroutes / rerouted_queries if rerouted_queries > 0 else 0
            ),
            "common_routing_paths": dict(routing_paths),
            "avg_reroutes_per_query": (
                sum(r["reroute_count"] for r in routing_data) / total_queries
                if total_queries > 0
                else 0
            ),
        }

    @staticmethod
    def _generate_improvement_recommendations(
        patterns: Dict[str, Any], failures: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving intent classification and routing."""
        recommendations = []

        # Recommendations based on successful patterns
        for intent, pattern_data in patterns.items():
            if pattern_data["count"] >= 10:
                recommendations.append(
                    f"Intent '{intent}' has strong patterns - consider adding more training examples"
                )

        # Recommendations based on failure analysis
        for failure_type, failure_data in failures.items():
            if failure_data["count"] >= 5:
                if failure_data["avg_reroutes"] > 2:
                    recommendations.append(
                        f"High reroute rate for {failure_type} failures - review initial intent classification"
                    )

        return recommendations

    @staticmethod
    def _generate_routing_optimizations(
        performance: Dict[str, Any], routing: Dict[str, Any]
    ) -> List[str]:
        """Generate routing optimization suggestions."""
        optimizations = []

        # Check for assistants with low success rates
        for assistant, perf in performance.items():
            if perf["success_rate"] < 0.7:
                optimizations.append(
                    f"Assistant '{assistant}' has low success rate ({perf['success_rate']:.2%}) - review capabilities"
                )

        # Check for high reroute rates
        if routing["reroute_rate"] > 0.3:
            optimizations.append(
                f"High reroute rate ({routing['reroute_rate']:.2%}) - improve initial intent classification"
            )

        # Check for common routing paths that might indicate misrouting
        for path, count in routing["common_routing_paths"].items():
            if count > 10:
                optimizations.append(
                    f"Common rerouting path '{path}' ({count} times) - consider direct routing"
                )

        return optimizations

    @staticmethod
    def _extract_common_phrases(examples: List[Dict[str, Any]]) -> List[str]:
        """Extract common phrases from successful examples."""
        # Simple implementation - could be enhanced with NLP
        phrases = []
        for example in examples:
            query = example.get("query", "").lower()
            if len(query.split()) <= 5:  # Short phrases are likely important
                phrases.append(query)

        # Return most common phrases
        phrase_counts = defaultdict(int)
        for phrase in phrases:
            phrase_counts[phrase] += 1

        return sorted(phrase_counts.keys(), key=phrase_counts.get, reverse=True)[:5]

    @staticmethod
    def _get_most_common_assistant(examples: List[Dict[str, Any]]) -> str:
        """Get the most commonly used assistant for successful examples."""
        assistant_counts = defaultdict(int)
        for example in examples:
            assistant = example.get("assistant", "unknown")
            assistant_counts[assistant] += 1

        return (
            max(assistant_counts.keys(), key=assistant_counts.get)
            if assistant_counts
            else "unknown"
        )

    @staticmethod
    def _extract_first_user_query(conversation: Dict[str, Any]) -> str:
        """Extract the first user query from a conversation."""
        messages = conversation.get("messages", [])
        for msg in messages:
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    @staticmethod
    def _get_common_values(examples: List[Dict[str, Any]], key: str) -> List[str]:
        """Get most common values for a specific key in examples."""
        value_counts = defaultdict(int)
        for example in examples:
            value = example.get(key)
            if value:
                value_counts[value] += 1

        return sorted(value_counts.keys(), key=value_counts.get, reverse=True)[:3]

    @staticmethod
    def _generate_assistant_recommendations(
        assistant_type: str, logs: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate specific recommendations for an assistant."""
        recommendations = []

        # Check for common failure patterns
        failures = [log for log in logs if not log.get("success", False)]
        if len(failures) > len(logs) * 0.3:  # More than 30% failure rate
            recommendations.append(
                f"High failure rate for {assistant_type} - review assistant instructions"
            )

        # Check for function call issues
        function_failures = []
        for log in logs:
            for fc in log.get("function_calls", []):
                if not fc.get("success", False):
                    function_failures.append(fc.get("function_name", "unknown"))

        if function_failures:
            common_failures = defaultdict(int)
            for func in function_failures:
                common_failures[func] += 1

            most_common = max(common_failures.keys(), key=common_failures.get)
            recommendations.append(
                f"Function '{most_common}' frequently fails - review implementation"
            )

        return recommendations
