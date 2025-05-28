"""
Assistant Performance Service for OpenAI Assistants.
Monitors and analyzes OpenAI Assistant performance and effectiveness.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class AssistantPerformanceService:
    """
    Service for monitoring and analyzing OpenAI Assistant performance.
    Tracks metrics like success rates, response times, and function call effectiveness.
    """

    def __init__(self):
        self.performance_cache = {}
        self.cache_ttl = timedelta(minutes=15)

    def analyze_assistant_performance(
        self, assistant_logs: List[Dict[str, Any]], time_window: Optional[str] = "24h"
    ) -> Dict[str, Any]:
        """
        Analyze performance across all OpenAI Assistants.

        Args:
            assistant_logs: List of assistant interaction logs
            time_window: Time window for analysis (24h, 7d, 30d)

        Returns:
            Comprehensive performance analysis
        """
        logger.info(
            f"Analyzing performance for {len(assistant_logs)} assistant interactions"
        )

        # Filter logs by time window
        filtered_logs = self._filter_logs_by_time(assistant_logs, time_window)

        # Group by assistant type
        assistant_groups = defaultdict(list)
        for log in filtered_logs:
            assistant_type = log.get("assistant_type", "unknown")
            assistant_groups[assistant_type].append(log)

        # Analyze each assistant
        assistant_analysis = {}
        for assistant_type, logs in assistant_groups.items():
            assistant_analysis[assistant_type] = self._analyze_single_assistant(
                assistant_type, logs
            )

        # Generate overall metrics
        overall_metrics = self._calculate_overall_metrics(filtered_logs)

        # Identify performance trends
        trends = self._identify_performance_trends(filtered_logs)

        return {
            "time_window": time_window,
            "total_interactions": len(filtered_logs),
            "assistant_analysis": assistant_analysis,
            "overall_metrics": overall_metrics,
            "performance_trends": trends,
            "recommendations": self._generate_performance_recommendations(
                assistant_analysis
            ),
        }

    def track_function_call_performance(
        self, function_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the performance of function calls across assistants.

        Args:
            function_logs: List of function call logs

        Returns:
            Function call performance analysis
        """
        logger.info(f"Analyzing {len(function_logs)} function calls")

        # Group by function name
        function_groups = defaultdict(list)
        for log in function_logs:
            function_name = log.get("function_name", "unknown")
            function_groups[function_name].append(log)

        # Analyze each function
        function_analysis = {}
        for function_name, logs in function_groups.items():
            function_analysis[function_name] = self._analyze_function_performance(
                function_name, logs
            )

        # Find problematic functions
        problematic_functions = self._identify_problematic_functions(function_analysis)

        return {
            "total_function_calls": len(function_logs),
            "function_analysis": function_analysis,
            "problematic_functions": problematic_functions,
            "optimization_opportunities": self._identify_optimization_opportunities(
                function_analysis
            ),
        }

    def monitor_routing_effectiveness(
        self, routing_logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Monitor the effectiveness of assistant routing decisions.

        Args:
            routing_logs: List of routing decision logs

        Returns:
            Routing effectiveness analysis
        """
        logger.info(
            f"Analyzing routing effectiveness for {len(routing_logs)} interactions"
        )

        # Analyze direct routing vs rerouting
        direct_routes = [
            log for log in routing_logs if not log.get("was_rerouted", False)
        ]
        rerouted = [log for log in routing_logs if log.get("was_rerouted", False)]

        # Calculate success rates
        direct_success_rate = self._calculate_success_rate(direct_routes)
        reroute_success_rate = self._calculate_success_rate(rerouted)

        # Analyze routing patterns
        routing_patterns = self._analyze_routing_patterns(routing_logs)

        # Identify routing inefficiencies
        inefficiencies = self._identify_routing_inefficiencies(routing_logs)

        return {
            "total_interactions": len(routing_logs),
            "direct_routes": {
                "count": len(direct_routes),
                "success_rate": direct_success_rate,
            },
            "rerouted_interactions": {
                "count": len(rerouted),
                "success_rate": reroute_success_rate,
            },
            "routing_patterns": routing_patterns,
            "inefficiencies": inefficiencies,
            "routing_recommendations": self._generate_routing_recommendations(
                routing_patterns, inefficiencies
            ),
        }

    @staticmethod
    def _filter_logs_by_time(
        logs: List[Dict[str, Any]], time_window: str
    ) -> List[Dict[str, Any]]:
        """Filter logs by time window."""
        if time_window == "24h":
            cutoff = datetime.now() - timedelta(hours=24)
        elif time_window == "7d":
            cutoff = datetime.now() - timedelta(days=7)
        elif time_window == "30d":
            cutoff = datetime.now() - timedelta(days=30)
        else:
            return logs  # Return all if unknown window

        filtered = []
        for log in logs:
            timestamp_str = log.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    if timestamp >= cutoff:
                        filtered.append(log)
                except (ValueError, TypeError):
                    # Include logs with invalid timestamps
                    filtered.append(log)
            else:
                # Include logs without timestamps
                filtered.append(log)

        return filtered

    def _analyze_single_assistant(
        self, assistant_type: str, logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance metrics for a single assistant."""
        total_interactions = len(logs)
        successful = [log for log in logs if log.get("success", False)]
        failed = [log for log in logs if not log.get("success", False)]

        # Calculate basic metrics
        success_rate = (
            len(successful) / total_interactions if total_interactions > 0 else 0
        )

        # Response time analysis
        response_times = [
            log.get("response_time", 0) for log in logs if log.get("response_time")
        ]
        avg_response_time = statistics.mean(response_times) if response_times else 0
        median_response_time = (
            statistics.median(response_times) if response_times else 0
        )

        # Function call analysis
        all_function_calls = []
        for log in logs:
            all_function_calls.extend(log.get("function_calls", []))

        function_success_rate = 0
        if all_function_calls:
            successful_functions = sum(
                1 for fc in all_function_calls if fc.get("success", False)
            )
            function_success_rate = successful_functions / len(all_function_calls)

        # Error analysis
        error_types = defaultdict(int)
        for log in failed:
            error_type = log.get("error_type", "unknown")
            error_types[error_type] += 1

        # User satisfaction indicators (if available)
        satisfaction_scores = [
            log.get("user_satisfaction", 0)
            for log in logs
            if log.get("user_satisfaction")
        ]
        avg_satisfaction = (
            statistics.mean(satisfaction_scores) if satisfaction_scores else None
        )

        return {
            "total_interactions": total_interactions,
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "median_response_time": median_response_time,
            "function_call_success_rate": function_success_rate,
            "total_function_calls": len(all_function_calls),
            "error_distribution": dict(error_types),
            "avg_user_satisfaction": avg_satisfaction,
            "performance_grade": self._calculate_performance_grade(
                success_rate, avg_response_time, function_success_rate
            ),
        }

    def _analyze_function_performance(
        self, function_name: str, logs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze performance for a specific function."""
        total_calls = len(logs)
        successful = sum(1 for log in logs if log.get("success", False))

        # Execution time analysis
        execution_times = [
            log.get("execution_time", 0) for log in logs if log.get("execution_time")
        ]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0

        # Error analysis
        errors = [log for log in logs if not log.get("success", False)]
        error_types = defaultdict(int)
        for error in errors:
            error_type = error.get("error_type", "unknown")
            error_types[error_type] += 1

        # Parameter analysis
        parameter_patterns = self._analyze_parameter_patterns(logs)

        return {
            "total_calls": total_calls,
            "success_rate": successful / total_calls if total_calls > 0 else 0,
            "avg_execution_time": avg_execution_time,
            "error_distribution": dict(error_types),
            "parameter_patterns": parameter_patterns,
            "reliability_score": self._calculate_function_reliability(logs),
        }

    def _calculate_overall_metrics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall system metrics."""
        total_interactions = len(logs)
        successful = sum(1 for log in logs if log.get("success", False))

        # Response time statistics
        response_times = [
            log.get("response_time", 0) for log in logs if log.get("response_time")
        ]

        # Assistant usage distribution
        assistant_usage = defaultdict(int)
        for log in logs:
            assistant_type = log.get("assistant_type", "unknown")
            assistant_usage[assistant_type] += 1

        return {
            "overall_success_rate": (
                successful / total_interactions if total_interactions > 0 else 0
            ),
            "avg_response_time": (
                statistics.mean(response_times) if response_times else 0
            ),
            "p95_response_time": (
                statistics.quantiles(response_times, n=20)[18]
                if len(response_times) > 20
                else 0
            ),
            "assistant_usage_distribution": dict(assistant_usage),
            "system_health_score": self._calculate_system_health_score(logs),
        }

    @staticmethod
    def _identify_performance_trends(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify performance trends over time."""
        # Group logs by time periods
        hourly_metrics = defaultdict(list)

        for log in logs:
            timestamp_str = log.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    hour_key = timestamp.strftime("%Y-%m-%d-%H")
                    hourly_metrics[hour_key].append(log)
                except (ValueError, TypeError):
                    continue

        # Calculate hourly success rates
        hourly_success_rates = {}
        for hour, hour_logs in hourly_metrics.items():
            successful = sum(1 for log in hour_logs if log.get("success", False))
            hourly_success_rates[hour] = successful / len(hour_logs)

        # Identify trends
        success_rates = list(hourly_success_rates.values())
        if len(success_rates) >= 3:
            recent_avg = statistics.mean(success_rates[-3:])
            earlier_avg = (
                statistics.mean(success_rates[:-3])
                if len(success_rates) > 3
                else recent_avg
            )
            trend = (
                "improving"
                if recent_avg > earlier_avg
                else "declining" if recent_avg < earlier_avg else "stable"
            )
        else:
            trend = "insufficient_data"

        return {
            "hourly_success_rates": hourly_success_rates,
            "trend": trend,
            "peak_performance_hour": (
                max(hourly_success_rates.keys(), key=hourly_success_rates.get)
                if hourly_success_rates
                else None
            ),
            "lowest_performance_hour": (
                min(hourly_success_rates.keys(), key=hourly_success_rates.get)
                if hourly_success_rates
                else None
            ),
        }

    @staticmethod
    def _calculate_success_rate(logs: List[Dict[str, Any]]) -> float:
        """Calculate success rate for a list of logs."""
        if not logs:
            return 0.0
        successful = sum(1 for log in logs if log.get("success", False))
        return successful / len(logs)

    @staticmethod
    def _analyze_routing_patterns(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze routing patterns in the logs."""
        routing_paths = defaultdict(int)
        intent_to_assistant = defaultdict(list)

        for log in logs:
            initial_assistant = log.get("initial_assistant")
            final_assistant = log.get("final_assistant")
            intent = log.get("intent")

            if initial_assistant and final_assistant:
                if initial_assistant != final_assistant:
                    path = f"{initial_assistant} -> {final_assistant}"
                    routing_paths[path] += 1

            if intent and final_assistant:
                intent_to_assistant[intent].append(final_assistant)

        # Find most common assistant for each intent
        intent_preferences = {}
        for intent, assistants in intent_to_assistant.items():
            assistant_counts = defaultdict(int)
            for assistant in assistants:
                assistant_counts[assistant] += 1

            if assistant_counts:
                preferred = max(assistant_counts.keys(), key=assistant_counts.get)
                confidence = assistant_counts[preferred] / len(assistants)
                intent_preferences[intent] = {
                    "preferred_assistant": preferred,
                    "confidence": confidence,
                }

        return {
            "common_routing_paths": dict(routing_paths),
            "intent_assistant_preferences": intent_preferences,
        }

    @staticmethod
    def _identify_problematic_functions(
        function_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Identify functions with performance issues."""
        problematic = []

        for function_name, analysis in function_analysis.items():
            issues = []

            if analysis["success_rate"] < 0.8:
                issues.append(f"Low success rate: {analysis['success_rate']:.2%}")

            if analysis["avg_execution_time"] > 5.0:  # More than 5 seconds
                issues.append(f"Slow execution: {analysis['avg_execution_time']:.2f}s")

            if analysis["reliability_score"] < 0.7:
                issues.append(f"Low reliability: {analysis['reliability_score']:.2%}")

            if issues:
                problematic.append(
                    {
                        "function_name": function_name,
                        "issues": issues,
                        "total_calls": analysis["total_calls"],
                        "priority": "high" if len(issues) > 2 else "medium",
                    }
                )

        return sorted(problematic, key=lambda x: len(x["issues"]), reverse=True)

    @staticmethod
    def _identify_optimization_opportunities(
        function_analysis: Dict[str, Any],
    ) -> List[str]:
        """Identify optimization opportunities for functions."""
        opportunities = []

        # Find heavily used functions with room for improvement
        for function_name, analysis in function_analysis.items():
            if analysis["total_calls"] > 100:  # Heavily used
                if analysis["success_rate"] < 0.95:
                    opportunities.append(
                        f"Optimize '{function_name}' - high usage ({analysis['total_calls']} calls) "
                        f"but {analysis['success_rate']:.2%} success rate"
                    )

                if analysis["avg_execution_time"] > 2.0:
                    opportunities.append(
                        f"Speed up '{function_name}' - {analysis['avg_execution_time']:.2f}s average execution time"
                    )

        return opportunities

    @staticmethod
    def _identify_routing_inefficiencies(
        logs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Identify routing inefficiencies."""
        inefficiencies = []

        # Count rerouting patterns
        reroute_patterns = defaultdict(int)
        for log in logs:
            if log.get("was_rerouted", False):
                initial = log.get("initial_assistant", "unknown")
                final = log.get("final_assistant", "unknown")
                intent = log.get("intent", "unknown")

                pattern = f"{intent}: {initial} -> {final}"
                reroute_patterns[pattern] += 1

        # Identify frequently rerouted patterns
        for pattern, count in reroute_patterns.items():
            if count > 5:  # More than 5 occurrences
                inefficiencies.append(
                    {
                        "pattern": pattern,
                        "occurrences": count,
                        "suggestion": f"Consider routing {pattern.split(':')[0]} intent directly to final assistant",
                    }
                )

        return inefficiencies

    @staticmethod
    def _calculate_performance_grade(
        success_rate: float, response_time: float, function_success_rate: float
    ) -> str:
        """Calculate a performance grade for an assistant."""
        score = 0

        # Success rate (40% weight)
        if success_rate >= 0.95:
            score += 40
        elif success_rate >= 0.9:
            score += 35
        elif success_rate >= 0.8:
            score += 30
        elif success_rate >= 0.7:
            score += 20
        else:
            score += 10

        # Response time (30% weight)
        if response_time <= 1.0:
            score += 30
        elif response_time <= 2.0:
            score += 25
        elif response_time <= 3.0:
            score += 20
        elif response_time <= 5.0:
            score += 15
        else:
            score += 5

        # Function success rate (30% weight)
        if function_success_rate >= 0.95:
            score += 30
        elif function_success_rate >= 0.9:
            score += 25
        elif function_success_rate >= 0.8:
            score += 20
        elif function_success_rate >= 0.7:
            score += 15
        else:
            score += 5

        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    @staticmethod
    def _calculate_function_reliability(logs: List[Dict[str, Any]]) -> float:
        """Calculate reliability score for a function."""
        if not logs:
            return 0.0

        # Weight recent calls more heavily
        now = datetime.now()
        weighted_success = 0
        total_weight = 0

        for log in logs:
            timestamp_str = log.get("timestamp")
            success = log.get("success", False)

            # Calculate weight based on recency (more recent = higher weight)
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                    days_ago = (now - timestamp).days
                    weight = max(
                        0.1, 1.0 - (days_ago / 30)
                    )  # Linear decay over 30 days
                except (ValueError, TypeError):
                    weight = 1.0
            else:
                weight = 1.0

            weighted_success += weight if success else 0
            total_weight += weight

        return weighted_success / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _calculate_system_health_score(logs: List[Dict[str, Any]]) -> float:
        """Calculate overall system health score."""
        if not logs:
            return 0.0

        # Multiple factors contribute to health score
        success_rate = sum(1 for log in logs if log.get("success", False)) / len(logs)

        # Response time factor
        response_times = [
            log.get("response_time", 0) for log in logs if log.get("response_time")
        ]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0
        )
        response_factor = max(float(0), float(1 - (avg_response_time / 10)))

        # Error diversity factor (fewer error types = better)
        error_types = set()
        for log in logs:
            if not log.get("success", False):
                error_types.add(log.get("error_type", "unknown"))

        error_diversity_factor = max(float(0), float(1 - (len(error_types) / 10)))

        # Combine factors
        health_score = (
            success_rate * 0.6 + response_factor * 0.2 + error_diversity_factor * 0.2
        )

        return min(1.0, health_score)

    @staticmethod
    def _analyze_parameter_patterns(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze parameter usage patterns for a function."""
        parameter_usage = defaultdict(int)
        parameter_success = defaultdict(list)

        for log in logs:
            parameters = log.get("parameters", {})
            success = log.get("success", False)

            for param_name, param_value in parameters.items():
                parameter_usage[param_name] += 1
                parameter_success[param_name].append(success)

        # Calculate success rates per parameter
        parameter_analysis = {}
        for param_name, successes in parameter_success.items():
            if successes:
                success_rate = sum(successes) / len(successes)
                parameter_analysis[param_name] = {
                    "usage_count": parameter_usage[param_name],
                    "success_rate": success_rate,
                }

        return parameter_analysis

    @staticmethod
    def _generate_performance_recommendations(
        assistant_analysis: Dict[str, Any],
    ) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        for assistant_type, analysis in assistant_analysis.items():
            grade = analysis.get("performance_grade", "F")
            success_rate = analysis.get("success_rate", 0)
            response_time = analysis.get("avg_response_time", 0)

            if grade in ["D", "F"]:
                recommendations.append(
                    f"URGENT: {assistant_type} assistant needs immediate attention - Grade {grade}"
                )

            if success_rate < 0.8:
                recommendations.append(
                    f"Improve {assistant_type} assistant reliability - {success_rate:.2%} success rate"
                )

            if response_time > 3.0:
                recommendations.append(
                    f"Optimize {assistant_type} assistant response time - {response_time:.2f}s average"
                )

            if analysis.get("function_call_success_rate", 1) < 0.9:
                recommendations.append(
                    f"Debug function calls for {assistant_type} assistant - low success rate"
                )

        return recommendations

    @staticmethod
    def _generate_routing_recommendations(
        patterns: Dict[str, Any], inefficiencies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate routing improvement recommendations."""
        recommendations = []

        # Recommendations based on inefficiencies
        for inefficiency in inefficiencies:
            recommendations.append(inefficiency["suggestion"])

        # Recommendations based on patterns
        intent_prefs = patterns.get("intent_assistant_preferences", {})
        for intent, pref_data in intent_prefs.items():
            if pref_data["confidence"] > 0.8:
                recommendations.append(
                    f"Intent '{intent}' strongly prefers {pref_data['preferred_assistant']} assistant "
                    f"({pref_data['confidence']:.2%} confidence) - consider direct routing"
                )

        return recommendations
