"""
Conversation Analytics Service for OpenAI Assistants.
Analyzes conversation patterns, user satisfaction, and interaction quality.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class ConversationAnalyticsService:
    """
    Service for analyzing conversation patterns and user interactions with OpenAI Assistants.
    Focuses on conversation quality, user satisfaction, and interaction effectiveness.
    """

    def __init__(self):
        self.satisfaction_keywords = {
            "positive": [
                "thank",
                "thanks",
                "great",
                "perfect",
                "excellent",
                "good",
                "helpful",
                "awesome",
            ],
            "negative": [
                "wrong",
                "error",
                "bad",
                "terrible",
                "awful",
                "useless",
                "broken",
                "frustrated",
            ],
        }

    def analyze_conversation_quality(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze the quality of conversations with OpenAI Assistants.

        Args:
            conversations: List of conversation data

        Returns:
            Comprehensive conversation quality analysis
        """
        logger.info(f"Analyzing quality for {len(conversations)} conversations")

        # Basic conversation metrics
        total_conversations = len(conversations)
        completed_conversations = [
            c for c in conversations if c.get("completed", False)
        ]
        abandoned_conversations = [
            c for c in conversations if not c.get("completed", False)
        ]

        # Conversation length analysis
        lengths = [len(c.get("messages", [])) for c in conversations]
        avg_length = statistics.mean(lengths) if lengths else 0

        # Resolution analysis
        resolved_conversations = [c for c in conversations if c.get("resolved", False)]
        resolution_rate = (
            len(resolved_conversations) / total_conversations
            if total_conversations > 0
            else 0
        )

        # User satisfaction analysis
        satisfaction_analysis = self._analyze_user_satisfaction(conversations)

        # Intent resolution analysis
        intent_analysis = self._analyze_intent_resolution(conversations)

        # Assistant handoff analysis
        handoff_analysis = self._analyze_assistant_handoffs(conversations)

        return {
            "total_conversations": total_conversations,
            "completion_rate": (
                len(completed_conversations) / total_conversations
                if total_conversations > 0
                else 0
            ),
            "abandonment_rate": (
                len(abandoned_conversations) / total_conversations
                if total_conversations > 0
                else 0
            ),
            "avg_conversation_length": avg_length,
            "resolution_rate": resolution_rate,
            "satisfaction_analysis": satisfaction_analysis,
            "intent_analysis": intent_analysis,
            "handoff_analysis": handoff_analysis,
            "quality_score": self._calculate_quality_score(
                (
                    len(completed_conversations) / total_conversations
                    if total_conversations > 0
                    else 0
                ),
                resolution_rate,
                satisfaction_analysis.get("avg_satisfaction", 0),
            ),
        }

    def analyze_user_engagement_patterns(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze user engagement patterns and behavior.

        Args:
            conversations: List of conversation data

        Returns:
            User engagement analysis
        """
        logger.info(
            f"Analyzing engagement patterns for {len(conversations)} conversations"
        )

        # User activity patterns
        user_activity = defaultdict(list)
        for conv in conversations:
            user_id = conv.get("user_id")
            if user_id:
                user_activity[user_id].append(conv)

        # Engagement metrics
        total_users = len(user_activity)
        conversations_per_user = [len(convs) for convs in user_activity.values()]
        avg_conversations_per_user = (
            statistics.mean(conversations_per_user) if conversations_per_user else 0
        )

        # Time-based analysis
        time_analysis = self._analyze_conversation_timing(conversations)

        # Topic analysis
        topic_analysis = self._analyze_conversation_topics(conversations)

        # User journey analysis
        journey_analysis = self._analyze_user_journeys(user_activity)

        return {
            "total_users": total_users,
            "avg_conversations_per_user": avg_conversations_per_user,
            "user_distribution": self._categorize_users_by_activity(
                conversations_per_user
            ),
            "time_patterns": time_analysis,
            "topic_patterns": topic_analysis,
            "user_journeys": journey_analysis,
            "engagement_score": self._calculate_engagement_score(user_activity),
        }

    def identify_conversation_pain_points(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Identify common pain points and issues in conversations.

        Args:
            conversations: List of conversation data

        Returns:
            Pain point analysis and recommendations
        """
        logger.info(f"Identifying pain points in {len(conversations)} conversations")

        # Failed conversations analysis
        failed_conversations = [c for c in conversations if not c.get("success", True)]

        # Common failure reasons
        failure_reasons = defaultdict(int)
        for conv in failed_conversations:
            reason = conv.get("failure_reason", "unknown")
            failure_reasons[reason] += 1

        # Long conversations (potential complexity issues)
        long_conversations = [
            c for c in conversations if len(c.get("messages", [])) > 10
        ]

        # Conversations with multiple handoffs
        multi_handoff = [c for c in conversations if c.get("handoff_count", 0) > 2]

        # User frustration indicators
        frustration_indicators = self._detect_user_frustration(conversations)

        # Assistant confusion patterns
        confusion_patterns = self._detect_assistant_confusion(conversations)

        return {
            "failed_conversations": {
                "count": len(failed_conversations),
                "percentage": (
                    len(failed_conversations) / len(conversations) * 100
                    if conversations
                    else 0
                ),
                "common_reasons": dict(failure_reasons),
            },
            "complexity_issues": {
                "long_conversations": len(long_conversations),
                "multi_handoff_conversations": len(multi_handoff),
                "avg_handoffs": statistics.mean(
                    [c.get("handoff_count", 0) for c in conversations]
                ),
            },
            "user_frustration": frustration_indicators,
            "assistant_confusion": confusion_patterns,
            "recommendations": self._generate_pain_point_recommendations(
                failure_reasons, frustration_indicators, confusion_patterns
            ),
        }

    def _analyze_user_satisfaction(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user satisfaction from conversation data."""
        satisfaction_scores = []
        sentiment_scores = []

        for conv in conversations:
            # Explicit satisfaction score if available
            if "user_satisfaction" in conv:
                satisfaction_scores.append(conv["user_satisfaction"])

            # Sentiment analysis from messages
            messages = conv.get("messages", [])
            user_messages = [msg for msg in messages if msg.get("role") == "user"]

            for msg in user_messages:
                sentiment = self._analyze_message_sentiment(msg.get("content", ""))
                sentiment_scores.append(sentiment)

        avg_satisfaction = (
            statistics.mean(satisfaction_scores) if satisfaction_scores else 0
        )
        avg_sentiment = statistics.mean(sentiment_scores) if sentiment_scores else 0

        return {
            "avg_satisfaction": avg_satisfaction,
            "avg_sentiment": avg_sentiment,
            "satisfaction_distribution": self._get_satisfaction_distribution(
                satisfaction_scores
            ),
            "sentiment_trend": self._get_sentiment_trend(sentiment_scores),
        }

    @staticmethod
    def _analyze_intent_resolution(
        conversations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze how well different intents are resolved."""
        intent_outcomes = defaultdict(list)

        for conv in conversations:
            intent = conv.get("intent")
            resolved = conv.get("resolved", False)
            success = conv.get("success", False)

            if intent:
                intent_outcomes[intent].append(
                    {
                        "resolved": resolved,
                        "success": success,
                        "handoff_count": conv.get("handoff_count", 0),
                    }
                )

        # Calculate resolution rates by intent
        intent_analysis = {}
        for intent, outcomes in intent_outcomes.items():
            total = len(outcomes)
            resolved_count = sum(1 for o in outcomes if o["resolved"])
            success_count = sum(1 for o in outcomes if o["success"])
            avg_handoffs = statistics.mean([o["handoff_count"] for o in outcomes])

            intent_analysis[intent] = {
                "total_conversations": total,
                "resolution_rate": resolved_count / total if total > 0 else 0,
                "success_rate": success_count / total if total > 0 else 0,
                "avg_handoffs": avg_handoffs,
            }

        return intent_analysis

    def _analyze_assistant_handoffs(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze assistant handoff patterns."""
        handoff_patterns = defaultdict(int)
        total_handoffs = 0

        for conv in conversations:
            handoff_count = conv.get("handoff_count", 0)
            total_handoffs += handoff_count

            # Track handoff paths
            handoff_path = conv.get("handoff_path", [])
            if len(handoff_path) > 1:
                for i in range(len(handoff_path) - 1):
                    path = f"{handoff_path[i]} -> {handoff_path[i+1]}"
                    handoff_patterns[path] += 1

        return {
            "total_handoffs": total_handoffs,
            "avg_handoffs_per_conversation": (
                total_handoffs / len(conversations) if conversations else 0
            ),
            "common_handoff_patterns": dict(handoff_patterns),
            "handoff_success_rate": self._calculate_handoff_success_rate(conversations),
        }

    @staticmethod
    def _analyze_conversation_timing(
        conversations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze timing patterns in conversations."""
        hourly_activity = defaultdict(int)
        daily_activity = defaultdict(int)
        response_times = []

        for conv in conversations:
            # Extract timing information
            start_time_str = conv.get("start_time")
            if start_time_str:
                try:
                    start_time = datetime.fromisoformat(
                        start_time_str.replace("Z", "+00:00")
                    )
                    hour = start_time.hour
                    day = start_time.strftime("%A")

                    hourly_activity[hour] += 1
                    daily_activity[day] += 1
                except (ValueError, TypeError):
                    continue

            # Response time analysis
            messages = conv.get("messages", [])
            for i in range(len(messages) - 1):
                if (
                    messages[i].get("role") == "user"
                    and messages[i + 1].get("role") == "assistant"
                ):
                    # Calculate response time if timestamps available
                    user_time_str = messages[i].get("timestamp")
                    assistant_time_str = messages[i + 1].get("timestamp")

                    if user_time_str and assistant_time_str:
                        try:
                            user_time = datetime.fromisoformat(
                                user_time_str.replace("Z", "+00:00")
                            )
                            assistant_time = datetime.fromisoformat(
                                assistant_time_str.replace("Z", "+00:00")
                            )
                            response_time = (assistant_time - user_time).total_seconds()
                            response_times.append(response_time)
                        except (ValueError, TypeError):
                            continue

        return {
            "hourly_activity": dict(hourly_activity),
            "daily_activity": dict(daily_activity),
            "peak_hour": (
                max(hourly_activity.keys(), key=hourly_activity.get)
                if hourly_activity
                else None
            ),
            "peak_day": (
                max(daily_activity.keys(), key=daily_activity.get)
                if daily_activity
                else None
            ),
            "avg_response_time": (
                statistics.mean(response_times) if response_times else 0
            ),
            "median_response_time": (
                statistics.median(response_times) if response_times else 0
            ),
        }

    @staticmethod
    def _analyze_conversation_topics(
        conversations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Analyze conversation topics and patterns."""
        topics = defaultdict(int)
        intents = defaultdict(int)

        for conv in conversations:
            # Intent tracking
            intent = conv.get("intent")
            if intent:
                intents[intent] += 1

            # Simple topic extraction from messages
            messages = conv.get("messages", [])
            user_messages = [msg for msg in messages if msg.get("role") == "user"]

            for msg in user_messages:
                content = msg.get("content", "").lower()

                # Simple keyword-based topic detection
                if any(
                    word in content for word in ["transaction", "spending", "expense"]
                ):
                    topics["transactions"] += 1
                elif any(word in content for word in ["account", "balance", "bank"]):
                    topics["accounts"] += 1
                elif any(word in content for word in ["invoice", "payment", "bill"]):
                    topics["invoices"] += 1
                elif any(word in content for word in ["forecast", "predict", "future"]):
                    topics["forecasting"] += 1
                elif any(word in content for word in ["budget", "limit", "plan"]):
                    topics["budgeting"] += 1

        return {
            "topic_distribution": dict(topics),
            "intent_distribution": dict(intents),
            "most_common_topic": max(topics.keys(), key=topics.get) if topics else None,
            "most_common_intent": (
                max(intents.keys(), key=intents.get) if intents else None
            ),
        }

    def _analyze_user_journeys(
        self, user_activity: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Analyze user journey patterns."""
        journey_lengths = []
        return_users = 0

        for user_id, conversations in user_activity.items():
            journey_length = len(conversations)
            journey_lengths.append(journey_length)

            if journey_length > 1:
                return_users += 1

        return {
            "avg_journey_length": (
                statistics.mean(journey_lengths) if journey_lengths else 0
            ),
            "return_user_rate": (
                return_users / len(user_activity) * 100 if user_activity else 0
            ),
            "journey_distribution": self._get_journey_distribution(journey_lengths),
        }

    def _analyze_message_sentiment(self, message: str) -> float:
        """Simple sentiment analysis of a message."""
        message_lower = message.lower()

        positive_count = sum(
            1
            for word in self.satisfaction_keywords["positive"]
            if word in message_lower
        )
        negative_count = sum(
            1
            for word in self.satisfaction_keywords["negative"]
            if word in message_lower
        )

        if positive_count + negative_count == 0:
            return 0.0  # Neutral

        return (positive_count - negative_count) / (positive_count + negative_count)

    def _detect_user_frustration(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect indicators of user frustration."""
        frustration_indicators = {
            "repeated_questions": 0,
            "negative_sentiment": 0,
            "abandonment_after_failure": 0,
            "escalation_requests": 0,
        }

        for conv in conversations:
            messages = conv.get("messages", [])
            user_messages = [msg for msg in messages if msg.get("role") == "user"]

            # Check for repeated questions
            if len(user_messages) > 1:
                for i in range(1, len(user_messages)):
                    if self._messages_similar(user_messages[i - 1], user_messages[i]):
                        frustration_indicators["repeated_questions"] += 1
                        break

            # Check for negative sentiment
            for msg in user_messages:
                sentiment = self._analyze_message_sentiment(msg.get("content", ""))
                if sentiment < -0.3:
                    frustration_indicators["negative_sentiment"] += 1
                    break

            # Check for abandonment after failure
            if not conv.get("completed", False) and conv.get("had_errors", False):
                frustration_indicators["abandonment_after_failure"] += 1

            # Check for escalation requests
            for msg in user_messages:
                content = msg.get("content", "").lower()
                if any(
                    phrase in content
                    for phrase in ["human", "person", "support", "help me", "escalate"]
                ):
                    frustration_indicators["escalation_requests"] += 1
                    break

        return frustration_indicators

    @staticmethod
    def _detect_assistant_confusion(
        conversations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect patterns indicating assistant confusion."""
        confusion_patterns = {
            "multiple_reroutes": 0,
            "function_call_failures": 0,
            "unclear_responses": 0,
            "contradictory_responses": 0,
        }

        for conv in conversations:
            # Multiple reroutes indicate confusion
            if conv.get("handoff_count", 0) > 2:
                confusion_patterns["multiple_reroutes"] += 1

            # Function call failures
            function_calls = conv.get("function_calls", [])
            failed_calls = [fc for fc in function_calls if not fc.get("success", False)]
            if len(failed_calls) > 2:
                confusion_patterns["function_call_failures"] += 1

            # Check assistant messages for confusion indicators
            messages = conv.get("messages", [])
            assistant_messages = [
                msg for msg in messages if msg.get("role") == "assistant"
            ]

            for msg in assistant_messages:
                content = msg.get("content", "").lower()

                # Unclear responses
                if any(
                    phrase in content
                    for phrase in [
                        "not sure",
                        "unclear",
                        "confused",
                        "don't understand",
                    ]
                ):
                    confusion_patterns["unclear_responses"] += 1
                    break

        return confusion_patterns

    @staticmethod
    def _calculate_quality_score(
        completion_rate: float, resolution_rate: float, satisfaction: float
    ) -> float:
        """Calculate overall conversation quality score."""
        satisfaction = 0 if satisfaction is None else satisfaction
        # Weight the different factors
        score = (
            (completion_rate * 0.3)
            + (resolution_rate * 0.4)
            + (max(0, satisfaction) * 0.3)
        )
        return min(1.0, score)

    @staticmethod
    def _calculate_engagement_score(
        user_activity: Dict[str, List[Dict[str, Any]]],
    ) -> float:
        """Calculate user engagement score."""
        if not user_activity:
            return 0.0

        # Factors: return rate, conversation depth, session frequency
        return_users = sum(1 for convs in user_activity.values() if len(convs) > 1)
        return_rate = return_users / len(user_activity)

        avg_conversations = statistics.mean(
            [len(convs) for convs in user_activity.values()]
        )
        conversation_depth = min(1.0, avg_conversations / 5)  # Normalize to 0-1

        engagement_score = (return_rate * 0.6) + (conversation_depth * 0.4)
        return min(1.0, engagement_score)

    @staticmethod
    def _messages_similar(msg1: Dict[str, Any], msg2: Dict[str, Any]) -> bool:
        """Check if two messages are similar (indicating repetition)."""
        content1 = msg1.get("content", "").lower().strip()
        content2 = msg2.get("content", "").lower().strip()

        if not content1 or not content2:
            return False

        # Simple similarity check
        words1 = set(content1.split())
        words2 = set(content2.split())

        if len(words1) == 0 or len(words2) == 0:
            return False

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        similarity = intersection / union if union > 0 else 0
        return similarity > 0.7  # 70% word overlap

    @staticmethod
    def _get_satisfaction_distribution(scores: List[float]) -> Dict[str, int]:
        """Get distribution of satisfaction scores."""
        if not scores:
            return {}

        distribution = {"very_low": 0, "low": 0, "medium": 0, "high": 0, "very_high": 0}

        for score in scores:
            if score < 1:
                distribution["very_low"] += 1
            elif score < 2:
                distribution["low"] += 1
            elif score < 3:
                distribution["medium"] += 1
            elif score < 4:
                distribution["high"] += 1
            else:
                distribution["very_high"] += 1

        return distribution

    @staticmethod
    def _get_sentiment_trend(scores: List[float]) -> str:
        """Determine sentiment trend over time."""
        if len(scores) < 3:
            return "insufficient_data"

        # Compare first third to last third
        first_third = scores[: len(scores) // 3]
        last_third = scores[-len(scores) // 3 :]

        first_avg = statistics.mean(first_third)
        last_avg = statistics.mean(last_third)

        if last_avg > first_avg + 0.1:
            return "improving"
        elif last_avg < first_avg - 0.1:
            return "declining"
        else:
            return "stable"

    @staticmethod
    def _calculate_handoff_success_rate(conversations: List[Dict[str, Any]]) -> float:
        """Calculate success rate of conversations with handoffs."""
        handoff_conversations = [
            c for c in conversations if c.get("handoff_count", 0) > 0
        ]

        if not handoff_conversations:
            return 0.0

        successful_handoffs = sum(
            1 for c in handoff_conversations if c.get("success", False)
        )
        return successful_handoffs / len(handoff_conversations)

    @staticmethod
    def _categorize_users_by_activity(
        conversations_per_user: List[int],
    ) -> Dict[str, int]:
        """Categorize users by their activity level."""
        if not conversations_per_user:
            return {}

        categories = {"one_time": 0, "occasional": 0, "regular": 0, "power_user": 0}

        for count in conversations_per_user:
            if count == 1:
                categories["one_time"] += 1
            elif count <= 3:
                categories["occasional"] += 1
            elif count <= 10:
                categories["regular"] += 1
            else:
                categories["power_user"] += 1

        return categories

    @staticmethod
    def _get_journey_distribution(journey_lengths: List[int]) -> Dict[str, int]:
        """Get distribution of user journey lengths."""
        if not journey_lengths:
            return {}

        distribution = {
            "single_session": 0,
            "short_journey": 0,
            "medium_journey": 0,
            "long_journey": 0,
        }

        for length in journey_lengths:
            if length == 1:
                distribution["single_session"] += 1
            elif length <= 3:
                distribution["short_journey"] += 1
            elif length <= 7:
                distribution["medium_journey"] += 1
            else:
                distribution["long_journey"] += 1

        return distribution

    @staticmethod
    def _generate_pain_point_recommendations(
        failure_reasons: Dict[str, int],
        frustration_indicators: Dict[str, int],
        confusion_patterns: Dict[str, int],
    ) -> List[str]:
        """Generate recommendations to address pain points."""
        recommendations = []

        # Address common failure reasons
        for reason, count in failure_reasons.items():
            if count > 5:
                if reason == "intent_classification_error":
                    recommendations.append(
                        "Improve intent classification model with more training data"
                    )
                elif reason == "function_call_failure":
                    recommendations.append(
                        "Review and improve function call reliability"
                    )
                elif reason == "assistant_timeout":
                    recommendations.append(
                        "Optimize assistant response times and add timeout handling"
                    )
                else:
                    recommendations.append(
                        f"Address common failure: {reason} ({count} occurrences)"
                    )

        # Address user frustration
        if frustration_indicators.get("repeated_questions", 0) > 10:
            recommendations.append("Improve assistant memory and context understanding")

        if frustration_indicators.get("negative_sentiment", 0) > 20:
            recommendations.append("Review conversation flow and add empathy responses")

        if frustration_indicators.get("escalation_requests", 0) > 5:
            recommendations.append("Add human handoff capabilities for complex issues")

        # Address assistant confusion
        if confusion_patterns.get("multiple_reroutes", 0) > 15:
            recommendations.append(
                "Improve initial intent classification to reduce rerouting"
            )

        if confusion_patterns.get("function_call_failures", 0) > 10:
            recommendations.append(
                "Add better error handling and retry logic for function calls"
            )

        return recommendations
