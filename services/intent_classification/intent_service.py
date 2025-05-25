import logging
from typing import Dict, List, Optional, Any
import os
from services.intent_classification.intent_classifier import IntentClassifier
from services.intent_classification.conversation_data_processor import (
    ConversationDataProcessor,
)

logger = logging.getLogger(__name__)


class IntentService:
    """
    Service layer for intent classification in the Cashly Fin system.
    Integrates with the existing conversation flow and routes to appropriate assistants.
    """

    def __init__(self, model_path: str = "models/intent_classifier"):
        self.classifier = IntentClassifier(model_path)
        self.processor = ConversationDataProcessor()

        # Define assistant routing based on intents
        self.assistant_routing = {
            "transactions": "transaction_assistant",
            "accounts": "account_assistant",
            "invoices": "invoice_assistant",
            "bank_connection": "bank_connection_assistant",
            "payment_processing": "payment_processing_assistant",
            "forecasting": "forecasting_assistant",
            "budgets": "budget_assistant",
            "insights": "insights_assistant",
            "general": "general_assistant",
        }

        # Define confidence thresholds for different actions
        self.confidence_thresholds = {
            "high_confidence": 0.8,  # Route directly to assistant
            "medium_confidence": 0.6,  # Route with fallback options
            "low_confidence": 0.4,  # Use general assistant with context
        }

    def classify_and_route(
        self,
        query: str,
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Classify intent and determine routing strategy.

        Args:
            query: User's query text
            user_context: User context including accounts, preferences, etc.
            conversation_history: Previous conversation messages

        Returns:
            Dictionary with classification results and routing information
        """
        # Get intent classification
        classification = self.classifier.classify_intent(query)

        # Enhance classification with context
        enhanced_classification = self._enhance_with_context(
            classification, user_context, conversation_history
        )

        # Determine routing strategy
        routing_strategy = self._determine_routing_strategy(enhanced_classification)

        # Get fallback options
        fallback_options = self._get_fallback_options(query, enhanced_classification)

        return {
            "classification": enhanced_classification,
            "routing": routing_strategy,
            "fallback_options": fallback_options,
            "should_route": self._should_route_to_specialist(enhanced_classification),
            "recommended_assistant": self.assistant_routing.get(
                enhanced_classification["intent"], "general_assistant"
            ),
        }

    def _enhance_with_context(
        self,
        classification: Dict[str, Any],
        user_context: Optional[Dict] = None,
        conversation_history: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Enhance classification with user context and conversation history."""
        enhanced = classification.copy()

        # Check user context for hints
        if user_context:
            context_boost = self._get_context_boost(
                classification["intent"], user_context
            )
            enhanced["confidence"] = min(1.0, enhanced["confidence"] + context_boost)
            enhanced["context_factors"] = self._analyze_context_factors(
                classification["intent"], user_context
            )

        # Check conversation history for patterns
        if conversation_history:
            history_boost = self._get_history_boost(
                classification["intent"], conversation_history
            )
            enhanced["confidence"] = min(1.0, enhanced["confidence"] + history_boost)
            enhanced["conversation_context"] = self._analyze_conversation_context(
                classification["intent"], conversation_history
            )

        enhanced["enhancement_applied"] = True
        return enhanced

    def _get_context_boost(self, intent: str, user_context: Dict) -> float:
        """Calculate confidence boost based on user context."""
        boost = 0.0

        # Check if user has relevant integrations
        integrations = user_context.get("integrations", [])
        accounts = user_context.get("accounts", [])

        context_relevance = {
            "transactions": lambda: len(accounts) > 0,  # Has connected accounts
            "accounts": lambda: len(accounts) == 0
            or any(
                not acc.get("plaid_account_id") for acc in accounts
            ),  # No accounts or incomplete setup
            "invoices": lambda: any(
                integration.get("provider") == "stripe" for integration in integrations
            ),  # Has Stripe integration
            "forecasting": lambda: len(accounts) > 0
            and len(user_context.get("transactions", [])) > 10,
            "budgets": lambda: len(accounts) > 0,
            "insights": lambda: len(user_context.get("transactions", [])) > 5,
            "general": lambda: True,
        }

        if intent in context_relevance and context_relevance[intent]():
            boost += 0.1

        return boost

    def _get_history_boost(
        self, intent: str, conversation_history: List[Dict]
    ) -> float:
        """Calculate confidence boost based on conversation history."""
        if not conversation_history:
            return 0.0

        # Look at recent messages for intent patterns
        recent_messages = conversation_history[-3:]  # Last 3 messages

        intent_keywords = {
            "transactions": ["transaction", "spending", "expense", "income"],
            "accounts": [
                "account",
                "balance",
                "bank",
            ],
            "bank_connection": ["connect bank", "link bank", "account"],
            "payment_processing": ["process", "pay", "payment", "connect stripe"],
            "invoices": ["invoice", "payment", "bill", "client"],
            "forecasting": ["forecast", "predict", "future"],
            "budgets": ["budget", "limit", "plan"],
            "insights": ["analysis", "trend", "insight"],
        }

        if intent in intent_keywords:
            keywords = intent_keywords[intent]
            matches = 0

            for message in recent_messages:
                content = message.get("content", "").lower()
                matches += sum(1 for keyword in keywords if keyword in content)

            return min(0.2, matches * 0.05)  # Max boost of 0.2

        return 0.0

    def _analyze_context_factors(
        self, intent: str, user_context: Dict
    ) -> Dict[str, Any]:
        """Analyze context factors that influenced the classification."""
        factors = {}

        accounts = user_context.get("accounts", [])
        integrations = user_context.get("integrations", [])
        transactions = user_context.get("transactions", [])

        factors["account_count"] = len(accounts)
        factors["has_plaid_accounts"] = any(
            acc.get("plaid_account_id") for acc in accounts
        )
        factors["has_stripe_integration"] = any(
            integration.get("provider") == "stripe" for integration in integrations
        )
        factors["transaction_count"] = len(transactions)
        factors["has_sufficient_data"] = len(transactions) > 10

        return factors

    def _analyze_conversation_context(
        self, intent: str, conversation_history: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze conversation context."""
        context = {
            "message_count": len(conversation_history),
            "recent_intents": [],
            "conversation_flow": "new",  # new, continuing, follow_up
        }

        # Analyze recent message patterns
        if len(conversation_history) > 1:
            context["conversation_flow"] = "continuing"

            # Check if this is a follow-up question
            last_user_message = None
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    last_user_message = msg.get("content", "")
                    break

            if last_user_message and any(
                word in last_user_message.lower()
                for word in ["also", "and", "what about", "how about", "additionally"]
            ):
                context["conversation_flow"] = "follow_up"

        return context

    def _determine_routing_strategy(
        self, classification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Determine the best routing strategy based on classification."""
        confidence = classification["confidence"]
        intent = classification["intent"]

        if confidence >= self.confidence_thresholds["high_confidence"]:
            strategy = "direct_route"
            explanation = f"High confidence ({confidence:.2%}) - route directly to {intent} assistant"

        elif confidence >= self.confidence_thresholds["medium_confidence"]:
            strategy = "route_with_fallback"
            explanation = f"Medium confidence ({confidence:.2%}) - route to {intent} assistant with fallback"

        elif confidence >= self.confidence_thresholds["low_confidence"]:
            strategy = "general_with_context"
            explanation = f"Low confidence ({confidence:.2%}) - use general assistant with {intent} context"

        else:
            strategy = "general_fallback"
            explanation = (
                f"Very low confidence ({confidence:.2%}) - use general assistant"
            )

        return {
            "strategy": strategy,
            "confidence_level": self._get_confidence_level(confidence),
            "explanation": explanation,
            "primary_assistant": self.assistant_routing.get(
                intent, "general_assistant"
            ),
            "should_provide_options": confidence
            < self.confidence_thresholds["medium_confidence"],
        }

    def _get_confidence_level(self, confidence: float) -> str:
        """Get a human-readable confidence level."""
        if confidence >= self.confidence_thresholds["high_confidence"]:
            return "high"
        elif confidence >= self.confidence_thresholds["medium_confidence"]:
            return "medium"
        elif confidence >= self.confidence_thresholds["low_confidence"]:
            return "low"
        else:
            return "very_low"

    def _get_fallback_options(
        self, query: str, classification: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get fallback routing options."""
        # Get top 3 intent suggestions
        suggestions = self.classifier.get_intent_suggestions(query, top_k=3)

        fallback_options = []
        for suggestion in suggestions[1:]:  # Skip the primary suggestion
            if suggestion["confidence"] > 0.3:  # Only include reasonable alternatives
                fallback_options.append(
                    {
                        "intent": suggestion["intent"],
                        "confidence": suggestion["confidence"],
                        "assistant": self.assistant_routing.get(
                            suggestion["intent"], "general_assistant"
                        ),
                        "explanation": f"Alternative: {suggestion['intent']} ({suggestion['confidence']:.2%})",
                    }
                )

        return fallback_options[:2]  # Max 2 fallback options

    def _should_route_to_specialist(self, classification: Dict[str, Any]) -> bool:
        """Determine if query should be routed to a specialist assistant."""
        confidence = classification["confidence"]
        intent = classification["intent"]

        # Always route high confidence non-general intents
        if (
            confidence >= self.confidence_thresholds["high_confidence"]
            and intent != "general"
        ):
            return True

        # Route medium confidence for specific intents
        if confidence >= self.confidence_thresholds["medium_confidence"] and intent in [
            "transactions",
            "accounts",
            "invoices",
            "forecasting",
        ]:
            return True

        return False

    def train_with_new_data(
        self, conversations: List[Dict], tool_usage_stats: Dict
    ) -> Dict[str, Any]:
        """
        Train the intent classifier with new conversation data.

        Args:
            conversations: List of conversation dictionaries
            tool_usage_stats: Tool usage statistics

        Returns:
            Training results
        """
        logger.info("Training intent classifier with new data...")

        # Process the data
        training_df = self.processor.create_training_dataset(
            conversations=conversations, tool_usage=tool_usage_stats
        )

        if len(training_df) < 10:
            logger.warning("Insufficient training data - using existing model")
            return {
                "status": "skipped",
                "reason": "insufficient_data",
                "samples": len(training_df),
            }

        # Prepare training data for the classifier
        training_conversations = []
        for _, row in training_df.iterrows():
            training_conversations.append(
                {
                    "messages": [{"role": "user", "content": row["user_query"]}],
                    "intent_label": row["intent"],
                }
            )

        # Train the classifier
        results = self.classifier.train_with_conversation_data(training_conversations)

        logger.info(f"Intent classifier training completed: {results}")
        return results

    def get_routing_analytics(self, recent_queries: List[str]) -> Dict[str, Any]:
        """
        Analyze recent queries for routing performance insights.

        Args:
            recent_queries: List of recent user queries

        Returns:
            Analytics about intent distribution and routing patterns
        """
        analytics = {
            "total_queries": len(recent_queries),
            "intent_distribution": {},
            "confidence_distribution": {},
            "routing_strategies": {},
            "low_confidence_queries": [],
        }

        for query in recent_queries:
            result = self.classify_and_route(query)
            classification = result["classification"]
            routing = result["routing"]

            # Track intent distribution
            intent = classification["intent"]
            analytics["intent_distribution"][intent] = (
                analytics["intent_distribution"].get(intent, 0) + 1
            )

            # Track confidence distribution
            confidence_level = routing["confidence_level"]
            analytics["confidence_distribution"][confidence_level] = (
                analytics["confidence_distribution"].get(confidence_level, 0) + 1
            )

            # Track routing strategies
            strategy = routing["strategy"]
            analytics["routing_strategies"][strategy] = (
                analytics["routing_strategies"].get(strategy, 0) + 1
            )

            # Track low-confidence queries for improvement
            if (
                classification["confidence"]
                < self.confidence_thresholds["medium_confidence"]
            ):
                analytics["low_confidence_queries"].append(
                    {
                        "query": query,
                        "intent": intent,
                        "confidence": classification["confidence"],
                    }
                )

        return analytics
