"""
Hybrid intent classification service combining embeddings and keywords.
"""

import logging
from typing import Dict, List, Optional, Any

from services.intent_classification.intent_classifier import IntentClassifier
from services.intent_classification.embedding_intent_classifier import (
    EmbeddingIntentClassifier,
)

logger = logging.getLogger(__name__)


class HybridIntentService:
    """
    Combines embedding-based and keyword-based intent classification.
    """

    def __init__(self):
        self.keyword_classifier = IntentClassifier()
        self.embedding_classifier = EmbeddingIntentClassifier()

        # Strategy thresholds
        self.use_embedding_threshold = 2  # Min messages for embedding classification
        self.confidence_boost_threshold = 0.8  # Min similarity for confidence boost

    def classify_intent(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None,
        user_id: Optional[str] = None,
        user_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Classify intent using hybrid approach.

        Args:
            query: User query
            conversation_history: Previous messages
            user_id: User identifier
            user_context: Additional user context

        Returns:
            Classification result with intent and confidence
        """
        # Start with keyword classification
        keyword_result = self.keyword_classifier.classify_intent(query)

        # If no conversation history or too short, use keyword result
        if (
            not conversation_history
            or len(conversation_history) < self.use_embedding_threshold
        ):
            logger.info("Using keyword-only classification (insufficient history)")
            return keyword_result

        # Try embedding classification
        try:
            embedding_result = self.embedding_classifier.classify_with_context(
                query=query, conversation_history=conversation_history, user_id=user_id
            )

            # Combine results
            final_result = self._combine_results(
                keyword_result, embedding_result, query, user_context
            )

            return final_result

        except Exception as e:
            logger.error(f"Embedding classification failed: {e}")
            return keyword_result

    def _combine_results(
        self,
        keyword_result: Dict[str, Any],
        embedding_result: Dict[str, Any],
        query: str,
        user_context: Optional[Dict],
    ) -> Dict[str, Any]:
        """
        Combine keyword and embedding classification results.

        Args:
            keyword_result: Result from keyword classifier
            embedding_result: Result from embedding classifier
            query: Original query
            user_context: User context

        Returns:
            Combined classification result
        """
        # If embedding has high confidence, use it
        if embedding_result["confidence"] > 0.85:
            logger.info(
                f"Using embedding result (high confidence: {embedding_result['confidence']:.2%})"
            )
            return {
                **embedding_result,
                "method": "embedding_high_confidence",
                "keyword_intent": keyword_result["intent"],
                "keyword_confidence": keyword_result["confidence"],
            }

        # If both agree, boost confidence
        if keyword_result["intent"] == embedding_result["intent"]:
            confidence_boost = 0.1
            combined_confidence = min(
                keyword_result["confidence"] + confidence_boost, 0.95
            )

            logger.info(f"Classifiers agree on intent '{keyword_result['intent']}'")

            return {
                "intent": keyword_result["intent"],
                "confidence": combined_confidence,
                "method": "hybrid_agreement",
                "keyword_confidence": keyword_result["confidence"],
                "embedding_confidence": embedding_result["confidence"],
                "reasoning": f"Both methods agree on {keyword_result['intent']}",
            }

        # If they disagree, use weighted combination
        if keyword_result["confidence"] > embedding_result["confidence"]:
            primary_result = keyword_result
            secondary_result = embedding_result
        else:
            primary_result = embedding_result
            secondary_result = keyword_result

        # Check if user context supports either classification
        if user_context:
            context_adjusted = self._adjust_for_context(
                primary_result, secondary_result, user_context
            )
            if context_adjusted:
                return context_adjusted

        return {
            "intent": primary_result["intent"],
            "confidence": primary_result["confidence"]
            * 0.8,  # Reduce confidence due to disagreement
            "method": "hybrid_weighted",
            "primary_method": primary_result["method"],
            "alternative_intent": secondary_result["intent"],
            "alternative_confidence": secondary_result["confidence"],
            "reasoning": f"Methods disagree: {primary_result['method']} suggests {primary_result['intent']}",
        }

    @staticmethod
    def _adjust_for_context(
        primary_result: Dict[str, Any],
        secondary_result: Dict[str, Any],
        user_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Adjust classification based on user context.

        Args:
            primary_result: Primary classification
            secondary_result: Secondary classification
            user_context: User context

        Returns:
            Adjusted result or None
        """
        # Check if the user has relevant data for the intents
        context_support = {
            "transactions": len(user_context.get("transactions", [])) > 0,
            "accounts": len(user_context.get("accounts", [])) > 0,
            "invoices": user_context.get("stripe_connect", {}).get("connected", False),
            "forecasting": len(user_context.get("transactions", [])) > 10,
            "budgets": len(user_context.get("budgets", [])) > 0,
        }

        # If secondary intent has context support but primary doesn't
        primary_support = context_support.get(primary_result["intent"], True)
        secondary_support = context_support.get(secondary_result["intent"], True)

        if not primary_support and secondary_support:
            return {
                **secondary_result,
                "confidence": secondary_result["confidence"] * 1.1,  # Boost confidence
                "method": "hybrid_context_adjusted",
                "reasoning": f"Adjusted to {secondary_result['intent']} based on user context",
            }

        return None
