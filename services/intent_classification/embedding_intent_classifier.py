"""
Context-aware intent classifier using embeddings and vector similarity.
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from services.embeddings.openai_client import OpenAIEmbeddingClient
from services.embeddings.context_builder import ConversationContextBuilder
from services.embeddings.storage import EmbeddingStorage

logger = logging.getLogger(__name__)


class EmbeddingIntentClassifier:
    """Intent classifier using conversation embeddings."""

    def __init__(
        self,
        embedding_client: Optional[OpenAIEmbeddingClient] = None,
        context_builder: Optional[ConversationContextBuilder] = None,
        storage: Optional[EmbeddingStorage] = None,
    ):
        self.embedding_client = embedding_client or OpenAIEmbeddingClient()
        self.context_builder = context_builder or ConversationContextBuilder()
        self.storage = storage or EmbeddingStorage()

        # Intent confidence thresholds
        self.similarity_threshold = 0.8
        self.high_confidence_threshold = 0.9
        self.medium_confidence_threshold = 0.75

    def classify_with_context(
        self,
        query: str,
        conversation_history: List[Dict],
        user_id: Optional[str] = None,
    ) -> Dict[str, any]:
        """
        Classify intent using conversation context and embeddings.

        Args:
            query: Current user query
            conversation_history: Previous messages
            user_id: User identifier for personalization

        Returns:
            Classification result with intent and confidence
        """
        # Build context from conversation
        context = self.context_builder.build_context(conversation_history, query)

        # Generate embedding for the context
        embedding = self.embedding_client.create_embedding(context)

        if not embedding:
            logger.warning("Failed to create embedding, falling back to keyword method")
            return self._fallback_classification(query)

        # Find similar conversations
        similar_conversations = self.storage.find_similar_conversations(
            embedding=embedding,
            user_id=user_id,
            limit=5,
            similarity_threshold=self.similarity_threshold,
        )

        if not similar_conversations:
            logger.info(
                "No similar conversations found, using query-only classification"
            )
            return self._query_only_classification(query, embedding)

        # Analyze similar conversations
        intent, confidence = self._analyze_similar_conversations(
            similar_conversations, query
        )

        # Get routing recommendations
        routing_recommendation = self._get_routing_recommendation(similar_conversations)

        return {
            "intent": intent,
            "confidence": confidence,
            "method": "embedding_similarity",
            "similar_conversations": len(similar_conversations),
            "routing_recommendation": routing_recommendation,
            "reasoning": self._build_reasoning(
                similar_conversations, intent, confidence
            ),
        }

    @staticmethod
    def _analyze_similar_conversations(
        similar_conversations: List[Dict], query: str
    ) -> Tuple[str, float]:
        """
        Analyze similar conversations to determine intent.

        Args:
            similar_conversations: List of similar conversation records
            query: Current query for additional context

        Returns:
            Tuple of (intent, confidence)
        """
        # Weight intents by similarity and success
        intent_scores = defaultdict(float)
        assistant_scores = defaultdict(float)

        for conv in similar_conversations:
            similarity = conv["similarity"]
            success = conv["success_indicator"]
            intent = conv["intent"]
            assistant = conv["assistant_type"]

            # Weight by similarity and success
            weight = similarity * (1.2 if success else 0.8)

            intent_scores[intent] += weight
            assistant_scores[assistant] += weight

        # Get the highest scoring intent
        if not intent_scores:
            return "general", 0.5

        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        intent = best_intent[0]

        # Calculate confidence based on score distribution
        total_score = sum(intent_scores.values())
        confidence = best_intent[1] / total_score if total_score > 0 else 0.5

        # Adjust confidence based on similarity scores
        avg_similarity = sum(c["similarity"] for c in similar_conversations) / len(
            similar_conversations
        )
        confidence = confidence * avg_similarity

        return intent, min(confidence, 0.95)  # Cap at 95%

    @staticmethod
    def _get_routing_recommendation(
        similar_conversations: List[Dict],
    ) -> Dict[str, any]:
        """
        Get routing recommendations based on similar conversations.

        Args:
            similar_conversations: List of similar conversation records

        Returns:
            Routing recommendation dictionary
        """
        # Count successful assistants
        assistant_success = defaultdict(lambda: {"count": 0, "success": 0})

        for conv in similar_conversations:
            assistant = conv["assistant_type"]
            assistant_success[assistant]["count"] += 1
            if conv["success_indicator"]:
                assistant_success[assistant]["success"] += 1

        # Calculate success rates
        recommendations = []
        for assistant, stats in assistant_success.items():
            success_rate = (
                stats["success"] / stats["count"] if stats["count"] > 0 else 0
            )
            recommendations.append(
                {
                    "assistant": assistant,
                    "success_rate": success_rate,
                    "sample_size": stats["count"],
                }
            )

        # Sort by success rate
        recommendations.sort(key=lambda x: x["success_rate"], reverse=True)

        return {
            "primary": (
                recommendations[0]["assistant"] if recommendations else "general"
            ),
            "alternatives": [r["assistant"] for r in recommendations[1:3]],
            "confidence": (
                recommendations[0]["success_rate"] if recommendations else 0.5
            ),
        }

    @staticmethod
    def _query_only_classification(
        query: str, embedding: List[float]
    ) -> Dict[str, any]:
        """
        Classify using only the query when no conversation history exists.

        Args:
            query: User query
            embedding: Query embedding

        Returns:
            Classification result
        """
        # Store this as a new conversation start
        # We'll use a simple keyword-based initial classification
        # This will improve over time as we gather more data

        keywords_to_intent = {
            "transaction": ["transaction", "payment", "expense", "income", "spent"],
            "account": ["account", "balance", "bank", "connect"],
            "invoice": ["invoice", "bill", "client", "payment"],
            "forecast": ["forecast", "predict", "future", "cash flow"],
            "budget": ["budget", "limit", "spending"],
        }

        query_lower = query.lower()

        for intent, keywords in keywords_to_intent.items():
            if any(keyword in query_lower for keyword in keywords):
                return {
                    "intent": intent,
                    "confidence": 0.7,
                    "method": "query_only_keywords",
                    "reasoning": f"Query contains keywords for {intent}",
                }

        return {
            "intent": "general",
            "confidence": 0.5,
            "method": "query_only_default",
            "reasoning": "No specific keywords found in query",
        }

    @staticmethod
    def _fallback_classification(query: str) -> Dict[str, any]:
        """Fallback classification when embedding fails."""
        return {
            "intent": "general",
            "confidence": 0.3,
            "method": "fallback",
            "reasoning": "Embedding generation failed",
        }

    @staticmethod
    def _build_reasoning(
        similar_conversations: List[Dict], intent: str, confidence: float
    ) -> str:
        """Build human-readable reasoning for the classification."""
        if not similar_conversations:
            return "No similar conversations found"

        avg_similarity = sum(c["similarity"] for c in similar_conversations) / len(
            similar_conversations
        )

        return (
            f"Found {len(similar_conversations)} similar conversations "
            f"with average similarity of {avg_similarity:.2%}. "
            f"Most common intent was '{intent}' with confidence {confidence:.2%}."
        )
