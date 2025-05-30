"""
Async intent resolver using vector similarity search.
Determines intent through context-aware embedding search.
"""

import logging
from typing import Dict, Any, List, Optional

from services.embeddings.openai_client import OpenAIEmbeddingClient
from services.search.async_vector_search import AsyncVectorSearchService
from services.intent_determination.intent_determiner import IntentDeterminer
from services.intent_determination.routing_intelligence import RoutingIntelligence
from services.conversations.aggregator_service import AggregatorService

logger = logging.getLogger(__name__)


class AsyncIntentResolver:
    """Async intent resolution using similarity search."""

    def __init__(self):
        self.embedding_client = OpenAIEmbeddingClient()
        self.search_service = AsyncVectorSearchService()
        self.intent_determiner = IntentDeterminer()
        self.routing_intelligence = RoutingIntelligence()
        self.aggregator_service = AggregatorService()

    async def resolve_intent(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        user_id: str,
        conversation_id: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve intent asynchronously using context and similarity search.

        Args:
            query: Current user query
            conversation_history: Previous messages
            user_id: User identifier
            conversation_id: Conversation identifier
            user_context: Additional user context

        Returns:
            Complete intent resolution with routing
        """
        try:
            # Process conversation context
            processed_context = self.aggregator_service.process_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                conversation_history=conversation_history,
                user_context=user_context,
            )

            # Generate embedding (sync for now)
            embedding = self.embedding_client.create_embedding(
                processed_context["embedding_text"]
            )

            if not embedding:
                logger.warning("Failed to generate embedding")
                return self._fallback_resolution(query)

            # Search for similar conversations (async)
            search_results = await self.search_service.search_similar(
                embedding=embedding, user_id=user_id, limit=10
            )

            # Determine intent
            intent, confidence, analysis = self.intent_determiner.determine_intent(
                search_results=[r.to_dict() for r in search_results],
                query_context=processed_context["key_information"],
            )

            # Get routing recommendation
            routing = self.routing_intelligence.recommend_assistant(
                intent=intent,
                search_results=[r.to_dict() for r in search_results],
                user_preferences=(
                    user_context.get("preferences") if user_context else None
                ),
            )

            # Build complete resolution
            resolution = {
                "intent": intent,
                "confidence": confidence,
                "recommended_assistant": routing["assistant"],
                "routing_confidence": routing["confidence"],
                "method": "async_context_aware_similarity",
                "analysis": {
                    **analysis,
                    "routing": routing,
                    "context_summary": processed_context["key_information"],
                    "search_results_count": len(search_results),
                },
            }

            logger.info(
                f"Resolved intent: {intent} ({confidence:.1%}) "
                f"-> {routing['assistant']}"
            )

            return resolution

        except Exception as e:
            logger.error(f"Async intent resolution failed: {e}")
            return self._fallback_resolution(query)

    @staticmethod
    def _fallback_resolution(query: str) -> Dict[str, Any]:
        """Fallback resolution when async method fails."""
        query_lower = query.lower()

        # Simple keyword matching
        if any(word in query_lower for word in ["transaction", "payment", "expense"]):
            intent = "transactions"
        elif any(word in query_lower for word in ["invoice", "bill", "client"]):
            intent = "invoices"
        elif any(word in query_lower for word in ["forecast", "predict", "future"]):
            intent = "forecasting"
        elif any(word in query_lower for word in ["budget", "limit", "spending"]):
            intent = "budgets"
        else:
            intent = "general"

        return {
            "intent": intent,
            "confidence": 0.6,
            "recommended_assistant": f"{intent}_assistant",
            "routing_confidence": 0.5,
            "method": "keyword_fallback",
            "analysis": {"reason": "Async resolution failed, using keywords"},
        }

    async def close(self):
        """Close async resources."""
        await self.search_service.close()
