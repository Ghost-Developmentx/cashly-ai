# services/intent_determination/async_intent_resolver.py
"""
Async intent resolver using vector similarity search.
Determines intent through context-aware embedding search.
"""

import logging
from typing import Dict, Any, List, Optional

from ..embeddings.async_embedding_client import AsyncOpenAIEmbeddingClient
from ..search.async_vector_search import AsyncVectorSearchService
from .intent_determiner import IntentDeterminer
from .routing_intelligence import RoutingIntelligence
from .context_aggregator import AsyncContextAggregator

logger = logging.getLogger(__name__)


class AsyncIntentResolver:
    """
    Handles asynchronous intent resolution by processing context, generating embeddings,
    performing similarity searches, and determining intent with appropriate routing decision-making.

    This class is designed to resolve user intents asynchronously by leveraging vector-based
    searches and routing intelligence. It integrates multiple services for embedding generation,
    context aggregation, search, and analysis, ensuring a comprehensive and structured resolution
    procedure. The fallback mechanism ensures reliability in case of resolution failures.

    Attributes
    ----------
    embedding_client : AsyncOpenAIEmbeddingClient
        Service for generating text embeddings asynchronously.
    search_service : AsyncVectorSearchService
        Service for searching similar embeddings or contexts asynchronously.
    intent_determiner : IntentDeterminer
        Component for determining intent from search results and context.
    routing_intelligence : RoutingIntelligence
        Component for recommending routing or assistants based on intent and user preferences.
    context_aggregator : AsyncContextAggregator
        Service for aggregating and processing conversation context asynchronously.
    """

    def __init__(self):
        self.embedding_client = AsyncOpenAIEmbeddingClient()
        self.search_service = AsyncVectorSearchService()
        self.intent_determiner = IntentDeterminer()
        self.routing_intelligence = RoutingIntelligence()
        self.context_aggregator = AsyncContextAggregator()

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
            # Process conversation context asynchronously
            processed_context = (
                await self.context_aggregator.process_conversation_async(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    conversation_history=conversation_history,
                    user_context=user_context,
                    query=query,
                )
            )

            # Generate embedding asynchronously
            embedding = await self.embedding_client.create_embedding(
                processed_context["embedding_text"]
            )

            if not embedding:
                logger.warning("Failed to generate embedding")
                return self._create_fallback_resolution(query)

            # Search for similar conversations asynchronously
            search_results = await self.search_service.search_similar(
                embedding=embedding, user_id=user_id, limit=10
            )

            # Determine intent (sync is fine here - just computation)
            intent, confidence, analysis = self.intent_determiner.determine_intent(
                search_results=search_results,
                query_context=processed_context["key_information"],
            )

            # Get routing recommendation (sync is fine here - just computation)
            routing = self.routing_intelligence.recommend_assistant(
                intent=intent,
                search_results=search_results,
                user_preferences=(
                    user_context.get("preferences") if user_context else None
                ),
            )

            # Build complete resolution
            resolution = self._build_resolution(
                intent=intent,
                confidence=confidence,
                routing=routing,
                analysis=analysis,
                context_summary=processed_context["key_information"],
                search_results_count=len(search_results),
            )

            logger.info(
                f"Resolved intent: {intent} ({confidence:.1%}) "
                f"-> {routing['assistant']}"
            )

            return resolution

        except Exception as e:
            logger.error(f"Async intent resolution failed: {e}", exc_info=True)
            return self._create_fallback_resolution(query)

    @staticmethod
    def _build_resolution(
        intent: str,
        confidence: float,
        routing: Dict[str, Any],
        analysis: Dict[str, Any],
        context_summary: Dict[str, Any],
        search_results_count: int,
    ) -> Dict[str, Any]:
        """Build a complete resolution response."""
        return {
            "intent": intent,
            "confidence": confidence,
            "recommended_assistant": routing["assistant"],
            "routing_confidence": routing["confidence"],
            "method": "async_context_aware_similarity",
            "analysis": {
                **analysis,
                "routing": routing,
                "context_summary": context_summary,
                "search_results_count": search_results_count,
            },
        }

    def _create_fallback_resolution(self, query: str) -> Dict[str, Any]:
        """Create fallback resolution when async method fails."""
        query_lower = query.lower()

        # Simple keyword matching
        intent = self._determine_fallback_intent(query_lower)

        return {
            "intent": intent,
            "confidence": 0.6,
            "recommended_assistant": f"{intent}_assistant",
            "routing_confidence": 0.5,
            "method": "keyword_fallback",
            "analysis": {"reason": "Async resolution failed, using keywords"},
        }

    @staticmethod
    def _determine_fallback_intent(query_lower: str) -> str:
        """Determine intent using simple keyword matching."""
        keyword_intents = {
            "transactions": ["transaction", "payment", "expense", "spent"],
            "invoices": ["invoice", "bill", "client"],
            "forecasting": ["forecast", "predict", "future"],
            "budgets": ["budget", "limit", "spending"],
            "accounts": ["account", "balance", "bank"],
        }

        for intent, keywords in keyword_intents.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent

        return "general"

    async def close(self):
        """Close async resources."""
        await self.embedding_client.close()
        await self.search_service.close()
