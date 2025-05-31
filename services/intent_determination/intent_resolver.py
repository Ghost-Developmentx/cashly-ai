# services/intent_determination/async_intent_resolver.py
"""
Async intent resolver using vector similarity search.
Determines intent through context-aware embedding search.
"""

import logging
from typing import Dict, Any, List, Optional

from ..embeddings.async_embedding_client import AsyncOpenAIEmbeddingClient
from services.search.async_vector_search import AsyncVectorSearchService
from .intent_determiner import IntentDeterminer
from .routing_intelligence import RoutingIntelligence
from .context_aggregator import AsyncContextAggregator

logger = logging.getLogger(__name__)

_shared_search_service = AsyncVectorSearchService()


class AsyncIntentResolver:
    """
    Handles asynchronous intent resolution.
    """

    _shared_embedding_client: Optional[AsyncOpenAIEmbeddingClient] = None

    def __init__(self):
        # Use a shared embedding client to avoid connection issues
        if AsyncIntentResolver._shared_embedding_client is None:
            AsyncIntentResolver._shared_embedding_client = AsyncOpenAIEmbeddingClient()
        self.embedding_client = AsyncIntentResolver._shared_embedding_client

        self.search_service = _shared_search_service
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
        """
        try:
            # Log the start of processing
            logger.info(f"Starting intent resolution for query: '{query}'")

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

            # Log embedding generation
            logger.info(
                f"Generating embedding for text ({len(processed_context['embedding_text'])} chars)"
            )

            # Generate embedding asynchronously
            embedding = await self.embedding_client.create_embedding(
                processed_context["embedding_text"]
            )

            if not embedding:
                logger.warning("Failed to generate embedding")
                return self._create_fallback_resolution(query)

            # Continue with search and resolution...
            search_results = await self.search_service.search_similar(
                embedding=embedding, user_id=None, limit=10
            )

            intent, confidence, analysis = self.intent_determiner.determine_intent(
                search_results=search_results,
                query_context=processed_context["key_information"],
            )

            routing = self.routing_intelligence.recommend_assistant(
                intent=intent,
                search_results=search_results,
                user_preferences=(
                    user_context.get("preferences") if user_context else None
                ),
            )

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
