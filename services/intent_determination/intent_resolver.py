"""
Main service that ties together similarity search and intent determination.
"""

import logging
from typing import Dict, Any, List, Optional

from services.embeddings.openai_client import OpenAIEmbeddingClient
from services.search.vector_search import VectorSearchService
from services.intent_determination.intent_determiner import IntentDeterminer
from services.intent_determination.routing_intelligence import RoutingIntelligence
from services.conversations.aggregator_service import AggregatorService

logger = logging.getLogger(__name__)


class IntentResolver:
    """Resolves intent using context-aware similarity search."""

    def __init__(self):
        self.embedding_client = OpenAIEmbeddingClient()
        self.search_service = VectorSearchService()
        self.intent_determiner = IntentDeterminer()
        self.routing_intelligence = RoutingIntelligence()
        self.aggregator_service = AggregatorService()

    def resolve_intent(
        self,
        query: str,
        conversation_history: List[Dict[str, Any]],
        user_id: str,
        conversation_id: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve intent using full context and similarity search.
        """
        logger.info(f"🎯 IntentResolver.resolve_intent called with query: '{query}'")
        logger.info(f"   User ID: {user_id}")
        logger.info(f"   Conversation ID: {conversation_id}")
        logger.info(
            f"   History length: {len(conversation_history) if conversation_history else 0}"
        )

        try:
            # FIXED: For new conversations, add the query to the history
            if not conversation_history:
                logger.info("Empty conversation, using query as initial message")
                # Create a minimal conversation with just the query
                from datetime import datetime

                conversation_history = [
                    {
                        "role": "user",
                        "content": query,
                        "created_at": datetime.now().isoformat(),
                    }
                ]

            # Process conversation context
            processed_context = self.aggregator_service.process_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                conversation_history=conversation_history,
                user_context=user_context,
            )

            embedding_text = processed_context["embedding_text"]

            if query not in embedding_text:
                logger.warning(
                    f"Query '{query}' not found in embedding text, appending it"
                )
                embedding_text = f"{embedding_text}\nCurrent query: {query}"

            embedding = self.embedding_client.create_embedding(embedding_text)

            if not embedding:
                logger.warning("Failed to generate embedding")
                return self._fallback_resolution(query)

            search_results = self.search_service.search_similar(
                embedding=embedding,
                user_id=None,
                limit=10,
                similarity_threshold=0.5,
            )

            # Log what we found
            if search_results:
                logger.info(
                    f"Found {len(search_results)} similar conversations, "
                    f"top similarity: {search_results[0].similarity_score:.3f}"
                )

            # Determine intent
            intent, confidence, analysis = self.intent_determiner.determine_intent(
                search_results=search_results,
                query_context=processed_context["key_information"],
            )

            # Get routing recommendation
            routing = self.routing_intelligence.recommend_assistant(
                intent=intent,
                search_results=search_results,
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
                "method": "context_aware_similarity",
                "analysis": {
                    **analysis,
                    "routing": routing,
                    "context_summary": processed_context["key_information"],
                },
            }

            logger.info(
                f"Resolved intent: {intent} ({confidence:.1%}) "
                f"-> {routing['assistant']}"
            )

            return resolution

        except Exception as e:
            logger.error(f"Intent resolution failed: {e}")
            import traceback

            traceback.print_exc()
            return self._fallback_resolution(query)

    @staticmethod
    def _fallback_resolution(query: str) -> Dict[str, Any]:
        """Fallback resolution when a context-aware method fails."""
        # Simple keyword-based fallback
        query_lower = query.lower()

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
            "analysis": {"reason": "Context-aware resolution failed, using keywords"},
        }
