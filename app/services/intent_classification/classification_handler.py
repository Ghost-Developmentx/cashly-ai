"""
Handles vector-based classification for the async intent service.
"""

import logging
from typing import Dict, Optional, List, Any
from ..intent_determination.intent_resolver import AsyncIntentResolver

logger = logging.getLogger(__name__)


class AsyncClassificationHandler:
    """Handles vector-based intent classification."""

    def __init__(self, resolver: AsyncIntentResolver):
        self.resolver = resolver

    async def classify_with_vectors(
        self,
        query: str,
        user_context: Optional[Dict],
        conversation_history: Optional[List[Dict]],
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt vector-based classification.

        Args:
            query: User's query
            user_context: Optional user context
            conversation_history: Optional conversation history

        Returns:
            Classification result or None if confidence too low
        """
        try:
            logger.info(f"🔍 Attempting vector classification for: '{query}'")

            # Prepare context
            user_id = self._get_user_id(user_context)
            conversation_id = self._get_conversation_id(user_context, user_id)

            # Use async resolver
            resolution = await self.resolver.resolve_intent(
                query=query,
                conversation_history=conversation_history or [],
                user_id=user_id,
                conversation_id=conversation_id,
                user_context=user_context,
            )

            # Log results
            self._log_resolution(resolution)

            # Apply confidence boosting if needed
            resolution = self._apply_confidence_boosting(resolution)

            return resolution if resolution["confidence"] >= 0.55 else None

        except Exception as e:
            logger.error(f"❌ Vector classification error: {e}", exc_info=True)
            return None

    @staticmethod
    def _get_user_id(user_context: Optional[Dict]) -> str:
        """Extract user ID from context."""
        if user_context:
            # Check multiple possible locations for user_id
            if "user_id" in user_context:
                return str(user_context["user_id"])
            # Check if it's nested in metadata
            if "metadata" in user_context and "user_id" in user_context["metadata"]:
                return str(user_context["metadata"]["user_id"])

        # Log warning when falling back to anonymous
        logger.warning("No user_id found in context, using anonymous")
        return "anonymous"

    @staticmethod
    def _get_conversation_id(user_context: Optional[Dict], user_id: str) -> str:
        """Extract or generate conversation ID."""
        if user_context and "conversation_id" in user_context:
            return user_context["conversation_id"]
        return f"temp_{user_id}"

    @staticmethod
    def _log_resolution(resolution: Dict[str, Any]):
        """Log resolution details."""
        logger.info(f"📊 Vector resolution result:")
        logger.info(f"   Intent: {resolution['intent']}")
        logger.info(f"   Confidence: {resolution['confidence']:.3f}")
        logger.info(f"   Method: {resolution.get('method', 'unknown')}")

    @staticmethod
    def _apply_confidence_boosting(resolution: Dict[str, Any]) -> Dict[str, Any]:
        """Apply confidence boosting based on similarity."""
        if "analysis" in resolution and "avg_similarity" in resolution.get(
            "analysis", {}
        ):
            avg_similarity = resolution["analysis"]["avg_similarity"]

            if avg_similarity >= 0.65:
                raw_confidence = resolution["confidence"]
                boosted_confidence = min(raw_confidence * 1.2, 0.95)

                logger.info(
                    f"   Boosted confidence from {raw_confidence:.3f} to "
                    f"{boosted_confidence:.3f} due to similarity {avg_similarity:.3f}"
                )
                resolution["confidence"] = boosted_confidence

        return resolution
