"""
Learns from conversation outcomes to improve intent classification.
"""

import logging
from typing import Dict, List, Optional
from services.embeddings.openai_client import OpenAIEmbeddingClient
from services.embeddings.context_builder import ConversationContextBuilder
from services.embeddings.async_embeddings import AsyncEmbeddingStorage

logger = logging.getLogger(__name__)


class IntentLearner:
    """Learns from conversation outcomes to improve classification."""

    def __init__(
        self,
        embedding_client: Optional[OpenAIEmbeddingClient] = None,
        context_builder: Optional[ConversationContextBuilder] = None,
        storage: Optional[AsyncEmbeddingStorage] = None,
    ):
        self.embedding_client = embedding_client or OpenAIEmbeddingClient()
        self.context_builder = context_builder or ConversationContextBuilder()
        self.storage = storage or AsyncEmbeddingStorage()

    def learn_from_conversation(
        self,
        conversation_id: str,
        user_id: str,
        conversation_history: List[Dict],
        final_intent: str,
        final_assistant: str,
        success_indicator: bool,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Learn from a completed conversation.

        Args:
            conversation_id: Unique conversation ID
            user_id: User identifier
            conversation_history: Full conversation history
            final_intent: Final classified intent
            final_assistant: Assistant that handled the conversation
            success_indicator: Whether the conversation was successful
            metadata: Additional metadata

        Returns:
            True if learning was successful
        """
        try:
            # Build context from the full conversation
            if not conversation_history:
                logger.warning(f"No conversation history for {conversation_id}")
                return False

            # Get the last user query
            last_query = None
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    last_query = msg.get("content", "")
                    break

            if not last_query:
                logger.warning(f"No user query found in conversation {conversation_id}")
                return False

            # Build context
            context = self.context_builder.build_context(
                conversation_history, last_query
            )

            # Generate embedding
            embedding = self.embedding_client.create_embedding(context)

            if not embedding:
                logger.error(
                    f"Failed to generate embedding for conversation {conversation_id}"
                )
                return False

            # Extract metadata
            conversation_metadata = self.context_builder.extract_metadata(
                conversation_history, final_intent, final_assistant
            )

            # Merge with provided metadata
            if metadata:
                conversation_metadata.update(metadata)

            # Store the embedding
            embedding_id = self.storage.store_embedding(
                conversation_id=conversation_id,
                user_id=user_id,
                embedding=embedding,
                intent=final_intent,
                assistant_type=final_assistant,
                metadata=conversation_metadata,
                success_indicator=success_indicator,
            )

            if embedding_id:
                logger.info(
                    f"Learned from conversation {conversation_id}: "
                    f"intent={final_intent}, success={success_indicator}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to learn from conversation {conversation_id}: {e}")
            return False
