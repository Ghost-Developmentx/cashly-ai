"""
Async implementation of intent learning from conversations.
"""

import logging
from typing import Dict, List, Optional
from ..embeddings.async_embeddings import AsyncEmbeddingStorage
from .learning_context_builder import LearningContextBuilder
from .learning_metadata_extractor import LearningMetadataExtractor
from ..embeddings.async_embedding_client import AsyncOpenAIEmbeddingClient

logger = logging.getLogger(__name__)


class AsyncIntentLearner:
    """
    AsyncIntentLearner is responsible for asynchronously learning from
    conversations by generating embeddings and storing them with relevant metadata.

    This class allows learning from user and assistant interactions after the
    conversation has concluded. It validates inputs, processes conversation history,
    extracts metadata, and uses AI-driven embedding generation to analyze and store
    structured learning data asynchronously.

    Attributes
    ----------
    storage : AsyncEmbeddingStorage
        Asynchronous storage client for embeddings and metadata.
    context_builder : LearningContextBuilder
        Builder for generating learning context from conversation history.
    metadata_extractor : LearningMetadataExtractor
        Extractor for metadata from conversations.
    embedding_client : AsyncOpenAIEmbeddingClient
        Client for generating embeddings asynchronously from context text.
    """

    def __init__(self, storage: AsyncEmbeddingStorage):
        self.storage = storage
        self.context_builder = LearningContextBuilder()
        self.metadata_extractor = LearningMetadataExtractor()
        self.embedding_client = AsyncOpenAIEmbeddingClient()

    async def learn_from_conversation(
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
        Learn from a completed conversation asynchronously.

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
            # Validate input
            if not conversation_history:
                logger.warning(f"No conversation history for {conversation_id}")
                return False

            # Extract last user query
            last_query = self._extract_last_query(conversation_history)
            if not last_query:
                logger.warning(f"No user query found in conversation {conversation_id}")
                return False

            # Build context and generate embedding asynchronously
            context = await self.context_builder.build_context_async(
                conversation_history, last_query
            )

            embedding = await self._generate_embedding_async(context)
            if not embedding:
                logger.error(f"Failed to generate embedding for {conversation_id}")
                return False

            # Extract metadata
            conversation_metadata = self.metadata_extractor.extract_metadata(
                conversation_history, final_intent, final_assistant
            )

            # Merge with provided metadata
            if metadata:
                conversation_metadata.update(metadata)

            # Store the embedding asynchronously
            embedding_id = await self.storage.store_embedding(
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
                    f"✅ Learned from conversation {conversation_id}: "
                    f"intent={final_intent}, success={success_indicator}"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to learn from conversation {conversation_id}: {e}")
            return False

    @staticmethod
    def _extract_last_query(conversation_history: List[Dict]) -> Optional[str]:
        """Extract the last user query from conversation history."""
        for msg in reversed(conversation_history):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return None

    async def _generate_embedding_async(self, context: str) -> Optional[List[float]]:
        """
        Generate embedding asynchronously.

        Args:
            context: Context string to embed

        Returns:
            Embedding vector or None if failed
        """
        try:
            embedding = await self.embedding_client.create_embedding(context)
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
