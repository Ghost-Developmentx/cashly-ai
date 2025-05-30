"""
Async OpenAI embedding client implementation.
"""

import logging
from typing import List, Optional
from openai import AsyncOpenAI

from config.openai import OpenAIConfig
from .embedding_processor import EmbeddingProcessor
from .token_manager import TokenManager
from .retry_handler import AsyncRetryHandler

logger = logging.getLogger(__name__)


class AsyncOpenAIEmbeddingClient:
    """
    Asynchronous client for interacting with OpenAI API to generate text embeddings.

    This class encapsulates functionality to generate single or batch text embeddings
    asynchronously using the OpenAI API. It handles request preparation, retry logic,
    and response mapping, ensuring robust interaction with the OpenAI service.

    Attributes
    ----------
    config : OpenAIConfig or None
        Configuration object containing OpenAI API-related settings. If not provided,
        it is initialized from the environment.
    client : AsyncOpenAI
        Asynchronous client instance for communicating with OpenAI APIs.
    token_manager : TokenManager
        Manages tokenization and truncation of text inputs.
    processor : EmbeddingProcessor
        Processes and formats text before embedding creation.
    retry_handler : AsyncRetryHandler
        Handles retries for embedding API calls.
    """

    def __init__(self, config: Optional[OpenAIConfig] = None):
        self.config = config or OpenAIConfig.from_env()
        self.client = AsyncOpenAI(api_key=self.config.api_key)

        # Initialize components
        self.token_manager = TokenManager(self.config)
        self.processor = EmbeddingProcessor()
        self.retry_handler = AsyncRetryHandler(self.config)

    async def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for a single text asynchronously.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        # Prepare text
        processed_text = self.processor.prepare_text(text)
        truncated_text = self.token_manager.truncate_text(processed_text)

        try:
            # Create embedding with retry logic
            response = await self.retry_handler.execute_with_retry(
                self._create_single_embedding, truncated_text
            )

            if response:
                embedding = response.data[0].embedding
                logger.debug(f"Created embedding of dimension {len(embedding)}")
                return embedding

            return None

        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return None

    async def create_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts asynchronously.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        if not texts:
            return []

        # Process and filter texts
        processed_data = self._prepare_batch_texts(texts)

        if not processed_data["valid_texts"]:
            logger.warning("No valid texts to embed")
            return [None] * len(texts)

        try:
            # Create embeddings with retry logic
            response = await self.retry_handler.execute_with_retry(
                self._create_batch_embeddings, processed_data["valid_texts"]
            )

            if response:
                # Map embeddings back to original positions
                return self._map_embeddings_to_original(
                    texts, processed_data["valid_indices"], response.data
                )

            return [None] * len(texts)

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return [None] * len(texts)

    async def _create_single_embedding(self, text: str):
        """Create a single embedding using the OpenAI API."""
        return await self.client.embeddings.create(
            input=text,
            model=self.config.embedding_model,
        )

    async def _create_batch_embeddings(self, texts: List[str]):
        """Create batch embeddings using the OpenAI API."""
        return await self.client.embeddings.create(
            input=texts,
            model=self.config.embedding_model,
        )

    def _prepare_batch_texts(self, texts: List[str]) -> dict:
        """Prepare texts for batch processing."""
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                processed = self.processor.prepare_text(text)
                truncated = self.token_manager.truncate_text(processed)
                valid_texts.append(truncated)
                valid_indices.append(i)

        return {"valid_texts": valid_texts, "valid_indices": valid_indices}

    @staticmethod
    def _map_embeddings_to_original(
        original_texts: List[str], valid_indices: List[int], embedding_data: List
    ) -> List[Optional[List[float]]]:
        """Map embeddings back to original text positions."""
        result = [None] * len(original_texts)

        for i, idx in enumerate(valid_indices):
            result[idx] = embedding_data[i].embedding

        return result

    async def close(self):
        """Close the async client."""
        await self.client.close()
