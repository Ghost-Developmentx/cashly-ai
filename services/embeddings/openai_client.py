"""
OpenAI embedding client implementation.
"""

import logging
from typing import List, Optional
import tiktoken

from services.embeddings.base_client import BaseEmbeddingClient
from config.openai import OpenAIConfig

logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """OpenAI embedding client using text-embedding-3-small."""

    def __init__(self, config: Optional[OpenAIConfig] = None):
        super().__init__(config or OpenAIConfig.from_env())
        self.encoding = tiktoken.encoding_for_model(self.config.embedding_model)

    def create_embedding(self, text: str) -> Optional[List[float]]:
        """
        Create embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if failed
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        # Truncate if necessary
        text = self._truncate_text(text)

        try:
            response = self._retry_with_backoff(
                self.client.embeddings.create,
                input=text,
                model=self.config.embedding_model,
            )

            embedding = response.data[0].embedding
            logger.debug(f"Created embedding of dimension {len(embedding)}")
            return embedding

        except Exception as e:
            logger.error(f"Failed to create embedding: {e}")
            return None

    def create_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """
        Create embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for failed embeddings)
        """
        if not texts:
            return []

        # Filter and truncate texts
        processed_texts = [self._truncate_text(t) for t in texts if t and t.strip()]

        if not processed_texts:
            logger.warning("No valid texts to embed")
            return [None] * len(texts)

        try:
            response = self._retry_with_backoff(
                self.client.embeddings.create,
                input=processed_texts,
                model=self.config.embedding_model,
            )

            # Map embeddings back to original text positions
            embeddings = []
            processed_idx = 0

            for original_text in texts:
                if original_text and original_text.strip():
                    embeddings.append(response.data[processed_idx].embedding)
                    processed_idx += 1
                else:
                    embeddings.append(None)

            logger.info(f"Created {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            return [None] * len(texts)

    def _truncate_text(self, text: str) -> str:
        """Truncate text to fit within token limits."""
        tokens = self.encoding.encode(text)

        if len(tokens) <= self.config.max_tokens:
            return text

        # Truncate and decode
        truncated_tokens = tokens[: self.config.max_tokens]
        truncated_text = self.encoding.decode(truncated_tokens)

        logger.warning(
            f"Truncated text from {len(tokens)} to {self.config.max_tokens} tokens"
        )
        return truncated_text

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
