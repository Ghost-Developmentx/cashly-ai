"""
Async OpenAI embedding client implementation.
"""

import logging
from typing import List, Optional
import asyncio
import httpx
from openai import AsyncOpenAI

from app.core.config import Settings
from app.db.singleton_registry import registry
from .embedding_processor import EmbeddingProcessor
from .token_manager import TokenManager
from .retry_handler import AsyncRetryHandler

logger = logging.getLogger(__name__)


class AsyncOpenAIEmbeddingClient:
    """
    Asynchronous client for OpenAI embeddings with event loop management.
    """

    def __init__(self, config: Optional[Settings] = None):
        self.config = config or Settings()
        self._loop_id: Optional[int] = None
        self._client: Optional[AsyncOpenAI] = None
        self._httpx_client: Optional[httpx.AsyncClient] = None

        # Components that don't need loop awareness
        self.token_manager = TokenManager(self.config)
        self.processor = EmbeddingProcessor()
        self.retry_handler = AsyncRetryHandler(self.config)

        # Semaphore for rate limiting
        self._semaphore = asyncio.Semaphore(5)

    async def is_valid(self) -> bool:
        """Check if this client is valid for the current event loop."""
        current_loop_id = id(asyncio.get_running_loop())
        return self._loop_id == current_loop_id and self._client is not None

    async def _ensure_client(self):
        """Ensure we have a valid client for the current event loop."""
        current_loop_id = id(asyncio.get_running_loop())

        if self._loop_id != current_loop_id or self._client is None:

            logger.info(f"🔄 Creating OpenAI client for loop {current_loop_id}")

            # Create a new HTTP client with proper settings
            self._httpx_client = httpx.AsyncClient(
                http2=True,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.config.request_timeout,
                    write=10.0,
                    pool=5.0,
                ),
            )

            # Create OpenAI client
            self._client = AsyncOpenAI(
                api_key=self.config.api_key,
                max_retries=0,  # We handle retries ourselves
                http_client=self._httpx_client,
            )

            self._loop_id = current_loop_id

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    async def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding with proper client management."""
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        # Ensure we have a valid client
        await self._ensure_client()

        processed_text = self.processor.prepare_text(text)
        truncated_text = self.token_manager.truncate_text(processed_text)

        try:
            response = await self.retry_handler.execute_with_retry(
                self._create_single_embedding, truncated_text
            )
            if response:
                return response.data[0].embedding
            logger.error("No response from embedding creation")
            return None
        except Exception as e:
            logger.error(f"Failed to create embedding: {e}", exc_info=True)
            return None

    async def create_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        if not texts:
            return []

        processed = self._prepare_batch_texts(texts)
        if not processed["valid_texts"]:
            return [None] * len(texts)

        try:
            response = await self.retry_handler.execute_with_retry(
                self._create_batch_embeddings, processed["valid_texts"]
            )
            if response and len(response.data) == len(processed["valid_indices"]):
                return self._map_embeddings_to_original(
                    texts, processed["valid_indices"], response.data
                )
            logger.warning("Mismatch in embedding response length")
            return [None] * len(texts)
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}", exc_info=True)
            return [None] * len(texts)

    # ---------------------------------------------------------------------
    # low‑level helpers
    # ---------------------------------------------------------------------

    # call just before the API request
    async def _create_single_embedding(self, text: str):
        """Create a single embedding."""
        async with self._semaphore:
            return await self._client.embeddings.create(
                input=text, model=self.config.embedding_model
            )

    async def _create_batch_embeddings(self, texts: List[str]):
        async with self._semaphore:
            return await self._client.embeddings.create(
                input=texts,
                model=self.config.embedding_model,
            )

    def _prepare_batch_texts(self, texts: List[str]) -> dict:
        valid_texts, valid_indices = [], []
        for i, t in enumerate(texts):
            if t and t.strip():
                valid_texts.append(
                    self.token_manager.truncate_text(self.processor.prepare_text(t))
                )
                valid_indices.append(i)
        return {"valid_texts": valid_texts, "valid_indices": valid_indices}

    @staticmethod
    def _map_embeddings_to_original(
        original_texts: List[str], valid_indices: List[int], embedding_data: List
    ) -> List[Optional[List[float]]]:
        result = [None] * len(original_texts)
        for i, idx in enumerate(valid_indices):
            result[idx] = embedding_data[i].embedding
        return result

    @classmethod
    async def get_instance(cls) -> "AsyncOpenAIEmbeddingClient":
        """Get a singleton instance using the registry."""

        async def create_client():
            return cls()

        return await registry.get_or_create("openai_embedding_client", create_client)
