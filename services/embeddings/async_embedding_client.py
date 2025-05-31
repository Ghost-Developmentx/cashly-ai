"""
Async OpenAI embedding client implementation.
"""

import psutil, os
import logging
from typing import List, Optional
import asyncio
import httpx
from openai import AsyncOpenAI

from config.openai import OpenAIConfig
from .embedding_processor import EmbeddingProcessor
from .token_manager import TokenManager
from .retry_handler import AsyncRetryHandler

logger = logging.getLogger(__name__)


class AsyncOpenAIEmbeddingClient:
    """
    Asynchronous client for interacting with OpenAI API to generate text embeddings.
    """

    _instance: Optional["AsyncOpenAIEmbeddingClient"] = None
    _lock = asyncio.Lock()
    _semaphore = asyncio.Semaphore(5)  # Limit concurrent embedding requests

    def __init__(self, config: Optional[OpenAIConfig] = None):
        # ensure singleton initialization
        if hasattr(self, "_initialized"):
            return

        self.config = config or OpenAIConfig.from_env()
        self._initialized = True

        # ---- custom HTTP transport (HTTP/2, explicit timeouts) ------------
        _httpx_client = httpx.AsyncClient(
            http2=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            timeout=httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=10.0,
                pool=5.0,
            ),
        )

        # ---- OpenAI async client (SDK retries disabled) --------------------
        self.client = AsyncOpenAI(
            api_key=self.config.api_key,
            max_retries=0,
            http_client=_httpx_client,
        )

        # sub‑components
        self.token_manager = TokenManager(self.config)
        self.processor = EmbeddingProcessor()
        self.retry_handler = AsyncRetryHandler(self.config)

    # ---------------------------------------------------------------------
    # singleton accessor
    # ---------------------------------------------------------------------
    @classmethod
    async def get_instance(cls, config: Optional[OpenAIConfig] = None):
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(config)
            return cls._instance

    # ---------------------------------------------------------------------
    # public helpers
    # ---------------------------------------------------------------------
    async def create_embedding(self, text: str) -> Optional[List[float]]:
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

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
    @staticmethod
    def _log_sockets(tag: str):
        proc = psutil.Process(os.getpid())
        conns = proc.net_connections(kind="inet")
        outbound = sum(1 for c in conns if c.raddr)
        logger.debug(
            "%s | fds=%d sockets=%d outbound=%d",
            tag,
            proc.num_fds(),
            len(conns),
            outbound,
        )

    # call just before the API request
    async def _create_single_embedding(self, text: str):
        self._log_sockets("📡 BEFORE embedding")
        async with self._semaphore:
            resp = await self.client.embeddings.create(
                input=text, model=self.config.embedding_model
            )
        self._log_sockets("📡 AFTER embedding")
        return resp

    async def _create_batch_embeddings(self, texts: List[str]):
        self._log_sockets("📡 BEFORE embedding")
        async with self._semaphore:
            self._log_sockets("📡 AFTER embedding")
            return await self.client.embeddings.create(
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

    async def close(self):
        await self.client.close()
        logger.info("AsyncOpenAIEmbeddingClient closed")
