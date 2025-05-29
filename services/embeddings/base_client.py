"""
Base embedding client with retry logic and error handling.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import openai
from openai import OpenAI

from config.openai import OpenAIConfig

logger = logging.getLogger(__name__)


class BaseEmbeddingClient(ABC):
    """Abstract base class for embedding clients."""

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key)

    @abstractmethod
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for a single text."""
        pass

    @abstractmethod
    def create_embeddings(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Create embeddings for multiple texts."""
        pass

    def _retry_with_backoff(self, func, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        for attempt in range(self.config.max_retries):
            try:
                return func(*args, **kwargs)
            except openai.RateLimitError as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(
                        f"Rate limit exceeded after {self.config.max_retries} attempts"
                    )
                    raise
                wait_time = (2**attempt) + 0.1
                logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                return None
            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
        return None
