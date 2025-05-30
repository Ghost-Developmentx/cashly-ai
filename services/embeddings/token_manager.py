"""
Manages token counting and text truncation for embeddings.
"""

import logging
import tiktoken
from typing import List
from config.openai import OpenAIConfig

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Manages token operations, including counting, truncating, and estimating tokens for given texts,
    while considering model-specific encoding and token limits.

    This class is designed to interact with OpenAI-configured models and provides utility methods to handle
    strings in terms of tokenization. It can fall back to approximate character-based calculations if
    tokenizer loads fail or other exceptions occur.

    Attributes
    ----------
    config : OpenAIConfig
        Configuration object containing model-related parameters.
    max_tokens : int
        The maximum number of tokens allowed as per the model configuration.
    encoding : tiktoken.Encoding
        Tokenizer encoding object used for tokenizing and decoding text.
    """

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.max_tokens = config.max_tokens

        try:
            self.encoding = tiktoken.encoding_for_model(config.embedding_model)
        except Exception as e:
            logger.warning(
                f"Failed to load tokenizer for {config.embedding_model}: {e}"
            )
            # Fallback to cl100k_base encoding
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.error(f"Failed to count tokens: {e}")
            # Fallback to character-based estimation
            return len(text) // 4

    def truncate_text(self, text: str) -> str:
        """
        Truncate text to fit within token limits.

        Args:
            text: Text to truncate

        Returns:
            Truncated text
        """
        try:
            tokens = self.encoding.encode(text)

            if len(tokens) <= self.max_tokens:
                return text

            # Truncate and decode
            truncated_tokens = tokens[: self.max_tokens]
            truncated_text = self.encoding.decode(truncated_tokens)

            logger.debug(
                f"Truncated text from {len(tokens)} to {self.max_tokens} tokens"
            )
            return truncated_text

        except Exception as e:
            logger.error(f"Failed to truncate with tokenizer: {e}")
            # Fallback to character-based truncation
            max_chars = self.max_tokens * 4  # Rough estimate
            if len(text) > max_chars:
                logger.warning(f"Using character-based truncation")
                return text[:max_chars]
            return text

    def estimate_batch_tokens(self, texts: List[str]) -> int:
        """
        Estimate total tokens for a batch of texts.

        Args:
            texts: List of texts

        Returns:
            Estimated total tokens
        """
        total = 0
        for text in texts:
            total += self.count_tokens(text)
        return total
