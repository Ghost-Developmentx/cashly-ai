"""
Processes and prepares text for embedding generation.
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingProcessor:
    """
    A processor for preparing text data for embedding generation.

    The `EmbeddingProcessor` provides methods to preprocess individual texts
    as well as batches of texts. Preprocessing includes removal of URLs and
    email addresses, normalization of whitespace, and handling of special characters.
    It can also combine multiple texts into a single string using a specified separator.

    Attributes
    ----------
    url_pattern : Pattern
        Compiled regular expression pattern for matching URLs.
    email_pattern : Pattern
        Compiled regular expression pattern for matching email addresses.
    whitespace_pattern : Pattern
        Compiled regular expression pattern for normalizing whitespace.
    """

    def __init__(self):
        # Preprocessing patterns
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.email_pattern = re.compile(r"\S+@\S+")
        self.whitespace_pattern = re.compile(r"\s+")

    def prepare_text(self, text: str) -> str:
        """
        Prepare text for embedding generation.

        Args:
            text: Raw text

        Returns:
            Processed text
        """
        if not text:
            return ""

        # Apply preprocessing steps
        processed = text
        processed = self._remove_urls(processed)
        processed = self._remove_emails(processed)
        processed = self._normalize_whitespace(processed)
        processed = self._remove_special_characters(processed)

        return processed.strip()

    def prepare_batch(self, texts: List[str]) -> List[str]:
        """
        Prepare a batch of texts.

        Args:
            texts: List of raw texts

        Returns:
            List of processed texts
        """
        return [self.prepare_text(text) for text in texts]

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        return self.url_pattern.sub("[URL]", text)

    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        return self.email_pattern.sub("[EMAIL]", text)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text."""
        return self.whitespace_pattern.sub(" ", text)

    @staticmethod
    def _remove_special_characters(text: str) -> str:
        """Remove problematic special characters."""
        # Remove zero-width characters
        text = text.replace("\u200b", "")
        text = text.replace("\u200c", "")
        text = text.replace("\u200d", "")
        text = text.replace("\ufeff", "")

        # Remove other problematic characters
        text = text.replace("\x00", "")  # Null character

        return text

    @staticmethod
    def combine_texts(texts: List[str], separator: str = " ") -> str:
        """
        Combine multiple texts into one.

        Args:
            texts: List of texts to combine
            separator: Text separator

        Returns:
            Combined text
        """
        return separator.join(filter(None, texts))
