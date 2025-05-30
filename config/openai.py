"""
OpenAI configuration settings.
"""

import os
from dataclasses import dataclass


@dataclass
class OpenAIConfig:
    """
    Configuration for OpenAI API integration.

    Represents the configuration required to interact with the OpenAI API,
    including authentication credentials, model specification, and operational
    parameters. This class provides an interface to define the configuration
    either directly or through environment variables.

    Attributes
    ----------
    api_key : str
        The OpenAI API key required for authentication.
    embedding_model : str
        The name of the embedding model to be used (default: "text-embedding-3-small").
    embedding_dimensions : int
        Dimensionality of the embeddings produced by the model (default: 1536).
    max_tokens : int
        Maximum number of tokens allowed for the specified embedding model
        (default: 8191 for "text-embedding-3-small").
    request_timeout : int
        Timeout duration (in seconds) for API requests (default: 30).
    max_retries : int
        Maximum number of retries for API requests in case of failure (default: 3).
    """

    api_key: str
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    max_tokens: int = 8191  # Max tokens for text-embedding-3-small
    request_timeout: int = 30
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        """Create config from environment variables."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")

        return cls(
            api_key=api_key,
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ),
            embedding_dimensions=int(os.getenv("OPENAI_EMBEDDING_DIMENSIONS", "1536")),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "8191")),
            request_timeout=int(os.getenv("OPENAI_REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
        )
