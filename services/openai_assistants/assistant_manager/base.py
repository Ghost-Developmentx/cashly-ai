"""
Base manager class for async assistant operations.
Provides core client initialization and common methods.
"""

import logging
from typing import Optional
from openai import AsyncOpenAI
from .config import AssistantConfig
from .types import AssistantType

logger = logging.getLogger(__name__)


class BaseManager:
    """
    Manage assistants and their configuration.

    The BaseManager class is responsible for managing AI assistant configurations,
    validating them, and interacting with an async client for API communication.
    It provides utilities to retrieve assistant identifiers and validate availability.
    This class primarily serves as a backend module for interacting with AI assistants.

    Attributes
    ----------
    config : AssistantConfig
        Configuration object for managing assistants. If none is provided,
        a default AssistantConfig instance is created.
    client : AsyncOpenAI
        Asynchronous client for interacting with the OpenAI API.
    """

    def __init__(self, config: Optional[AssistantConfig] = None):
        """
        Initialize base manager.

        Args:
            config: Optional configuration object
        """
        self.config = config or AssistantConfig()

        # Validate configuration
        validation = self.config.validate()
        if not validation["valid"]:
            logger.warning(f"Configuration issues: {validation['issues']}")

        # Initialize async client
        self.client = AsyncOpenAI(api_key=self.config.api_key)

        logger.info(
            f"Initialized base manager with {len(validation['configured_assistants'])} "
            f"configured assistants"
        )

    def get_assistant_id(self, assistant_type: AssistantType) -> Optional[str]:
        """Get assistant ID for a given type."""
        return self.config.get_assistant_id(assistant_type)

    def is_assistant_available(self, assistant_type: AssistantType) -> bool:
        """Check if an assistant is available."""
        return self.config.is_assistant_configured(assistant_type)