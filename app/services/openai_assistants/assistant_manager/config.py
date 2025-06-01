"""
Configuration management for the async assistant manager.
Handles environment variables and default settings.
"""

import os
from typing import Dict, Optional, Any
from .types import AssistantType


class AssistantConfig:
    """
    Handles configuration for an AI assistant system.

    This class is responsible for loading, maintaining, and validating the
    configuration of an AI assistant system. It retrieves settings from
    environment variables and validates whether the required parameters
    are configured properly. It also manages assistant-specific IDs used for
    various functionalities.

    Attributes
    ----------
    api_key : str
        The API key used for authenticating with the OpenAI service.
    model : str
        The model identifier to use for assistant operations.
    timeout : int
        The request timeout configuration, specified in seconds.
    max_retries : int
        The maximum number of retry attempts for requests.
    retry_delay : float
        The delay between retry attempts, specified in seconds.
    assistant_ids : dict
        A dictionary mapping assistant types to their corresponding IDs.
    """

    def __init__(self):
        # API Configuration
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")

        # Timeout Configuration
        self.timeout = int(os.getenv("ASSISTANT_TIMEOUT", "30"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.retry_delay = float(os.getenv("RETRY_DELAY", "1.0"))

        # Assistant IDs
        self.assistant_ids = self._load_assistant_ids()

    @staticmethod
    def _load_assistant_ids() -> Dict[AssistantType, Optional[str]]:
        """Load assistant IDs from environment."""
        return {
            AssistantType.TRANSACTION: os.getenv("TRANSACTION_ASSISTANT_ID"),
            AssistantType.ACCOUNT: os.getenv("ACCOUNT_ASSISTANT_ID"),
            AssistantType.BANK_CONNECTION: os.getenv("BANK_CONNECTION_ASSISTANT_ID"),
            AssistantType.PAYMENT_PROCESSING: os.getenv("PAYMENT_PROCESSING_ASSISTANT_ID"),
            AssistantType.INVOICE: os.getenv("INVOICE_ASSISTANT_ID"),
            AssistantType.FORECASTING: os.getenv("FORECASTING_ASSISTANT_ID"),
            AssistantType.BUDGET: os.getenv("BUDGET_ASSISTANT_ID"),
            AssistantType.INSIGHTS: os.getenv("INSIGHTS_ASSISTANT_ID"),
        }

    def get_assistant_id(self, assistant_type: AssistantType) -> Optional[str]:
        """Get assistant ID for a given type."""
        return self.assistant_ids.get(assistant_type)

    def is_assistant_configured(self, assistant_type: AssistantType) -> bool:
        """Check if an assistant is configured."""
        return self.get_assistant_id(assistant_type) is not None

    def validate(self) -> Dict[str, Any]:
        """Validate configuration."""
        issues = []

        if not self.api_key:
            issues.append("OPENAI_API_KEY not set")

        # Check which assistants are configured
        configured = []
        missing = []

        for assistant_type in AssistantType:
            if self.is_assistant_configured(assistant_type):
                configured.append(assistant_type.value)
            else:
                missing.append(assistant_type.value)

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "configured_assistants": configured,
            "missing_assistants": missing
        }