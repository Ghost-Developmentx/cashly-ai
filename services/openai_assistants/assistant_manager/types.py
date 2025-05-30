"""
Type definitions for the async assistant manager.
Provides core enums and response models.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum


class AssistantType(Enum):
    """Enumeration of available assistant types."""
    TRANSACTION = "transaction"
    ACCOUNT = "account"
    BANK_CONNECTION = "bank_connection"
    PAYMENT_PROCESSING = "payment_processing"
    INVOICE = "invoice"
    FORECASTING = "forecasting"
    BUDGET = "budget"
    INSIGHTS = "insights"
    GENERAL = "general"


@dataclass
class AssistantResponse:
    """Response from an OpenAI Assistant."""
    content: str
    assistant_type: AssistantType
    function_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "assistant_type": self.assistant_type.value,
            "function_calls": self.function_calls,
            "metadata": self.metadata,
            "success": self.success,
            "error": self.error
        }


@dataclass
class ContextEnhancement:
    """Represents context enhancement for a query."""
    original_query: str
    enhanced_query: str
    context_parts: List[str]
    additional_instructions: str

    @property
    def has_context(self) -> bool:
        """Check if context was added."""
        return len(self.context_parts) > 0