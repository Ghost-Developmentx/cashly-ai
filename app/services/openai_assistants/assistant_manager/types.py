"""
Type definitions for the async assistant manager.
Provides core enums and response models.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum


class AssistantType(Enum):
    """
    Represents different categories of assistant functionalities.

    This enumeration classifies various functionalities into specific
    categories to facilitate organization, identification, and usage
    within financial and operational systems.

    Attributes
    ----------
    TRANSACTION : str
        Represents functionalities related to transactions.
    ACCOUNT : str
        Represents functionalities associated with accounts.
    BANK_CONNECTION : str
        Represents functionalities involving connections to banks.
    PAYMENT_PROCESSING : str
        Represents functionalities for handling payment processing.
    INVOICE : str
        Represents functionalities dealing with invoices.
    FORECASTING : str
        Represents functionalities for financial forecasting.
    BUDGET : str
        Represents functionalities focused on budgeting.
    INSIGHTS : str
        Represents functionalities providing analytical insights.
    GENERAL : str
        Represents general or miscellaneous functionalities.
    """
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
    """
    Representation of an Assistant's response containing details about content,
    type, function calls, metadata, success status, and an optional error message.

    This class is used to encapsulate all necessary details of a response generated
    by an Assistant. It organizes metadata, execution details of functions called,
    and provides information about whether the operation was successful or not.

    Attributes
    ----------
    content : str
        The primary content or message generated in the Assistant's response.
    assistant_type : AssistantType
        The type/category of the assistant generating this response.
    function_calls : list of dict
        A collection of function call details made during the response generation.
    metadata : dict
        Additional metadata associated with the response.
    success : bool
        Indicates whether the response generation was successful.
    error : str or None, optional
        Describes details of any error encountered during response generation.
    """
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
    """
    Represents an enhancement of a query with additional context and instructions.

    This data class is designed to hold information about a query and its enhanced
    version, which includes additional instructions and context segments. It allows
    for the determination of whether any actual context has been added to the query.

    Attributes
    ----------
    original_query : str
        The original, unmodified query provided by the user.
    enhanced_query : str
        The query after enhancement with contextual information or modifications.
    context_parts : List[str]
        A list of context parts or segments added to enhance the original query.
    additional_instructions : str
        Additional instructions or guidance included alongside the enhanced query.
    """
    original_query: str
    enhanced_query: str
    context_parts: List[str]
    additional_instructions: str

    @property
    def has_context(self) -> bool:
        """Check if context was added."""
        return len(self.context_parts) > 0