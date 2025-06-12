from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


class AssistantType(str, Enum):
    """Available assistant types."""
    TRANSACTION = "transaction"
    ACCOUNT = "account"
    INVOICE = "invoice"
    BANK_CONNECTION = "bank_connection"
    PAYMENT_PROCESSING = "payment_processing"
    FORECASTING = "forecasting"
    BUDGET = "budget"
    INSIGHTS = "insights"

@dataclass
class AssistantConfig:
    """Configuration for a single assistant."""
    name: str
    model: str
    tools: List[str]
    instructions: str
    assistant_id: Optional[str] = None

@dataclass
class AssistantResponse:
    """Response from assistant query."""
    content: str
    assistant_type: AssistantType
    function_calls: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    thread_id: Optional[str] = None