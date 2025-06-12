from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any

from app.schemas.assistant import AssistantType


class Intent(str, Enum):
    """Possible query intents."""
    TRANSACTION_QUERY = "transaction_query"
    TRANSACTION_CREATE = "transaction_create"
    TRANSACTION_UPDATE = "transaction_update"
    TRANSACTION_DELETE = "transaction_delete"
    ACCOUNT_BALANCE = "account_balance"
    ACCOUNT_CONNECT = "account_connect"
    INVOICE_CREATE = "invoice_create"
    INVOICE_MANAGE = "invoice_manage"
    PAYMENT_SETUP = "payment_setup"
    FORECAST = "forecast"
    BUDGET = "budget"
    INSIGHTS = "insights"
    GENERAL = "general"

@dataclass
class ClassificationResult:
    """Result of query classification."""
    intent: Intent
    confidence: float
    suggested_assistant: AssistantType
    keywords_found: List[str]
    method: str = "keyword"  # keyword, ml, or hybrid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "suggested_assistant": self.suggested_assistant.value,
            "keywords_found": self.keywords_found,
            "method": self.method
        }
