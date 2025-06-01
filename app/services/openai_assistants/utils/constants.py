"""
Centralized constants and mappings for OpenAI Assistant integration.
"""

from typing import Dict, List, Set
from enum import Enum


# Intent to Assistant Mapping
INTENT_TO_ASSISTANT_MAPPING: Dict[str, str] = {
    "transactions": "transaction",
    "accounts": "account",
    "invoices": "invoice",
    "forecasting": "forecasting",
    "budgets": "budget",
    "insights": "insights",
    "general": "transaction",  # Default fallback
    "bank_connection": "bank_connection",
    "payment_processing": "payment_processing",
}

# Cross-Assistant Routing Patterns
CROSS_ROUTING_PATTERNS: Dict[str, Dict[str, str]] = {
    "bank_connection": {
        "account_balance": "account",
        "total_balance": "account",
        "account_details": "account",
        "how_much_money": "account",
    },
    "account": {
        "transactions": "transaction",
        "spending": "transaction",
        "expenses": "transaction",
        "recent_activity": "transaction",
    },
    "transaction": {
        "forecast": "forecasting",
        "predict": "forecasting",
        "cash_flow": "forecasting",
        "invoice": "invoice",
        "future": "forecasting",
    },
}

# Function to Action Mapping
FUNCTION_TO_ACTION_MAPPING: Dict[str, str] = {
    "get_user_accounts": "show_accounts",
    "get_transactions": "show_transactions",
    "create_transaction": "transaction_created",
    "update_transaction": "transaction_updated",
    "delete_transaction": "transaction_deleted",
    "get_invoices": "show_invoices",
    "create_invoice": "invoice_created",
    "send_invoice": "send_invoice",
    "initiate_plaid_connection": "initiate_plaid_connection",
    "setup_stripe_connect": "setup_stripe_connect",
    "check_stripe_connect_status": "check_stripe_connect_status",
    "create_stripe_connect_dashboard_link": "create_stripe_connect_dashboard_link",
    "forecast_cash_flow": "show_forecast",
    "generate_budget": "show_budget",
    "analyze_trends": "show_trends",
    "detect_anomalies": "show_anomalies",
    "categorize_transactions": "transactions_categorized",
}

# Assistant Keywords for Routing
ASSISTANT_KEYWORDS: Dict[str, List[str]] = {
    "bank_connection": [
        "connect",
        "link",
        "add",
        "integrate",
        "setup",
        "plaid",
        "new account",
        "another account",
        "connect bank",
        "add bank",
        "link account",
        "integrate account",
    ],
    "payment_processing": [
        "payment",
        "process payment",
        "pay",
        "stripe",
        "charge",
        "send money",
        "transfer",
        "payment method",
        "credit card",
        "debit card",
    ],
}

# Invoice Context Keywords
INVOICE_CONTEXT_KEYWORDS: List[str] = [
    "send it",
    "yes send",
    "go ahead",
    "send that",
    "send the invoice",
    "confirm",
    "looks good",
    "send to",
    "send invoice",
]

# Invoice History Phrases
INVOICE_HISTORY_PHRASES: List[str] = [
    "created a draft invoice",
    "invoice draft created",
    "ready to send",
    "invoice for",
]

# Routing Trigger Phrases
ROUTING_TRIGGER_PHRASES: List[str] = [
    "transaction assistant",
    "account assistant",
    "invoice assistant",
    "refer to the",
    "ask the",
    "contact the",
]

# Assistant Mention Keywords
ASSISTANT_MENTION_KEYWORDS: Dict[str, str] = {
    "transaction": "transaction",
    "account": "account",
    "bank_connection": "bank_connection",
    "payment_processing": "payment_processing",
    "invoice": "invoice",
    "forecasting": "forecasting",
    "budget": "budget",
    "insights": "insights",
}


# Response Thresholds
class ResponseThresholds:
    MIN_FUNCTION_CALLS_FOR_NO_REROUTE = 1
    MIN_CONTENT_LENGTH_FOR_NO_REROUTE = 150


# Routing Strategies
class RoutingStrategy(Enum):
    DIRECT_ROUTE = "direct_route"
    ROUTE_WITH_FALLBACK = "route_with_fallback"
    GENERAL_WITH_CONTEXT = "general_with_context"
    GENERAL_FALLBACK = "general_fallback"
    ERROR = "error"


# Default Values
DEFAULT_CONVERSATION_HISTORY_LIMIT = 10
DEFAULT_RECENT_MESSAGES_COUNT = 4
DEFAULT_INVOICE_DUE_DAYS = 30
