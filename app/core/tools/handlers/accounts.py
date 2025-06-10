"""
Account tool handlers.
Migrated from app/services/fin/tool_handlers/account_handlers.py
"""

import logging
from typing import Dict, Any
from ..registry import tool_registry
from ..schemas import ACCOUNT_SCHEMAS

logger = logging.getLogger(__name__)

@tool_registry.register(
    name="get_user_accounts",
    description="Get information about the user's connected bank accounts",
    schema=ACCOUNT_SCHEMAS["GET_USER_ACCOUNTS"],
    category="accounts"
)
async def get_user_accounts(context: Dict[str, Any]) -> Dict[str, Any]:
    """Get all user accounts with summary information."""
    user_context = context.get("user_context", {})
    accounts = user_context.get("accounts", [])

    if not accounts:
        return {
            "message": "No bank accounts connected yet. Would you like to connect one?",
            "accounts": [],
            "total_balance": 0,
            "account_count": 0,
            "has_accounts": False,
            "suggestion": "connect_account"
        }

    # Calculate total balance
    total_balance = sum(
        float(acc.get("balance", 0)) for acc in accounts
    )

    # Format accounts for display
    formatted_accounts = []
    for account in accounts:
        formatted_accounts.append({
            "id": account.get("id"),
            "name": account.get("name", "Unknown Account"),
            "institution": account.get("institution", "Unknown Bank"),
            "type": account.get("account_type", "Unknown"),
            "balance": float(account.get("balance", 0)),
            "last_updated": account.get("last_updated")
        })

    return {
        "accounts": formatted_accounts,
        "total_balance": round(total_balance, 2),
        "account_count": len(accounts),
        "has_accounts": True,
        "message": f"You have {len(accounts)} connected account{'s' if len(accounts) != 1 else ''} with a total balance of ${total_balance:,.2f}"
    }

@tool_registry.register(
    name="get_account_details",
    description="Get detailed information about a specific account",
    schema=ACCOUNT_SCHEMAS["GET_ACCOUNT_DETAILS"],
    category="accounts"
)
async def get_account_details(context: Dict[str, Any]) -> Dict[str, Any]:
    """Get detailed information about a specific account."""
    tool_args = context["tool_args"]
    user_context = context.get("user_context", {})

    account_id = tool_args.get("account_id")
    if not account_id:
        return {"error": "Account ID is required"}

    # Find the account
    accounts = user_context.get("accounts", [])
    account = next(
        (acc for acc in accounts if str(acc.get("id")) == str(account_id)),
        None
    )

    if not account:
        return {
            "error": f"Account with ID {account_id} not found",
            "available_accounts": [
                {"id": acc.get("id"), "name": acc.get("name")}
                for acc in accounts
            ]
        }

    # Get recent transactions for this account
    transactions = context.get("transactions", [])
    account_transactions = [
        t for t in transactions
        if str(t.get("account_id")) == str(account_id)
    ]

    # Calculate account statistics
    recent_transactions = account_transactions[:10]  # Last 10 transactions

    return {
        "account": {
            "id": account.get("id"),
            "name": account.get("name", "Unknown Account"),
            "institution": account.get("institution", "Unknown Bank"),
            "type": account.get("account_type", "Unknown"),
            "balance": float(account.get("balance", 0)),
            "available_balance": float(account.get("available_balance", account.get("balance", 0))),
            "currency": account.get("currency", "USD"),
            "last_updated": account.get("last_updated")
        },
        "recent_transactions": recent_transactions,
        "transaction_count": len(account_transactions),
        "connection_status": account.get("connection_status", "active")
    }

@tool_registry.register(
    name="initiate_plaid_connection",
    description="Start the process to connect a new bank account via Plaid",
    schema=ACCOUNT_SCHEMAS["INITIATE_PLAID_CONNECTION"],
    category="accounts"
)
async def initiate_plaid_connection(context: Dict[str, Any]) -> Dict[str, Any]:
    """Initiate Plaid connection for a new bank account."""
    user_id = context.get("user_id")

    if not user_id:
        return {"error": "User ID is required for bank connection"}

    return {
        "action": "initiate_plaid_connection",
        "user_id": user_id,
        "message": "I'll help you connect your bank account. This will open a secure connection to your bank.",
        "instructions": "Click the button below to securely connect your bank account through Plaid."
    }

@tool_registry.register(
    name="disconnect_account",
    description="Disconnect a bank account",
    schema=ACCOUNT_SCHEMAS["DISCONNECT_ACCOUNT"],
    category="accounts",
    requires_confirmation=True
)
async def disconnect_account(context: Dict[str, Any]) -> Dict[str, Any]:
    """Disconnect a bank account."""
    tool_args = context["tool_args"]
    user_context = context.get("user_context", {})
    user_id = context.get("user_id")

    account_id = tool_args.get("account_id")
    if not account_id:
        return {"error": "Account ID is required"}

    # Verify account exists
    accounts = user_context.get("accounts", [])
    account = next(
        (acc for acc in accounts if str(acc.get("id")) == str(account_id)),
        None
    )

    if not account:
        return {"error": f"Account with ID {account_id} not found"}

    return {
        "action": "disconnect_account",
        "account_id": account_id,
        "account_name": account.get("name", "Unknown Account"),
        "user_id": user_id,
        "message": f"Are you sure you want to disconnect {account.get('name', 'this account')}? This will remove all associated data.",
        "requires_confirmation": True,
        "warning": "This action cannot be undone. All transaction history for this account will be removed."
    }
