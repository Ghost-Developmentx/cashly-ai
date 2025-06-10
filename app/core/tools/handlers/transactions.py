"""
Transaction tool handlers.
Migrated from app/services/fin/tool_handlers/transaction_handlers.py
"""

import logging
from typing import Dict, Any
from ..registry import tool_registry
from ..schemas import TRANSACTION_SCHEMAS
from ..helpers.transaction_helpers import (
    filter_transactions,
    calculate_summary,
    get_category_breakdown,
    get_transaction_date,
    resolve_account,
    prepare_updates
)

logger = logging.getLogger(__name__)


@tool_registry.register(
    name="get_transactions",
    description="Retrieve and filter user transactions",
    schema=TRANSACTION_SCHEMAS["GET_TRANSACTIONS"],
    category="transactions"
)
async def get_transactions(context: Dict[str, Any]) -> Dict[str, Any]:
    """Get filtered transactions for the user."""
    tool_args = context["tool_args"]
    user_context = context.get("user_context", {})
    transactions = context.get("transactions", [])

    # Extract filters
    days = tool_args.get("days", 30)
    category = tool_args.get("category")
    account_id = tool_args.get("account_id")
    account_name = tool_args.get("account_name")
    transaction_type = tool_args.get("type", "all")

    # Filter transactions
    filtered = await filter_transactions(
        transactions=transactions,
        user_context=user_context,
        days=days,
        category=category,
        account_id=account_id,
        account_name=account_name,
        transaction_type=transaction_type
    )

    # Calculate summary
    summary = calculate_summary(filtered)

    # Get category breakdown for expenses
    expense_transactions = [t for t in filtered if float(t.get("amount", 0)) < 0]
    category_breakdown = get_category_breakdown(expense_transactions)

    return {
        "transactions": filtered,
        "summary": summary,
        "category_breakdown": category_breakdown,
        "count": len(filtered),
        "filters_applied": {
            "days": days,
            "category": category,
            "account": account_id or account_name,
            "type": transaction_type
        }
    }

@tool_registry.register(
    name="create_transaction",
    description="Create a new transaction",
    schema=TRANSACTION_SCHEMAS["CREATE_TRANSACTION"],
    category="transactions",
    requires_confirmation=True
)
async def create_transaction(context: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new transaction - returns action for frontend."""
    tool_args = context["tool_args"]
    user_context = context.get("user_context", {})

    # Validate required fields
    if "amount" not in tool_args:
        return {"error": "Amount is required"}
    if "description" not in tool_args:
        return {"error": "Description is required"}

    # Parse amount
    try:
        amount = float(tool_args["amount"])
        if amount == 0:
            return {"error": "Amount cannot be zero"}
    except (ValueError, TypeError):
        return {"error": "Invalid amount value"}

    # Get account
    account = await resolve_account(
        user_context=user_context,
        account_id=tool_args.get("account_id"),
        account_name=tool_args.get("account_name")
    )

    if not account:
        return {
            "error": "No account specified or found. Please specify an account."
        }

    # Prepare transaction data
    transaction_data = {
        "amount": amount,
        "description": tool_args["description"].strip(),
        "category": tool_args.get("category", "Uncategorized"),
        "date": get_transaction_date(tool_args.get("date")),
        "account_id": account["id"],
        "account_name": account.get("name", "Unknown Account"),
        "recurring": tool_args.get("recurring", False),
        "created_via_ai": True
    }

    return {
        "action": "create_transaction",
        "transaction": transaction_data,
        "message": f"I'll create this transaction for ${abs(amount):.2f} in your {account.get('name', 'account')}."
    }

@tool_registry.register(
    name="update_transaction",
    description="Update an existing transaction",
    schema=TRANSACTION_SCHEMAS["UPDATE_TRANSACTION"],
    category="transactions",
    requires_confirmation=True
)
async def update_transaction(context: Dict[str, Any]) -> Dict[str, Any]:
    """Update a transaction - returns action for frontend."""
    tool_args = context["tool_args"]

    transaction_id = tool_args.get("transaction_id")
    if not transaction_id:
        return {"error": "Transaction ID is required"}

    # Prepare updates
    updates = await prepare_updates(tool_args)
    if "error" in updates:
        return updates

    if not updates:
        return {"error": "No valid updates provided"}

    return {
        "action": "update_transaction",
        "transaction_id": transaction_id,
        "updates": updates,
        "message": "I'll update this transaction for you."
    }

@tool_registry.register(
    name="delete_transaction",
    description="Delete a transaction",
    schema=TRANSACTION_SCHEMAS["DELETE_TRANSACTION"],
    category="transactions",
    requires_confirmation=True
)
async def delete_transaction(context: Dict[str, Any]) -> Dict[str, Any]:
    """Delete a transaction - returns action for frontend."""
    tool_args = context["tool_args"]
    transaction_id = tool_args.get("transaction_id")

    if not transaction_id:
        return {"error": "Transaction ID is required for deletion"}

    return {
        "action": "delete_transaction",
        "transaction_id": transaction_id,
        "message": "I'll delete this transaction for you.",
        "requires_confirmation": True
    }

@tool_registry.register(
    name="categorize_transactions",
    description="Bulk categorize uncategorized transactions",
    schema=TRANSACTION_SCHEMAS["CATEGORIZE_TRANSACTIONS"],
    category="transactions"
)
async def categorize_transactions(context: Dict[str, Any]) -> Dict[str, Any]:
    """Categorize transactions using AI."""
    transactions = context.get("transactions", [])

    # Find uncategorized transactions
    uncategorized = [
        t for t in transactions
        if not t.get("category") or t.get("category").lower() == "uncategorized"
    ]

    if not uncategorized:
        return {
            "message": "All transactions are already categorized!",
            "categorized_count": 0
        }

    return {
        "action": "categorize_transactions",
        "transaction_count": len(uncategorized),
        "message": f"I'll categorize {len(uncategorized)} transactions for you using AI analysis."
    }


