from typing import Any, Dict, Optional


def get_user_accounts(user_id: str, user_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve information about a user's connected bank accounts.

    Args:
        user_id: Unique user identifier
        user_context: Dictionary containing user data (including 'accounts')

    Returns:
        Dictionary with account stats and balance summary.
    """
    accounts = user_context.get("accounts", [])

    return {
        "account_count": len(accounts),
        "accounts": accounts,
        "total_balance": sum(acc.get("balance", 0) for acc in accounts),
        "has_accounts": len(accounts) > 0,
    }


def get_account_details(
    user_id: str, account_id: str, user_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Return detailed info for a specific account by ID.

    Args:
        user_id: Unique user identifier
        account_id: ID of the account to retrieve
        user_context: User profile data containing accounts

    Returns:
        Dictionary with account detail or error message.
    """
    accounts = user_context.get("accounts", [])
    account = next(
        (acc for acc in accounts if str(acc.get("id")) == str(account_id)), None
    )

    if not account:
        return {"error": f"Account with ID {account_id} not found"}

    return {
        "account": account,
        "balance": account.get("balance", 0),
        "name": account.get("name", "Unknown Account"),
        "type": account.get("account_type", "Unknown"),
        "institution": account.get("institution", "Unknown Bank"),
    }


def initiate_plaid_connection(
    user_id: str, institution_preference: Optional[str] = None
) -> Dict[str, Any]:
    """
    Simulate starting the Plaid link process.

    Args:
        user_id: The user's ID
        institution_preference: Optional string for preferred bank type

    Returns:
        Dictionary representing a UI action and Plaid step
    """
    return {
        "action": "initiate_plaid_connection",
        "user_id": user_id,
        "institution_preference": institution_preference,
        "message": "I'll help you connect your bank account securely through Plaid. This will allow me to provide better financial insights and track your spending automatically.",
        "next_step": "plaid_link_token",
    }


def disconnect_account(user_id: str, account_id: str) -> Dict[str, Any]:
    """
    Return a simulated response for disconnecting a bank account.

    Args:
        user_id: The user's ID
        account_id: The account to disconnect

    Returns:
        Dictionary containing the disconnect action and confirmation message
    """
    return {
        "action": "disconnect_account",
        "account_id": account_id,
        "user_id": user_id,
        "message": f"I'll disconnect account {account_id} for you. This will remove access to transaction data from this account.",
        "requires_confirmation": True,
    }
