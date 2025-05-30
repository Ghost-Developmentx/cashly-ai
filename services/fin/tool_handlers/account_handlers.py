"""
Async handlers for account-related tools.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class AsyncAccountHandlers:
    """
    Handles asynchronous operations related to user bank accounts.

    This class manages operations such as retrieving account details, initiating
    connections to financial institutions (e.g., through Plaid), and disconnecting
    accounts. Each method operates asynchronously and is designed for scenarios
    requiring account management in a connected financial application.

    Attributes
    ----------
    rails_client : Any
        Client instance used for interacting with the Rails backend.
    """

    def __init__(self, rails_client):
        self.rails_client = rails_client

    @staticmethod
    async def get_user_accounts(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetches user account details and aggregates account-related information.

        This static method takes a context dictionary containing user details,
        validates and retrieves the "accounts" field, and calculates key metrics
        such as the total number of accounts, the total account balance, and
        whether the user has any accounts. The results are returned in a dictionary
        format.

        Parameters
        ----------
        context : Dict[str, Any]
            A dictionary encapsulating user contextual data. It must contain
            a key "user_context", which further includes a key "accounts"
            representing a list of accounts. Each account in the list should
            contain (optional) a "balance" field.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the following keys:
            - "account_count": The number of accounts (int).
            - "accounts": The list of account data provided in the input (List[Dict[str, Any]]).
            - "total_balance": The total sum of balances across all accounts (int or float).
            - "has_accounts": A boolean indicating whether the user has any accounts (bool).
        """
        user_context = context["user_context"]
        accounts = user_context.get("accounts", [])

        return {
            "account_count": len(accounts),
            "accounts": accounts,
            "total_balance": sum(acc.get("balance", 0) for acc in accounts),
            "has_accounts": len(accounts) > 0,
        }

    @staticmethod
    async def get_account_details(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves account details for a specific account based on the provided account ID
        from the context. The method searches through the user's context data to locate
        the account and retrieve relevant details such as balance, account type, name,
        and associated financial institution.

        Parameters
        ----------
        context : Dict[str, Any]
            A dictionary containing necessary context for finding account details. It
            must include "tool_args" with the specified "account_id" and "user_context"
            with a list of accounts.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the details of the account if found. The returned
            dictionary includes:
                - "account" : Dict[str, Any]
                    The matched account object.
                - "balance" : float
                    The account's balance or 0 if not specified.
                - "name" : str
                    The name of the account or "Unknown Account" if not specified.
                - "type" : str
                    The account type or "Unknown" if not specified.
                - "institution" : str
                    The institution name or "Unknown Bank" if not specified.

            If the account is not found, an error response is returned:
                - "error" : str
                    A message indicating the account with the given ID was not found.
        """
        tool_args = context["tool_args"]
        user_context = context["user_context"]

        account_id = tool_args.get("account_id")
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

    @staticmethod
    async def initiate_plaid_connection(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initiates a Plaid connection for a user by preparing necessary details and instructions.

        This method constructs a response containing information required to initiate the Plaid
        account connection process for a user. It utilizes the provided context to extract user-specific
        information and tool arguments. This response includes the user's ID, an optional institution
        preference, a message indicating the purpose of the action, and the next step to be taken
        in the Plaid connection process.

        Parameters
        ----------
        context : Dict[str, Any]
            A dictionary containing contextual information, including `tool_args`, which specifies
            additional arguments for the tool, and `user_id`, which identifies the user.

        Returns
        -------
        Dict[str, Any]
            A dictionary with details to guide the Plaid connection workflow, including:
            - `action`: the action to be initiated.
            - `user_id`: the ID of the user initiating the Plaid connection.
            - `institution_preference`: a preference regarding the financial institution, if provided.
            - `message`: a user-facing message about the action.
            - `next_step`: the next action step in the workflow.
        """
        tool_args = context["tool_args"]
        user_id = context["user_id"]

        return {
            "action": "initiate_plaid_connection",
            "user_id": user_id,
            "institution_preference": tool_args.get("institution_preference"),
            "message": "I'll help you connect your bank account securely through Plaid.",
            "next_step": "plaid_link_token",
        }

    @staticmethod
    async def disconnect_account(context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Disconnects an account based on the provided context.

        This static asynchronous method facilitates the disconnection of an account
        by utilizing context data that includes the account ID and user ID. The method
        returns information regarding the account disconnection action, alongside a
        confirmation message and a flag for requiring user confirmation.

        Parameters
        ----------
        context : Dict[str, Any]
            A dictionary containing the context for the operation. It must include
            "tool_args" (a dictionary that may contain "account_id") and "user_id"
            (identifier of the user performing the action).

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the following keys:
            - "action": The action performed (always "disconnect_account").
            - "account_id": The ID of the account to be disconnected.
            - "user_id": The ID of the user requesting the disconnection.
            - "message": A message to display on completion.
            - "requires_confirmation": A boolean indicating if user confirmation is
              required to finalize the operation.
        """
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        account_id = tool_args.get("account_id")

        return {
            "action": "disconnect_account",
            "account_id": account_id,
            "user_id": user_id,
            "message": f"I'll disconnect account {account_id} for you.",
            "requires_confirmation": True,
        }
