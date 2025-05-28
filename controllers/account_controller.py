"""
Controller for account management endpoints.
Handles account-related HTTP requests and delegates to business services.
"""

from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController


class AccountController(BaseController):
    """Controller for account management operations"""

    def __init__(self):
        super().__init__()

    def get_account_status(self) -> Tuple[Dict[str, Any], int]:
        """
        Get user account status for Fin queries

        Expected JSON input:
        {
            "user_id": "user_123",
            "user_context": {
                "accounts": [...],
                "budgets": [...],
                ...
            }
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id"])

            # Extract parameters
            user_id = data.get("user_id")
            user_context = data.get("user_context", {})

            # Extract account information
            accounts = user_context.get("accounts", [])

            # Build response
            result = {
                "account_count": len(accounts),
                "accounts": accounts,
                "total_balance": sum(acc.get("balance", 0) for acc in accounts),
                "has_accounts": len(accounts) > 0,
                "plaid_connected": any(acc.get("plaid_account_id") for acc in accounts),
            }

            self.logger.info(
                f"Retrieved account status for user {user_id}: {len(accounts)} accounts"
            )

            return self.success_response(result)

        return self.handle_request(_handle)

    def initiate_account_connection(self) -> Tuple[Dict[str, Any], int]:
        """
        Initiate a Plaid connection process from Fin

        Expected JSON input:
        {
            "user_id": "user_123",
            "institution_preference": "major_bank" (optional)
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id"])

            # Extract parameters
            user_id = data.get("user_id")
            institution_preference = data.get("institution_preference")

            # Build connection initiation response
            result = {
                "action": "initiate_plaid_connection",
                "user_id": user_id,
                "institution_preference": institution_preference,
                "message": "I'll help you connect your bank account securely through Plaid.",
                "next_step": "plaid_link_token",
                "instructions": [
                    "Click the 'Connect Bank Account' button below",
                    "Select your bank from the list",
                    "Enter your online banking credentials",
                    "Select which accounts to connect",
                    "Complete the verification process",
                ],
            }

            self.logger.info(f"Initiated Plaid connection for user {user_id}")
            if institution_preference:
                self.logger.info(f"Institution preference: {institution_preference}")

            return self.success_response(result)

        return self.handle_request(_handle)
