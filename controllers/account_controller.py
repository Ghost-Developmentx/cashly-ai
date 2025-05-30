"""
Controller for account management endpoints.
Handles Plaid connections and account status.
"""

import asyncio
from typing import Dict, Any, Tuple
from datetime import datetime
from controllers.base_controller import BaseController
from services.fin.async_tool_registry import AsyncToolRegistry


class AccountController(BaseController):
    """Controller for account operations"""

    def __init__(self):
        super().__init__()
        self.tool_registry = AsyncToolRegistry()

    def get_account_status(self) -> Tuple[Dict[str, Any], int]:
        """
        Get user account status for Fin queries.

        Expected JSON input:
        {
            "user_id": "user_123",
            "user_context": {...}
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id", "user_context"])

            user_id = data.get("user_id")
            user_context = data.get("user_context", {})

            # Run async tool
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.tool_registry.execute(
                        tool_name="get_user_accounts",
                        tool_args={},
                        user_id=user_id,
                        transactions=[],
                        user_context=user_context,
                    )
                )
            finally:
                loop.close()

            if "error" in result:
                return self.error_response(result["error"], 400)

            # Add additional status info
            result["status"] = {
                "has_accounts": result.get("has_accounts", False),
                "account_count": result.get("account_count", 0),
                "last_updated": datetime.now().isoformat(),
            }

            return self.success_response(result)

        return self.handle_request(_handle)

    def initiate_account_connection(self) -> Tuple[Dict[str, Any], int]:
        """
        Initiate a Plaid connection process.

        Expected JSON input:
        {
            "user_id": "user_123",
            "institution_preference": "major_bank"  # Optional
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id"])

            user_id = data.get("user_id")
            institution_preference = data.get("institution_preference")

            # Run async tool
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.tool_registry.execute(
                        tool_name="initiate_plaid_connection",
                        tool_args={"institution_preference": institution_preference},
                        user_id=user_id,
                        transactions=[],
                        user_context={},
                    )
                )
            finally:
                loop.close()

            if "error" in result:
                return self.error_response(result["error"], 400)

            self.logger.info(f"Initiated Plaid connection for user {user_id}")
            return self.success_response(result)

        return self.handle_request(_handle)

    def disconnect_account(self) -> Tuple[Dict[str, Any], int]:
        """
        Disconnect a bank account.

        Expected JSON input:
        {
            "user_id": "user_123",
            "account_id": "acc_456"
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id", "account_id"])

            user_id = data.get("user_id")
            account_id = data.get("account_id")

            # Run async tool
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.tool_registry.execute(
                        tool_name="disconnect_account",
                        tool_args={"account_id": account_id},
                        user_id=user_id,
                        transactions=[],
                        user_context={},
                    )
                )
            finally:
                loop.close()

            if "error" in result:
                return self.error_response(result["error"], 400)

            self.logger.info(
                f"Initiated account disconnection for user {user_id}, "
                f"account {account_id}"
            )
            return self.success_response(result)

        return self.handle_request(_handle)
