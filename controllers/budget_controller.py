"""
Controller for budget generation endpoints.
Now uses async budget service.
"""

import asyncio
from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from services.budget import AsyncBudgetService


class BudgetController(BaseController):
    """Controller for budget operations"""

    def __init__(self):
        super().__init__()
        self.budget_service = AsyncBudgetService()

    def generate_budget(self) -> Tuple[Dict[str, Any], int]:
        """
        Generate budget recommendations based on spending patterns.

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [...],
            "monthly_income": 5000.00  # Optional
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id", "transactions"])

            user_id = data.get("user_id")
            transactions = data.get("transactions", [])
            monthly_income = data.get("monthly_income")

            self.logger.info(f"Generating budget for user {user_id}")

            # Run async service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.budget_service.generate_budget(
                        user_id, transactions, monthly_income
                    )
                )
            finally:
                loop.close()

            if "error" in result:
                self.logger.error(f"Budget generation failed: {result['error']}")
                return self.error_response(result["error"], 400)

            self.logger.info(f"Budget generated successfully for user {user_id}")
            return self.success_response(result)

        return self.handle_request(_handle)
