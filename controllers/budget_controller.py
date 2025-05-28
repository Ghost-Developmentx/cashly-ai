"""
Controller for budget generation endpoints.
Handles budget-related HTTP requests and delegates to business services.
"""

from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from services.budget_service import BudgetService


class BudgetController(BaseController):
    """Controller for budget generation operations"""

    def __init__(self):
        super().__init__()
        self.budget_service = BudgetService()

    def generate_budget(self) -> Tuple[Dict[str, Any], int]:
        """
        Generate budget recommendations based on spending patterns

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [
                {"date": "2025-01-01", "amount": -50.00, "category": "groceries"},
                {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
                ...
            ],
            "income": 5000.00
        }

        Returns:
            Tuple of (response_dict, status_code)
        """

        def _handle():
            data = self.get_request_data()

            # Validate required fields
            self.validate_required_fields(data, ["user_id", "transactions"])

            # Extract parameters
            user_id = data.get("user_id")
            transactions = data.get("transactions", [])
            monthly_income = data.get("income", 0)

            # Validate transactions data
            if not transactions:
                raise ValueError("Transactions list cannot be empty")

            # Log request details
            self.logger.info(
                f"Generating budget recommendations for user {user_id} "
                f"with income ${monthly_income}"
            )

            # Delegate to service
            result = self.budget_service.generate_budget(
                user_id=user_id,
                transactions=transactions,
                monthly_income=monthly_income,
            )

            # Log result
            budget_categories = len(
                result.get("recommended_budget", {}).get("by_category", {})
            )
            self.logger.info(
                f"Budget recommendations generated for {budget_categories} categories"
            )

            return self.success_response(result)

        return self.handle_request(_handle)
