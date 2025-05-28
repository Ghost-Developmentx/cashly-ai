"""
Controller for financial insights and trend analysis endpoints.
Handles insights-related HTTP requests and delegates to business services.
"""

from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from services.insight_service import InsightService


class InsightsController(BaseController):
    """Controller for financial insights and trend analysis operations"""

    def __init__(self):
        super().__init__()
        self.insight_service = InsightService()

    def analyze_trends(self) -> Tuple[Dict[str, Any], int]:
        """
        Analyze financial trends and patterns

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [
                {"date": "2025-01-01", "amount": -50.00, "category": "groceries"},
                {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
                ...
            ],
            "period": "3m"  # 1m, 3m, 6m, 1y
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
            period = data.get("period", "3m")

            # Validate transactions data
            if not transactions:
                raise ValueError("Transactions list cannot be empty")

            # Validate period
            valid_periods = ["1m", "3m", "6m", "1y"]
            if period not in valid_periods:
                raise ValueError(f"Period must be one of: {', '.join(valid_periods)}")

            # Log request details
            self.logger.info(
                f"Analyzing trends for user {user_id} over period {period}"
            )

            # Delegate to service
            result = self.insight_service.analyze_trends(
                user_id=user_id, transactions=transactions, period=period
            )

            # Log result
            insights_count = len(result.get("insights", []))
            self.logger.info(
                f"Trend analysis completed with {insights_count} insights generated"
            )

            return self.success_response(result)

        return self.handle_request(_handle)
