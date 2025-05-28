"""
Controller for cash flow forecasting endpoints.
Handles forecasting-related HTTP requests and delegates to business services.
"""

from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from services.forecast_service import ForecastService


class ForecastController(BaseController):
    """Controller for cash flow forecasting operations"""

    def __init__(self):
        super().__init__()
        self.forecast_service = ForecastService()

    def forecast_cash_flow(self) -> Tuple[Dict[str, Any], int]:
        """
        Generate cash flow forecast

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [
                {"date": "2025-01-01", "amount": 1000.00, "category": "income"},
                {"date": "2025-01-05", "amount": -200.00, "category": "utilities"},
                ...
            ],
            "forecast_days": 30
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
            forecast_days = data.get("forecast_days", 30)

            # Validate transactions data
            if not transactions:
                raise ValueError("Transactions list cannot be empty")

            # Log request details
            self.logger.info(
                f"Forecasting cash flow for user {user_id} for {forecast_days} days"
            )

            # Delegate to service
            result = self.forecast_service.forecast_cash_flow(
                user_id=user_id, transactions=transactions, forecast_days=forecast_days
            )

            # Log result
            forecast_count = len(result.get("forecast", []))
            self.logger.info(f"Forecast completed with {forecast_count} days predicted")

            return self.success_response(result)

        return self.handle_request(_handle)

    def forecast_cash_flow_scenario(self) -> Tuple[Dict[str, Any], int]:
        """
        Generate a cash flow forecast with scenario adjustments

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [...],
            "forecast_days": 30,
            "adjustments": {
                "category_adjustments": {"1": 500, "2": -200},
                "income_adjustment": 1000,
                "expense_adjustment": 500,
                "recurring_transactions": [...]
            }
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
            forecast_days = data.get("forecast_days", 30)
            adjustments = data.get("adjustments", {})

            # Validate transactions data
            if not transactions:
                raise ValueError("Transactions list cannot be empty")

            # Log request details
            self.logger.info(f"Generating scenario forecast for user {user_id}")

            # Delegate to service
            result = self.forecast_service.forecast_cash_flow_scenario(
                user_id=user_id,
                transactions=transactions,
                forecast_days=forecast_days,
                adjustments=adjustments,
            )

            self.logger.info("Scenario forecast completed")

            return self.success_response(result)

        return self.handle_request(_handle)
