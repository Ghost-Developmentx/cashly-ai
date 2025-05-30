"""
Controller for cash flow forecasting endpoints.
Now uses async forecast service.
"""

import asyncio
from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from services.forecast import AsyncForecastService


class ForecastController(BaseController):
    """Controller for forecast operations"""

    def __init__(self):
        super().__init__()
        self.forecast_service = AsyncForecastService()

    def forecast_cash_flow(self) -> Tuple[Dict[str, Any], int]:
        """Generate cash flow forecast"""

        def _handle():
            data = self.get_request_data()
            self.validate_required_fields(data, ["user_id", "transactions"])

            user_id = data.get("user_id")
            transactions = data.get("transactions", [])
            forecast_days = data.get("forecast_days", 30)

            # Run async service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.forecast_service.forecast_cash_flow(
                        user_id, transactions, forecast_days
                    )
                )
            finally:
                loop.close()

            return self.success_response(result)

        return self.handle_request(_handle)

    def forecast_cash_flow_scenario(self) -> Tuple[Dict[str, Any], int]:
        """Generate a scenario-based forecast"""

        def _handle():
            data = self.get_request_data()
            self.validate_required_fields(data, ["user_id", "transactions"])

            user_id = data.get("user_id")
            transactions = data.get("transactions", [])
            forecast_days = data.get("forecast_days", 30)
            adjustments = data.get("adjustments", {})

            # Run async service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.forecast_service.forecast_cash_flow_scenario(
                        user_id, transactions, forecast_days, adjustments
                    )
                )
            finally:
                loop.close()

            return self.success_response(result)

        return self.handle_request(_handle)
