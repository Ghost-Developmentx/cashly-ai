"""
Controller for financial insights endpoints.
Now uses async insight service.
"""

import asyncio
from typing import Dict, Any, Tuple
from controllers.base_controller import BaseController
from services.insights import AsyncInsightService


class InsightsController(BaseController):
    """Controller for insights operations"""

    def __init__(self):
        super().__init__()
        self.insight_service = AsyncInsightService()

    def analyze_trends(self) -> Tuple[Dict[str, Any], int]:
        """
        Analyze financial trends and patterns.

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [...],
            "period": "3m"  # Options: 1m, 3m, 6m, 1y
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
            period = data.get("period", "3m")

            # Validate period
            valid_periods = ["1m", "3m", "6m", "1y"]
            if period not in valid_periods:
                return self.error_response(
                    f"Invalid period. Must be one of: {', '.join(valid_periods)}", 400
                )

            self.logger.info(f"Analyzing trends for user {user_id}, period: {period}")

            # Run async service
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self.insight_service.analyze_trends(user_id, transactions, period)
                )
            finally:
                loop.close()

            if "error" in result:
                self.logger.error(f"Trend analysis failed: {result['error']}")
                return self.error_response(result["error"], 400)

            self.logger.info(
                f"Trend analysis completed. Found {len(result.get('insights', []))} insights"
            )
            return self.success_response(result)

        return self.handle_request(_handle)

    def get_financial_summary(self) -> Tuple[Dict[str, Any], int]:
        """
        Get a comprehensive financial summary.

        Expected JSON input:
        {
            "user_id": "user_123",
            "transactions": [...],
            "include_insights": true
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
            include_insights = data.get("include_insights", True)

            # Run async service for trend analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                # Get 3-month trends by default
                trends = loop.run_until_complete(
                    self.insight_service.analyze_trends(user_id, transactions, "3m")
                )
            finally:
                loop.close()

            # Build summary
            summary = {
                "user_id": user_id,
                "summary": trends.get("summary", {}),
                "spending_trends": trends.get("spending_trends", {}),
                "income_trends": trends.get("income_trends", {}),
                "patterns": trends.get("patterns", []),
            }

            if include_insights:
                summary["insights"] = trends.get("insights", [])

            return self.success_response(summary)

        return self.handle_request(_handle)
