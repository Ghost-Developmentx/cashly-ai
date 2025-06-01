"""
Analyzes and applies forecast scenarios.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ScenarioAnalyzer:
    """
    Analyze and adjust financial forecast scenarios.

    This class provides capabilities to modify and analyze financial forecast scenarios by applying
    various adjustments, such as income or expense variations, category-specific changes, and more.
    Additionally, it recalculates summary projections based on daily forecasts and adjustments applied.

    Attributes
    ----------
    None
    """

    async def apply_adjustments(
        self, base_forecast: Dict[str, Any], adjustments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply scenario adjustments to forecast.

        Args:
            base_forecast: Base forecast results
            adjustments: Scenario adjustments

        Returns:
            Adjusted forecast
        """
        adjusted_forecast = base_forecast.copy()

        # Apply income adjustment
        if "income_adjustment" in adjustments:
            adjusted_forecast = await self._adjust_income(
                adjusted_forecast, adjustments["income_adjustment"]
            )

        # Apply expense adjustment
        if "expense_adjustment" in adjustments:
            adjusted_forecast = await self._adjust_expenses(
                adjusted_forecast, adjustments["expense_adjustment"]
            )

        # Apply category adjustments
        if "category_adjustments" in adjustments:
            adjusted_forecast = await self._adjust_categories(
                adjusted_forecast, adjustments["category_adjustments"]
            )

        # Recalculate summary
        adjusted_forecast["summary"] = self._recalculate_summary(
            adjusted_forecast["daily_forecast"]
        )

        return adjusted_forecast

    @staticmethod
    async def _adjust_income(
        forecast: Dict[str, Any], adjustment: float
    ) -> Dict[str, Any]:
        """Apply income adjustment."""
        daily_adjustment = adjustment / 30  # Distribute monthly adjustment

        for day in forecast["daily_forecast"]:
            day["predicted_income"] += daily_adjustment
            day["predicted_income"] = max(0, day["predicted_income"])
            day["net_change"] = day["predicted_income"] - day["predicted_expenses"]

        return forecast

    @staticmethod
    async def _adjust_expenses(
        forecast: Dict[str, Any], adjustment: float
    ) -> Dict[str, Any]:
        """Apply expense adjustment."""
        daily_adjustment = adjustment / 30  # Distribute monthly adjustment

        for day in forecast["daily_forecast"]:
            day["predicted_expenses"] += daily_adjustment
            day["predicted_expenses"] = max(0, day["predicted_expenses"])
            day["net_change"] = day["predicted_income"] - day["predicted_expenses"]

        return forecast

    async def _adjust_categories(
        self, forecast: Dict[str, Any], category_adjustments: Dict[str, float]
    ) -> Dict[str, Any]:
        """Apply category-specific adjustments."""
        # This is simplified - would need category breakdown in forecast
        total_adjustment = sum(category_adjustments.values())

        return await self._adjust_expenses(forecast, total_adjustment)

    @staticmethod
    def _recalculate_summary(daily_forecast: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Recalculate forecast summary."""
        total_income = sum(day["predicted_income"] for day in daily_forecast)
        total_expenses = sum(day["predicted_expenses"] for day in daily_forecast)

        return {
            "projected_income": round(total_income, 2),
            "projected_expenses": round(total_expenses, 2),
            "projected_net": round(total_income - total_expenses, 2),
            "ending_balance": round(total_income - total_expenses, 2),
        }
