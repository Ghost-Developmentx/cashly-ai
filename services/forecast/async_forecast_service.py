"""
Async cash flow forecasting service.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from .forecast_calculator import ForecastCalculator
from .forecast_aggregator import ForecastAggregator
from .scenario_analyzer import ScenarioAnalyzer

logger = logging.getLogger(__name__)


class AsyncForecastService:
    """
    Provide asynchronous cash flow forecasting services.

    This class offers methods for generating cash flow forecasts, including baseline
    forecasts based on historical transactions and scenario-based forecasts
    incorporating user-defined adjustments. It integrates with auxiliary services
    for aggregation, calculations, and scenario analysis.

    Attributes
    ----------
    calculator : ForecastCalculator
        Instance responsible for calculating forecasts.
    aggregator : ForecastAggregator
        Instance responsible for aggregating transaction data.
    scenario_analyzer : ScenarioAnalyzer
        Instance responsible for analyzing and applying scenario adjustments.
    """

    def __init__(self):
        self.calculator = ForecastCalculator()
        self.aggregator = ForecastAggregator()
        self.scenario_analyzer = ScenarioAnalyzer()

    async def forecast_cash_flow(
        self, user_id: str, transactions: List[Dict[str, Any]], forecast_days: int = 30
    ) -> Dict[str, Any]:
        """
        Generate cash flow forecast.

        Args:
            user_id: User identifier
            transactions: Historical transactions
            forecast_days: Days to forecast

        Returns:
            Forecast results
        """
        try:
            if not transactions:
                return self._empty_forecast_response(forecast_days)

            # Aggregate historical data
            historical_data = await self.aggregator.aggregate_transactions(transactions)

            # Calculate forecast
            forecast = await self.calculator.calculate_forecast(
                historical_data, forecast_days
            )

            # Prepare response
            return self._format_forecast_response(
                forecast, historical_data, forecast_days
            )

        except Exception as e:
            logger.error(f"Forecast generation failed: {e}")
            return {
                "error": f"Failed to generate forecast: {str(e)}",
                "forecast_days": forecast_days,
            }

    async def forecast_cash_flow_scenario(
        self,
        user_id: str,
        transactions: List[Dict[str, Any]],
        forecast_days: int = 30,
        adjustments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate forecast with scenario adjustments.

        Args:
            user_id: User identifier
            transactions: Historical transactions
            forecast_days: Days to forecast
            adjustments: Scenario adjustments

        Returns:
            Scenario forecast results
        """
        try:
            # Get base forecast
            base_forecast = await self.forecast_cash_flow(
                user_id, transactions, forecast_days
            )

            if "error" in base_forecast:
                return base_forecast

            # Apply scenario adjustments
            scenario_forecast = await self.scenario_analyzer.apply_adjustments(
                base_forecast, adjustments or {}
            )

            # Add scenario metadata
            scenario_forecast["scenario"] = {
                "adjustments_applied": adjustments or {},
                "base_forecast_included": True,
            }

            return scenario_forecast

        except Exception as e:
            logger.error(f"Scenario forecast failed: {e}")
            return {
                "error": f"Failed to generate scenario forecast: {str(e)}",
                "forecast_days": forecast_days,
            }

    @staticmethod
    def _format_forecast_response(
        forecast: Dict[str, Any],
        historical_data: Dict[str, Any],
        forecast_days: int,
    ) -> Dict[str, Any]:
        """Format forecast response."""
        return {
            "forecast_days": forecast_days,
            "start_date": datetime.now().date().isoformat(),
            "end_date": (datetime.now() + timedelta(days=forecast_days))
            .date()
            .isoformat(),
            "daily_forecast": forecast["daily_predictions"],
            "summary": {
                "projected_income": forecast["total_income"],
                "projected_expenses": forecast["total_expenses"],
                "projected_net": forecast["net_change"],
                "ending_balance": forecast["ending_balance"],
                "confidence_score": forecast["confidence"],
            },
            "historical_context": {
                "avg_daily_income": historical_data["avg_income"],
                "avg_daily_expenses": historical_data["avg_expenses"],
                "transaction_count": historical_data["count"],
            },
        }

    @staticmethod
    def _empty_forecast_response(forecast_days: int) -> Dict[str, Any]:
        """Return empty forecast response."""
        return {
            "forecast_days": forecast_days,
            "message": "No historical data available for forecasting",
            "daily_forecast": [],
            "summary": {
                "projected_income": 0,
                "projected_expenses": 0,
                "projected_net": 0,
                "ending_balance": 0,
                "confidence_score": 0,
            },
        }
