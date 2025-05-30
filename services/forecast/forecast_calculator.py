"""
Core forecast calculation logic.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class ForecastCalculator:
    """
    A class for calculating financial forecasts.

    This class provides methods to analyze historical financial data and generate
    forecasts for future income and expenses. It is designed to identify patterns
    and seasonal trends from historical data, generate daily predictions for a
    specified period, and calculate confidence scores for the forecast results.

    Attributes
    ----------
    min_history_days : int
        The minimum number of historical days required for creating a forecast.
    confidence_threshold : float
        The threshold confidence level required for considering forecasts reliable.
    """

    def __init__(self):
        self.min_history_days = 14
        self.confidence_threshold = 0.7

    async def calculate_forecast(
        self, historical_data: Dict[str, Any], forecast_days: int
    ) -> Dict[str, Any]:
        """
        Calculate forecast based on historical data.

        Args:
            historical_data: Aggregated historical data
            forecast_days: Number of days to forecast

        Returns:
            Forecast results
        """
        # Extract patterns
        patterns = await self._extract_patterns(historical_data)

        # Generate daily predictions
        daily_predictions = await self._generate_daily_predictions(
            patterns, forecast_days
        )

        # Calculate summary metrics
        summary = self._calculate_summary(daily_predictions)

        # Calculate confidence score
        confidence = self._calculate_confidence(historical_data)

        return {
            "daily_predictions": daily_predictions,
            "patterns": patterns,
            "confidence": confidence,
            **summary,
        }

    async def _extract_patterns(
        self, historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract spending and income patterns."""
        return {
            "daily_income": {
                "mean": historical_data["avg_income"],
                "std": historical_data.get("std_income", 0),
                "pattern": self._detect_pattern(
                    historical_data.get("income_by_day", {})
                ),
            },
            "daily_expenses": {
                "mean": historical_data["avg_expenses"],
                "std": historical_data.get("std_expenses", 0),
                "pattern": self._detect_pattern(
                    historical_data.get("expenses_by_day", {})
                ),
            },
            "recurring": historical_data.get("recurring_transactions", []),
            "seasonality": self._detect_seasonality(historical_data),
        }

    async def _generate_daily_predictions(
        self, patterns: Dict[str, Any], forecast_days: int
    ) -> List[Dict[str, Any]]:
        """Generate daily forecast predictions."""
        predictions = []
        current_date = datetime.now().date()

        for day in range(forecast_days):
            forecast_date = current_date + timedelta(days=day + 1)

            # Base predictions on patterns
            income = self._predict_daily_amount(patterns["daily_income"], forecast_date)
            expenses = self._predict_daily_amount(
                patterns["daily_expenses"], forecast_date
            )

            # Add recurring transactions
            income += self._get_recurring_amount(
                patterns["recurring"], forecast_date, "income"
            )
            expenses += self._get_recurring_amount(
                patterns["recurring"], forecast_date, "expense"
            )

            predictions.append(
                {
                    "date": forecast_date.isoformat(),
                    "predicted_income": round(income, 2),
                    "predicted_expenses": round(expenses, 2),
                    "net_change": round(income - expenses, 2),
                    "confidence": self._get_daily_confidence(patterns, day),
                }
            )

        return predictions

    @staticmethod
    def _predict_daily_amount(
        pattern_data: Dict[str, Any], date: datetime.date
    ) -> float:
        """Predict amount for a specific date."""
        base_amount = pattern_data["mean"]

        # Apply day-of-week pattern
        day_name = date.strftime("%A")
        day_multiplier = pattern_data["pattern"].get(day_name, 1.0)

        # Add some randomness based on historical std
        std = pattern_data["std"]
        if std > 0:
            variation = np.random.normal(0, std * 0.3)
            base_amount += variation

        return max(0, base_amount * day_multiplier)

    def _get_recurring_amount(
        self,
        recurring: List[Dict[str, Any]],
        date: datetime.date,
        transaction_type: str,
    ) -> float:
        """Calculate recurring transactions for date."""
        total = 0

        for item in recurring:
            if item["type"] != transaction_type:
                continue

            # Check if transaction occurs on this date
            if self._is_recurring_date(item, date):
                total += abs(item["amount"])

        return total

    @staticmethod
    def _is_recurring_date(recurring_item: Dict[str, Any], date: datetime.date) -> bool:
        """Check if recurring transaction occurs on date."""
        frequency = recurring_item.get("frequency", "monthly")

        if frequency == "daily":
            return True
        elif frequency == "weekly":
            return date.weekday() == recurring_item.get("day_of_week", 0)
        elif frequency == "monthly":
            return date.day == recurring_item.get("day_of_month", 1)

        return False

    @staticmethod
    def _detect_pattern(daily_data: Dict[str, float]) -> Dict[str, float]:
        """Detect day-of-week patterns."""
        if not daily_data:
            return {}

        # Calculate average for each day
        day_totals = {}
        day_counts = {}

        for date_str, amount in daily_data.items():
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            day_name = date.strftime("%A")

            if day_name not in day_totals:
                day_totals[day_name] = 0
                day_counts[day_name] = 0

            day_totals[day_name] += amount
            day_counts[day_name] += 1

        # Calculate multipliers
        overall_avg = sum(day_totals.values()) / sum(day_counts.values())

        pattern = {}
        for day_name in day_totals:
            day_avg = day_totals[day_name] / day_counts[day_name]
            pattern[day_name] = day_avg / overall_avg if overall_avg > 0 else 1.0

        return pattern

    @staticmethod
    def _detect_seasonality(historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect seasonal patterns."""
        # Simplified seasonality detection
        return {"detected": False, "pattern": "none"}

    @staticmethod
    def _calculate_summary(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate forecast summary."""
        total_income = sum(p["predicted_income"] for p in predictions)
        total_expenses = sum(p["predicted_expenses"] for p in predictions)

        return {
            "total_income": round(total_income, 2),
            "total_expenses": round(total_expenses, 2),
            "net_change": round(total_income - total_expenses, 2),
            "ending_balance": round(total_income - total_expenses, 2),  # Simplified
        }

    @staticmethod
    def _calculate_confidence(historical_data: Dict[str, Any]) -> float:
        """Calculate forecast confidence score."""
        # Base confidence on data quality
        factors = []

        # Factor 1: Amount of historical data
        data_points = historical_data.get("count", 0)
        data_factor = min(data_points / 100, 1.0)
        factors.append(data_factor)

        # Factor 2: Data consistency
        if historical_data.get("std_income", 0) > 0:
            income_cv = (
                historical_data["std_income"] / historical_data["avg_income"]
                if historical_data["avg_income"] > 0
                else 1
            )
            consistency_factor = max(0, 1 - income_cv)
            factors.append(consistency_factor)

        # Calculate weighted confidence
        confidence = sum(factors) / len(factors) if factors else 0.5
        return round(confidence, 2)

    @staticmethod
    def _get_daily_confidence(patterns: Dict[str, Any], days_ahead: int) -> float:
        """Get confidence for specific day."""
        # Confidence decreases with time
        base_confidence = patterns.get("confidence", 0.7)
        decay_rate = 0.02  # 2% per day

        confidence = base_confidence * (1 - decay_rate * days_ahead)
        return max(0.3, round(confidence, 2))
