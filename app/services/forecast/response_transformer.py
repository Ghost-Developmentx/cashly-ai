"""
Transform forecast data to match frontend expectations.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime

class ForecastResponseTransformer:
    """Transform backend forecast data to frontend format."""

    @staticmethod
    def transform_to_frontend_format(
            forecast_data: Dict[str, Any],
            historical_transactions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Transform backend forecast to match frontend ForecastData interface.

        Args:
            forecast_data: Raw forecast from calculator
            historical_transactions: Historical data for actual values

        Returns:
            Frontend-compatible forecast data
        """
        # Transform daily predictions to data points
        data_points = ForecastResponseTransformer._transform_daily_predictions(
            forecast_data["daily_forecast"],
            historical_transactions
        )

        # Calculate trend
        trend = ForecastResponseTransformer._calculate_trend(
            forecast_data["daily_forecast"]
        )

        # Build summary
        summary = {
            "totalProjected": forecast_data["summary"]["projected_net"],
            "averageDaily": ForecastResponseTransformer._calculate_daily_average(
                forecast_data["daily_forecast"]
            ),
            "trend": trend,
            "confidenceScore": forecast_data["summary"]["confidence_score"],
            "periodDays": forecast_data["forecast_days"]
        }

        return {
            "id": f"forecast-{datetime.now().timestamp()}",
            "title": f"{forecast_data['forecast_days']}-Day Cash Flow Forecast",
            "dataPoints": data_points,
            "summary": summary,
            "generatedAt": datetime.now().isoformat()
        }

    @staticmethod
    def _transform_daily_predictions(
            daily_predictions: List[Dict[str, Any]],
            historical_data: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Transform daily predictions to frontend format."""
        data_points = []

        # Add historical data points if available
        if historical_data:
            # Group historical by date and sum
            historical_by_date = {}
            for txn in historical_data:
                date = txn["date"]
                if date not in historical_by_date:
                    historical_by_date[date] = 0
                historical_by_date[date] += txn["amount"]

            # Add last 7 days of historical data
            for date, amount in sorted(historical_by_date.items())[-7:]:
                data_points.append({
                    "date": date,
                    "actual": amount,
                    "predicted": amount,  # Same as actual for historical
                    "confidence": 1.0
                })

        # Add predicted data points
        for prediction in daily_predictions:
            data_points.append({
                "date": prediction["date"],
                "predicted": prediction["net_change"],  # Use net change
                "confidence": prediction["confidence"]
            })

        return data_points

    @staticmethod
    def _calculate_trend(predictions: List[Dict[str, Any]]) -> str:
        """Calculate overall trend from predictions."""
        if not predictions:
            return "stable"

        # Compare the first week average to last week average
        first_week = predictions[:7] if len(predictions) >= 7 else predictions
        last_week = predictions[-7:] if len(predictions) >= 7 else predictions

        first_avg = sum(p["net_change"] for p in first_week) / len(first_week)
        last_avg = sum(p["net_change"] for p in last_week) / len(last_week)

        # 10% threshold for trend detection
        if last_avg > first_avg * 1.1:
            return "up"
        elif last_avg < first_avg * 0.9:
            return "down"
        else:
            return "stable"

    @staticmethod
    def _calculate_daily_average(predictions: List[Dict[str, Any]]) -> float:
        """Calculate average daily net change."""
        if not predictions:
            return 0.0

        total = sum(p["net_change"] for p in predictions)
        return round(total / len(predictions), 2)

    @staticmethod
    def add_scenario_comparisons(
            base_forecast: Dict[str, Any],
            optimistic_forecast: Dict[str, Any],
            pessimistic_forecast: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge multiple scenarios into single response."""
        # Start with base forecast
        result = base_forecast.copy()

        # Add optimistic/pessimistic values to each data point
        for i, point in enumerate(result["dataPoints"]):
            if i < len(optimistic_forecast["dataPoints"]):
                point["optimistic"] = optimistic_forecast["dataPoints"][i]["predicted"]
            if i < len(pessimistic_forecast["dataPoints"]):
                point["pessimistic"] = pessimistic_forecast["dataPoints"][i]["predicted"]

        # Add scenarios object
        result["scenarios"] = {
            "base": base_forecast["dataPoints"],
            "adjusted": None  # Will be filled when adjustments applied
        }

        return result