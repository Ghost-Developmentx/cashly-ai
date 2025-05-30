"""
Async handlers for analytics and forecasting tools.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

from services.forecast.async_forecast_service import AsyncForecastService
from services.budget.async_budget_service import AsyncBudgetService
from services.insights.async_insight_service import AsyncInsightService
from services.anomaly.async_anomaly_service import AsyncAnomalyService


class AsyncAnalyticsHandlers:
    """
    Handles asynchronous analytics services including forecasting, trend analysis,
    anomaly detection, budget generation, and spending categorization.

    This class is designed to integrate with multiple analytics services like
    ForecastService, BudgetService, InsightService, and AnomalyService, enabling
    users to perform financial data analysis efficiently. Each method is structured
    to accommodate asynchronous workflows, allowing integration with web services
    or real-time data analysis pipelines.

    Attributes
    ----------
    forecast_service : ForecastService
        Provides methods to forecast cash flow and financial scenarios.
    budget_service : BudgetService
        Offers tools to generate budget recommendations.
    insight_service : InsightService
        Analyzes trends and insights in financial data.
    anomaly_service : AnomalyService
        Identifies anomalous transactions in financial datasets.
    """

    def __init__(self):
        # Initialize async services
        self.forecast_service = AsyncForecastService()
        self.budget_service = AsyncBudgetService()
        self.insight_service = AsyncInsightService()
        self.anomaly_service = AsyncAnomalyService()

    async def forecast_cash_flow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cash flow forecast."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]

        days = tool_args.get("days", 30)
        adjustments = tool_args.get("adjustments", {})

        try:
            if adjustments:
                result = await self.forecast_service.forecast_cash_flow_scenario(
                    user_id, transactions, days, adjustments
                )
            else:
                result = await self.forecast_service.forecast_cash_flow(
                    user_id, transactions, days
                )

            return result

        except Exception as e:
            logger.error(f"Forecast error: {e}")
            return {"error": f"Failed to generate forecast: {str(e)}"}

    async def analyze_trends(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial trends."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]
        period = tool_args.get("period", "3m")

        return await self.insight_service.analyze_trends(user_id, transactions, period)

    async def detect_anomalies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalous transactions."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]
        threshold = tool_args.get("threshold")

        return await self.anomaly_service.detect_anomalies(
            user_id, transactions, threshold
        )

    async def generate_budget(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate budget recommendations."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]
        monthly_income = tool_args.get("monthly_income")

        return await self.budget_service.generate_budget(
            user_id, transactions, monthly_income
        )

    async def calculate_category_spending(
        self, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate spending by category."""
        tool_args = context["tool_args"]
        transactions = context["transactions"]

        categories = tool_args.get("categories", [])
        start_date_str = tool_args.get("start_date")
        end_date_str = tool_args.get("end_date")

        # Calculate date range
        today = datetime.now().date()
        if start_date_str and end_date_str:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        else:
            start_date = today - timedelta(days=30)
            end_date = today

        # Filter transactions
        filtered_txns = await self._filter_by_date_range(
            transactions, start_date, end_date
        )

        # Filter by categories if specified
        if categories:
            filtered_txns = await self._filter_by_categories(filtered_txns, categories)

        # Calculate totals
        totals = await self._calculate_category_totals(filtered_txns)

        return {
            "time_period": f"{start_date} to {end_date}",
            "total_spending": sum(totals.values()),
            "category_breakdown": totals,
            "transaction_count": len(filtered_txns),
        }

    @staticmethod
    async def _filter_by_date_range(
        transactions: List[Dict[str, Any]],
        start_date: datetime.date,
        end_date: datetime.date,
    ) -> List[Dict[str, Any]]:
        """Filter transactions by date range."""
        filtered = []
        for txn in transactions:
            try:
                txn_date = datetime.strptime(txn["date"], "%Y-%m-%d").date()
                if start_date <= txn_date <= end_date:
                    filtered.append(txn)
            except ValueError:
                logger.warning(f"Invalid date format: {txn.get('date')}")
        return filtered

    @staticmethod
    async def _filter_by_categories(
        transactions: List[Dict[str, Any]], categories: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter transactions by categories."""
        categories_lower = [c.lower() for c in categories]
        return [
            txn
            for txn in transactions
            if any(
                cat in str(txn.get("category", "")).lower() for cat in categories_lower
            )
        ]

    @staticmethod
    async def _calculate_category_totals(
        transactions: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Calculate spending totals by category."""
        totals = {}
        for txn in transactions:
            if float(txn["amount"]) < 0:  # Only expenses
                category = txn.get("category", "Uncategorized")
                totals[category] = totals.get(category, 0) + abs(float(txn["amount"]))
        return totals
