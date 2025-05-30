"""
Async handlers for analytics and forecasting tools.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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
        # Initialize services (these would need to be converted to async too)
        from services.forecast_service import ForecastService
        from services.budget_service import BudgetService
        from services.insight_service import InsightService
        from services.anomaly_service import AnomalyService

        self.forecast_service = ForecastService()
        self.budget_service = BudgetService()
        self.insight_service = InsightService()
        self.anomaly_service = AnomalyService()

    async def forecast_cash_flow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cash flow forecast."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]

        days = tool_args.get("days", 30)
        adjustments = tool_args.get("adjustments", {})

        try:
            # Note: These services would need async versions
            # For now, we'll use sync calls in an executor
            import asyncio

            loop = asyncio.get_event_loop()

            if adjustments:
                result = await loop.run_in_executor(
                    None,
                    self.forecast_service.forecast_cash_flow_scenario,
                    user_id,
                    transactions,
                    days,
                    adjustments,
                )
            else:
                result = await loop.run_in_executor(
                    None,
                    self.forecast_service.forecast_cash_flow,
                    user_id,
                    transactions,
                    days,
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

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None, self.insight_service.analyze_trends, user_id, transactions, period
            )

            return result

        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {"error": f"Failed to analyze trends: {str(e)}"}

    async def detect_anomalies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalous transactions."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]
        threshold = tool_args.get("threshold")

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None,
                self.anomaly_service.detect_anomalies,
                user_id,
                transactions,
                threshold,
            )

            return result

        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {"error": f"Failed to detect anomalies: {str(e)}"}

    async def generate_budget(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate budget recommendations."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]
        monthly_income = tool_args.get("monthly_income")

        try:
            import asyncio

            loop = asyncio.get_event_loop()

            result = await loop.run_in_executor(
                None,
                self.budget_service.generate_budget,
                user_id,
                transactions,
                monthly_income,
            )

            return result

        except Exception as e:
            logger.error(f"Budget generation error: {e}")
            return {"error": f"Failed to generate budget: {str(e)}"}

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
