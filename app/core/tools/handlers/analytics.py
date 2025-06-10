import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from ..registry import tool_registry
from ..schemas import ANALYTICS_SCHEMAS
from app.services.forecast.async_forecast_service import AsyncForecastService
from app.services.budget.async_budget_service import AsyncBudgetService
from app.services.insights.async_insight_service import AsyncInsightService
from app.services.anomaly.async_anomaly_service import AsyncAnomalyService
from ..helpers.analytics_helpers import filter_by_date_range, calculate_category_totals, filter_by_categories

logger = logging.getLogger(__name__)

class AnalyticsHandlers:
    """
    Handles analytics requests.
    """

    def __init__(self):
        self.schemas = ANALYTICS_SCHEMAS
        self.tool_registry = tool_registry
        self.forecast_service = AsyncForecastService()
        self.budget_service = AsyncBudgetService()
        self.insight_service = AsyncInsightService()
        self.anomaly_service = AsyncAnomalyService()

    @tool_registry.register(
        name="forecast_cash_flow",
        description="Get forecast for a given date",
        schema=ANALYTICS_SCHEMAS["FORECAST_CASH_FLOW"],
        category="analytics"
    )
    async def forecast_cash_flow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cash flow forecast."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]

        days = tool_args.get("days", 30)
        adjustments = tool_args.get("adjustments", {})

        logger.info(f"Forecast: User {user_id}, Days: {days}, Transactions: {len(transactions)}")

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

    @tool_registry.register(
        name="analyze_trends",
        description="Analyze financial trends and patterns in transactions",
        schema=ANALYTICS_SCHEMAS["ANALYZE_TRENDS"],
        category="analytics"
    )
    async def analyze_trends(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial trends."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]
        period = tool_args.get("period", "3m")

        return await self.insight_service.analyze_trends(user_id, transactions, period)

    @tool_registry.register(
        name="detect_anomalies",
        description="Detect unusual or anomalous transactions",
        schema=ANALYTICS_SCHEMAS["DETECT_ANOMALIES"],
        category="analytics"
    )
    async def detect_anomalies(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalous transactions."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        transactions = context["transactions"]
        threshold = tool_args.get("threshold")

        return await self.anomaly_service.detect_anomalies(
            user_id, transactions, threshold
        )

    @tool_registry.register(
        name="generate_budget",
        description="Generate budget recommendations based on spending patterns",
        schema=ANALYTICS_SCHEMAS["GENERATE_BUDGET"],
        category="analytics"
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

    @tool_registry.register(
        name="calculate_category_spending",
        description="Calculate spending in specific categories over a time period",
        schema=ANALYTICS_SCHEMAS["CALCULATE_CATEGORY_SPENDING"],
        category="analytics"
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
        filtered_txns = await filter_by_date_range(
            transactions, start_date, end_date
        )

        # Filter by categories if specified
        if categories:
            filtered_txns = await filter_by_categories(filtered_txns, categories)

        # Calculate totals
        totals = await calculate_category_totals(filtered_txns)

        return {
            "time_period": f"{start_date} to {end_date}",
            "total_spending": sum(totals.values()),
            "category_breakdown": totals,
            "transaction_count": len(filtered_txns),
        }
