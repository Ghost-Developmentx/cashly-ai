"""
Analyzes financial trends.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

from app.api.v1.schemas.insights import TrendDirection

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Analyze financial trends and insights based on transaction data.

    This class provides methods to analyze spending and income trends
    over time. It processes transaction data and calculates various
    metrics such as monthly totals, trends, volatility, stability,
    and categorization of spending or income sources. The results can
    help users identify patterns and gain insights into their financial
    behavior.

    Attributes
    ----------
    No specific attributes are defined for this class as all computations
    and utilities are encapsulated within its methods.
    """

    async def analyze_spending_trends(
        self, transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze spending trends over time."""
        # Group by month
        monthly_spending = defaultdict(float)
        category_spending = defaultdict(lambda: defaultdict(float))

        for txn in transactions:
            if float(txn["amount"]) < 0:  # Expenses
                amount = abs(float(txn["amount"]))
                date = datetime.strptime(txn["date"], "%Y-%m-%d")
                month_key = f"{date.year}-{date.month:02d}"
                category = txn.get("category", "Other")

                monthly_spending[month_key] += amount
                category_spending[month_key][category] += amount

        if not monthly_spending:
            return self._empty_trend_result()

        # Calculate trend
        trend_data = self._calculate_trend(monthly_spending)

        # Find top categories
        top_categories = self._find_top_categories(category_spending)

        # Analyze volatility
        volatility = self._calculate_volatility(list(monthly_spending.values()))

        return {
            "monthly_data": dict(monthly_spending),
            "monthly_average": trend_data["average"],
            "trend_percentage": trend_data["trend_percentage"],
            "trend_direction": trend_data["direction"],
            "volatility": volatility,
            "top_categories": top_categories,
            "highest_month": trend_data["highest_month"],
            "lowest_month": trend_data["lowest_month"],
        }

    async def analyze_income_trends(
        self, transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze income trends over time."""
        # Group by month
        monthly_income = defaultdict(float)
        income_sources = defaultdict(lambda: defaultdict(float))

        for txn in transactions:
            if float(txn["amount"]) > 0:  # Income
                amount = float(txn["amount"])
                date = datetime.strptime(txn["date"], "%Y-%m-%d")
                month_key = f"{date.year}-{date.month:02d}"
                source = self._categorize_income_source(txn.get("description", ""))

                monthly_income[month_key] += amount
                income_sources[month_key][source] += amount

        if not monthly_income:
            return self._empty_trend_result()

        # Calculate trend
        trend_data = self._calculate_trend(monthly_income)

        # Analyze stability
        stability_score = self._calculate_stability(list(monthly_income.values()))

        return {
            "monthly_data": dict(monthly_income),
            "monthly_average": trend_data["average"],
            "trend_percentage": trend_data["trend_percentage"],
            "trend_direction": trend_data["direction"],
            "stability_score": stability_score,
            "income_sources": self._summarize_income_sources(income_sources),
            "highest_month": trend_data["highest_month"],
            "lowest_month": trend_data["lowest_month"],
        }

    @staticmethod
    def _calculate_trend(monthly_data: Dict[str, float]) -> Dict[str, Any]:
        """Calculate trend from monthly data."""
        if not monthly_data:
            return {
                "average": 0,
                "trend_percentage": 0,
                "direction": "stable",
                "highest_month": None,
                "lowest_month": None,
            }

        sorted_months = sorted(monthly_data.items())
        values = [v for _, v in sorted_months]

        # Calculate average
        average = sum(values) / len(values)

        # Calculate trend (compare first third to last third)
        third = len(values) // 3
        if third > 0:
            early_avg = sum(values[:third]) / third
            recent_avg = sum(values[-third:]) / len(values[-third:])

            if early_avg > 0:
                trend_percentage = ((recent_avg - early_avg) / early_avg) * 100
            else:
                trend_percentage = 0
        else:
            trend_percentage = 0

        # Determine direction
        if trend_percentage > 5:
            direction = "increasing"
        elif trend_percentage < -5:
            direction = "decreasing"
        else:
            direction = "stable"

        # Find highest and lowest
        highest_month = max(sorted_months, key=lambda x: x[1])
        lowest_month = min(sorted_months, key=lambda x: x[1])

        return {
            "average": round(average, 2),
            "trend_percentage": round(trend_percentage, 1),
            "direction": direction,
            "highest_month": highest_month[0],
            "lowest_month": lowest_month[0],
        }

    @staticmethod
    def _find_top_categories(
        category_spending: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Find top spending categories."""
        category_totals = defaultdict(float)

        for month_data in category_spending.values():
            for category, amount in month_data.items():
                category_totals[category] += amount

        # Sort by total spending
        sorted_categories = sorted(
            category_totals.items(), key=lambda x: x[1], reverse=True
        )

        # Return top 5 with percentages
        total = sum(category_totals.values())
        top_categories = []

        for category, amount in sorted_categories[:5]:
            percentage = (amount / total * 100) if total > 0 else 0
            top_categories.append(
                {
                    "category": category,
                    "total_amount": round(amount, 2),
                    "percentage": round(percentage, 1),
                }
            )

        return top_categories

    @staticmethod
    def _calculate_volatility(values: List[float]) -> float:
        """Calculate spending volatility (coefficient of variation)."""
        if not values or len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance**0.5

        return round((std_dev / mean) * 100, 1)

    @staticmethod
    def _calculate_stability(values: List[float]) -> float:
        """Calculate income stability score (0-100)."""
        if not values or len(values) < 2:
            return 100.0

        # Lower volatility = higher stability
        mean = sum(values) / len(values)
        if mean == 0:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance**0.5
        cv = std_dev / mean

        # Convert to 0-100 score (lower CV = higher score)
        stability = max(0, min(100, (1 - cv) * 100))

        return round(stability, 1)

    @staticmethod
    def _categorize_income_source(description: str) -> str:
        """Categorize income source from description."""
        if not description:  # Guard against None/empty
            return "Other Income"

        description_lower = description.lower()

        if "salary" in description_lower or "payroll" in description_lower:
            return "Salary"
        elif "interest" in description_lower:
            return "Interest"
        elif "dividend" in description_lower:
            return "Dividends"
        elif "refund" in description_lower:
            return "Refunds"
        elif "transfer" in description_lower:
            return "Transfers"
        else:
            return "Other Income"


    @staticmethod
    def _summarize_income_sources(
        income_sources: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        """Summarize income sources."""
        source_totals = defaultdict(float)

        for month_data in income_sources.values():
            for source, amount in month_data.items():
                source_totals[source] += amount

        total = sum(source_totals.values())
        sources = []

        for source, amount in source_totals.items():
            percentage = (amount / total * 100) if total > 0 else 0
            sources.append(
                {
                    "source": source,
                    "total_amount": round(amount, 2),
                    "percentage": round(percentage, 1),
                }
            )

        return sorted(sources, key=lambda x: x["total_amount"], reverse=True)

    @staticmethod
    def _empty_trend_result() -> Dict[str, Any]:
        """Return complete empty trend result."""
        return {
            "monthly_data": {},
            "monthly_average": 0.0,
            "trend_percentage": 0.0,
            "trend_direction": "stable",
            "volatility": 0.0,
            "top_categories": [],
            "highest_month": {"month": "", "value": 0.0},
            "lowest_month": {"month": "", "value": 0.0},
            "volatility_score": 0.0,
            "direction": TrendDirection.STABLE,
            "change_percentage": 0.0,
            "average_monthly": 0.0
        }
