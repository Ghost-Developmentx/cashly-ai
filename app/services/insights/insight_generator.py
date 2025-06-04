"""
Generates actionable financial insights.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class InsightGenerator:
    """
    Generates actionable financial insights by analyzing spending trends, income trends,
    and detected financial patterns. Its purpose is to assist users in identifying areas
    for financial improvement, tracking recurring expenses, and enhancing financial health.

    The class provides methods to create insights derived from financial data such as
    spending and income trends, spending categories, and recurring patterns. Insights
    are prioritized and sorted to offer the most critical recommendations first.

    Attributes
    ----------
    No specific attributes defined for this class; however, it uses various static and
    asynchronous methods to compute and organize insights.
    """

    PRIORITY_ORDER = {"high": 1, "medium": 2, "low": 3}

    async def generate_insights(
            self,
            spending_trends: Dict[str, Any],
            income_trends: Dict[str, Any],
            patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate actionable insights.

        Args:
            spending_trends: Spending trend analysis
            income_trends: Income trend analysis
            patterns: Detected patterns

        Returns:
            List of insights
        """
        insights = []

        # Trend-based insights
        trend_insights = await self._generate_trend_insights(
            spending_trends, income_trends
        )
        insights.extend(trend_insights)

        # Pattern-based insights
        pattern_insights = await self._generate_pattern_insights(patterns)
        insights.extend(pattern_insights)

        # Category-based insights
        category_insights = await self._generate_category_insights(spending_trends)
        insights.extend(category_insights)

        # Financial health insights
        health_insights = await self._generate_health_insights(
            spending_trends, income_trends
        )
        insights.extend(health_insights)

        # Sort by priority using consistent string-to-integer mapping
        insights.sort(key=lambda x: self.PRIORITY_ORDER.get(x.get("priority", "low"), 999))

        return insights[:10]  # Top 10 insights

    @staticmethod
    async def _generate_trend_insights(
            spending_trends: Dict[str, Any], income_trends: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate insights from trends."""
        insights = []

        # Spending trend insights
        spending_direction = spending_trends.get("trend_direction", "stable")
        spending_change = spending_trends.get("trend_percentage", 0)

        if spending_direction == "increasing" and spending_change > 10:
            insights.append(
                {
                    "type": "spending_trend",
                    "title": "Spending Increase Alert",
                    "description": f"Your spending has increased by {spending_change:.1f}% recently",
                    "impact": "Review your recent expenses to identify areas for reduction",
                    "priority": "high" if spending_change > 20 else "medium",
                    "action_required": True,
                    "metadata": {
                        "change_percentage": spending_change,
                        "direction": spending_direction,
                        "severity": "high" if spending_change > 20 else "medium"
                    }
                }
            )

        # Income trend insights
        income_direction = income_trends.get("trend_direction", "stable")
        income_change = income_trends.get("trend_percentage", 0)

        if income_direction == "decreasing" and income_change < -10:
            insights.append(
                {
                    "type": "income_pattern",
                    "title": "Income Decrease Alert",
                    "description": f"Your income has decreased by {abs(income_change):.1f}% recently",
                    "impact": "Consider diversifying income sources or reducing expenses",
                    "priority": "high",
                    "action_required": True,
                    "metadata": {
                        "change_percentage": income_change,
                        "direction": income_direction,
                        "severity": "high"
                    }
                }
            )


        return insights

    @staticmethod
    async def _generate_pattern_insights(
            patterns: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate insights from detected patterns."""
        insights = []

        # Recurring transaction insights
        recurring = [p for p in patterns if p.get("type") == "recurring_transaction"]
        if recurring:
            total_recurring = sum(abs(p.get("amount", 0)) for p in recurring)
            insights.append(
                {
                    "type": "recurring_detection",
                    "title": f"{len(recurring)} Recurring Transactions Detected",
                    "description": f"You have ${total_recurring:.2f} in recurring monthly expenses",
                    "impact": "Review these subscriptions to ensure they're all necessary",
                    "priority": "medium",
                    "action_required": False,
                    "metadata": {
                        "count": len(recurring),
                        "total": total_recurring,
                        "severity": "medium"
                    },
                }
            )

        # Spending spike insights
        spikes = [p for p in patterns if p.get("type") == "spending_spike"]
        if spikes:
            recent_spikes = [
                s
                for s in spikes
                if (datetime.now() - datetime.strptime(s.get("date", "1900-01-01"), "%Y-%m-%d")).days < 30
            ]
            if recent_spikes:
                insights.append(
                    {
                        "type": "spending_trend",
                        "title": "Recent Spending Spikes",
                        "description": f"You had {len(recent_spikes)} unusual spending days recently",
                        "impact": "Review these high-spending days to avoid future surprises",
                        "priority": "medium",
                        "action_required": False,
                        "metadata": {
                            "spike_count": len(recent_spikes),
                            "severity": "medium"
                        }
                    }
                )

        return insights

    @staticmethod
    async def _generate_category_insights(
            spending_trends: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate category-based insights."""
        insights = []

        top_categories = spending_trends.get("top_categories", [])
        if top_categories:
            top_cat = top_categories[0]
            if top_cat.get("percentage", 0) > 40:
                insights.append(
                    {
                        "type": "category_concentration",
                        "title": f"High Spending in {top_cat['category']}",
                        "description": f"{top_cat['percentage']:.1f}% of your spending is in {top_cat['category']}",
                        "impact": "Medium impact on budget planning",
                        "priority": "medium",
                        "action_required": False,
                        "metadata": {
                            "category": top_cat['category'],
                            "percentage": top_cat['percentage'],
                            "severity": "medium"
                        },
                    }
                )

        return insights

    @staticmethod
    async def _generate_health_insights(
            spending_trends: Dict[str, Any], income_trends: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate financial health insights."""
        insights = []

        avg_income = income_trends.get("monthly_average", 0)
        avg_spending = spending_trends.get("monthly_average", 0)

        if avg_income > 0:
            savings_rate = ((avg_income - avg_spending) / avg_income) * 100

            if savings_rate < 10:
                insights.append(
                    {
                        "type": "saving_opportunity",
                        "title": "Low Savings Rate",
                        "description": f"You're saving only {savings_rate:.1f}% of your income",
                        "impact": "Aim to save at least 20% of your income for better financial health",
                        "priority": "high" if savings_rate < 5 else "medium",
                        "action_required": True,
                        "metadata": {
                            "current_rate": savings_rate,
                            "target_rate": 20,
                            "severity": "high" if savings_rate < 5 else "medium"
                        }
                    }
                )

            elif savings_rate > 30:
                insights.append(
                    {
                        "type": "saving_opportunity",
                        "title": "Excellent Savings Rate!",
                        "description": f"You're saving {savings_rate:.1f}% of your income",
                        "impact": "Keep up the great work! Consider investing surplus savings",
                        "priority": "low",  # Changed from integer 4 to string "low"
                        "action_required": False,
                        "metadata": {
                            "current_rate": savings_rate,
                            "severity": "positive"
                        }
                    }
                )

        return insights
