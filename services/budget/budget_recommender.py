"""
Generates budget recommendations.
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class BudgetRecommender:
    """
    BudgetRecommender class.

    Provides functionalities to generate actionable recommendations for improving
    personal financial management based on spending analysis and budget allocations.
    The recommendations aim to address problem areas, optimize savings, and monitor
    spending trends.

    Methods
    -------
    generate_recommendations(spending_analysis, budget_allocations)
        Asynchronously generates specific budget recommendations to encourage
        financial well-being.
    """

    @staticmethod
    async def generate_recommendations(
        spending_analysis: Dict[str, Any], budget_allocations: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Generate budget recommendations.

        Args:
            spending_analysis: Current spending analysis
            budget_allocations: Recommended budget allocations

        Returns:
            List of recommendations
        """
        recommendations = []

        # Check problem areas
        for problem in spending_analysis.get("problem_areas", []):
            recommendations.append(
                {
                    "type": "reduce_spending",
                    "category": problem["category"],
                    "priority": "high",
                    "message": f"Consider reducing {problem['category']} spending by "
                    f"${problem['potential_savings']:.2f}/month",
                    "impact": problem["potential_savings"],
                }
            )

        # Check savings rate
        total_budget = sum(budget_allocations.values())
        savings_allocation = budget_allocations.get("Savings", 0)
        savings_rate = (
            (savings_allocation / total_budget * 100) if total_budget > 0 else 0
        )

        if savings_rate < 10:
            recommendations.append(
                {
                    "type": "increase_savings",
                    "priority": "high",
                    "message": f"Your savings rate is {savings_rate:.1f}%. "
                    f"Aim for at least 10-20% of income",
                    "impact": total_budget * 0.1 - savings_allocation,
                }
            )

        # Check spending trends
        trends = spending_analysis.get("trends", {})
        if trends.get("direction") == "increasing":
            recommendations.append(
                {
                    "type": "monitor_spending",
                    "priority": "medium",
                    "message": f"Your spending has increased by {trends['change_percent']:.1f}% "
                    f"recently. Review your expenses",
                    "impact": trends.get("recent_period_avg", 0)
                    - trends.get("first_period_avg", 0),
                }
            )

        # Add general tips
        if not recommendations:
            recommendations.append(
                {
                    "type": "general",
                    "priority": "low",
                    "message": "Your budget looks healthy! Keep tracking your expenses",
                    "impact": 0,
                }
            )

        # Sort by priority and impact
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(
            key=lambda x: (priority_order.get(x["priority"], 3), -x["impact"])
        )

        return recommendations[:5]  # Top 5 recommendations
