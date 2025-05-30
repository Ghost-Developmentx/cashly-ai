"""
Analyzes spending patterns for budget generation.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class BudgetAnalyzer:
    """
    Analyzes and evaluates spending patterns from financial transactions.

    The `BudgetAnalyzer` class processes a list of transaction records to derive meaningful
    insights such as category totals, category averages, trends, and problem areas in
    spending. This data can help individuals or businesses better understand their spending
    habits and identify areas for financial improvement.

    Attributes
    ----------
    None
    """

    async def analyze_spending(
        self, transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze spending patterns from transactions.

        Args:
            transactions: List of transactions

        Returns:
            Spending analysis results
        """
        if not transactions:
            return self._empty_analysis()

        # Calculate date range
        dates = [datetime.strptime(t["date"], "%Y-%m-%d").date() for t in transactions]
        min_date = min(dates)
        max_date = max(dates)
        days_span = (max_date - min_date).days + 1
        months_span = max(1, days_span / 30)

        # Aggregate by category
        category_totals = defaultdict(float)
        category_counts = defaultdict(int)
        monthly_totals = defaultdict(float)

        for txn in transactions:
            if float(txn["amount"]) < 0:  # Expenses only
                amount = abs(float(txn["amount"]))
                category = txn.get("category", "Other")

                category_totals[category] += amount
                category_counts[category] += 1

                # Track monthly
                date = datetime.strptime(txn["date"], "%Y-%m-%d")
                month_key = f"{date.year}-{date.month:02d}"
                monthly_totals[month_key] += amount

        # Calculate averages
        category_averages = {
            cat: round(total / months_span, 2) for cat, total in category_totals.items()
        }

        # Find trends
        trends = await self._analyze_trends(monthly_totals)

        # Identify problem areas
        problem_areas = self._identify_problem_areas(
            category_averages, sum(category_averages.values())
        )

        return {
            "category_totals": dict(category_totals),
            "category_averages": category_averages,
            "monthly_totals": dict(monthly_totals),
            "trends": trends,
            "problem_areas": problem_areas,
            "period": f"{min_date} to {max_date}",
            "months_analyzed": round(months_span, 1),
        }

    @staticmethod
    async def _analyze_trends(monthly_totals: Dict[str, float]) -> Dict[str, Any]:
        """Analyze spending trends."""
        if len(monthly_totals) < 2:
            return {"direction": "stable", "change_percent": 0}

        # Sort by month
        sorted_months = sorted(monthly_totals.items())

        # Compare first half to second half
        mid_point = len(sorted_months) // 2
        first_half = sorted_months[:mid_point]
        second_half = sorted_months[mid_point:]

        first_avg = sum(m[1] for m in first_half) / len(first_half)
        second_avg = sum(m[1] for m in second_half) / len(second_half)

        if first_avg > 0:
            change_percent = ((second_avg - first_avg) / first_avg) * 100
        else:
            change_percent = 0

        direction = (
            "increasing"
            if change_percent > 5
            else ("decreasing" if change_percent < -5 else "stable")
        )

        return {
            "direction": direction,
            "change_percent": round(change_percent, 1),
            "first_period_avg": round(first_avg, 2),
            "recent_period_avg": round(second_avg, 2),
        }

    @staticmethod
    def _identify_problem_areas(
        category_averages: Dict[str, float], total_spending: float
    ) -> List[Dict[str, Any]]:
        """Identify categories with high spending."""
        if total_spending == 0:
            return []

        problem_areas = []

        # Define healthy spending percentages
        healthy_percentages = {
            "Housing": 0.30,
            "Food": 0.15,
            "Transportation": 0.15,
            "Entertainment": 0.10,
        }

        for category, amount in category_averages.items():
            percentage = amount / total_spending

            # Check against healthy percentages
            healthy_pct = healthy_percentages.get(category, 0.10)

            if percentage > healthy_pct * 1.2:  # 20% over healthy
                problem_areas.append(
                    {
                        "category": category,
                        "current_percentage": round(percentage * 100, 1),
                        "recommended_percentage": round(healthy_pct * 100, 1),
                        "monthly_amount": round(amount, 2),
                        "potential_savings": round(
                            amount - (total_spending * healthy_pct), 2
                        ),
                    }
                )

        # Sort by potential savings
        problem_areas.sort(key=lambda x: x["potential_savings"], reverse=True)

        return problem_areas[:5]  # Top 5 problem areas

    @staticmethod
    def _empty_analysis() -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "category_totals": {},
            "category_averages": {},
            "monthly_totals": {},
            "trends": {"direction": "stable", "change_percent": 0},
            "problem_areas": [],
            "period": "No data",
            "months_analyzed": 0,
        }
