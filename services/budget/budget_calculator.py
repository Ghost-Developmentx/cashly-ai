"""
Budget calculation logic.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class BudgetCalculator:
    """
    Class for budget calculation and management using standard allocation methods.

    This class provides tools to compute budget allocations, evaluate income and
    spending patterns, and derive insights such as savings rate and financial
    distribution. It serves as a framework for implementing budgeting principles
    like the 50/30/20 rule while allowing adjustments based on actual expenses.

    Attributes
    ----------
    standard_allocations : dict
        A dictionary defining standard budget percentages based on categories.
        Categories include 'Housing', 'Transportation', 'Food', 'Utilities',
        'Insurance', 'Healthcare', 'Entertainment', 'Personal', 'Savings', and
        'Other'.
    """

    def __init__(self):
        # Standard budget percentages (50/30/20 rule as baseline)
        self.standard_allocations = {
            "Housing": 0.28,
            "Transportation": 0.15,
            "Food": 0.12,
            "Utilities": 0.05,
            "Insurance": 0.10,
            "Healthcare": 0.05,
            "Entertainment": 0.05,
            "Personal": 0.05,
            "Savings": 0.10,
            "Other": 0.05,
        }

    @staticmethod
    async def calculate_monthly_income(transactions: List[Dict[str, Any]]) -> float:
        """Calculate average monthly income from transactions."""
        # Group income by month
        monthly_income = {}

        for txn in transactions:
            if float(txn["amount"]) > 0:  # Income
                date = datetime.strptime(txn["date"], "%Y-%m-%d")
                month_key = f"{date.year}-{date.month:02d}"

                if month_key not in monthly_income:
                    monthly_income[month_key] = 0
                monthly_income[month_key] += float(txn["amount"])

        if not monthly_income:
            return 0.0

        # Calculate average
        total_income = sum(monthly_income.values())
        months = len(monthly_income)

        return round(total_income / months, 2)

    async def calculate_budget(
        self, spending_analysis: Dict[str, Any], monthly_income: float
    ) -> Dict[str, float]:
        """
        Calculate budget allocations based on income and spending patterns.

        Args:
            spending_analysis: Current spending analysis
            monthly_income: Monthly income amount

        Returns:
            Budget allocations by category
        """
        if monthly_income <= 0:
            return {}

        allocations = {}

        # Start with standard allocations
        for category, percentage in self.standard_allocations.items():
            allocations[category] = round(monthly_income * percentage, 2)

        # Adjust based on actual spending patterns
        current_spending = spending_analysis.get("category_averages", {})

        for category, current_amount in current_spending.items():
            if category in allocations:
                # Blend standard and actual (70% standard, 30% actual)
                standard = allocations[category]
                allocations[category] = round(standard * 0.7 + current_amount * 0.3, 2)
            else:
                # New category not in standard
                allocations[category] = round(
                    current_amount * 0.9, 2
                )  # 10% reduction target

        # Ensure total doesn't exceed income
        total_allocated = sum(allocations.values())
        if total_allocated > monthly_income:
            # Scale down proportionally
            scale_factor = monthly_income / total_allocated * 0.95  # 5% buffer
            for category in allocations:
                allocations[category] = round(allocations[category] * scale_factor, 2)

        return allocations

    @staticmethod
    async def calculate_savings_rate(
        monthly_income: float, monthly_expenses: float
    ) -> float:
        """Calculate savings rate as percentage."""
        if monthly_income <= 0:
            return 0.0

        savings = monthly_income - monthly_expenses
        rate = (savings / monthly_income) * 100

        return round(max(0, rate), 1)
