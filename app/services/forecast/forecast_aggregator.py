"""
Aggregates transaction data for forecasting.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class ForecastAggregator:
    """
    Handles aggregation of financial transactions for forecasting purposes.

    The ForecastAggregator class is designed to process lists of transactions and aggregate
    them into meaningful and structured data. The aggregation includes daily summaries of
    income and expenses, statistical calculations, identification of recurring transactions,
    and spending breakdown by categories. The primary goal of this class is to support
    financial forecasting and analysis.

    Attributes
    ----------
    None.
    """

    async def aggregate_transactions(
            self, transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate transactions for forecasting.

        Args:
            transactions: List of transactions

        Returns:
            Aggregated data
        """
        if not transactions:
            return self._empty_aggregation()

        # Group by date
        daily_data = await self._group_by_date(transactions)

        # Calculate statistics
        stats = self._calculate_statistics(daily_data)

        # Find recurring transactions
        recurring = await self._find_recurring_transactions(transactions)

        # Analyze categories
        category_data = await self._analyze_categories(transactions)

        return {
            **stats,
            "daily_data": daily_data,
            "recurring_transactions": recurring,
            "category_breakdown": category_data,
            "count": len(transactions),
        }

    @staticmethod
    async def _group_by_date(
            transactions: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """Group transactions by date."""
        income_by_day = defaultdict(float)
        expenses_by_day = defaultdict(float)

        for txn in transactions:
            date_str = txn["date"]
            amount = float(txn["amount"])

            if amount > 0:
                income_by_day[date_str] += amount
            else:
                expenses_by_day[date_str] += abs(amount)

        return {
            "income_by_day": dict(income_by_day),
            "expenses_by_day": dict(expenses_by_day),
        }


    def _calculate_statistics(
            self, daily_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Calculate statistical measures."""
        income_values = list(daily_data["income_by_day"].values())
        expense_values = list(daily_data["expenses_by_day"].values())

        # Add zeros for days with no transactions
        all_dates = set(daily_data["income_by_day"].keys()) | set(
            daily_data["expenses_by_day"].keys()
        )
        days_count = len(all_dates)

        return {
            "avg_income": sum(income_values) / days_count if days_count > 0 else 0,
            "avg_expenses": sum(expense_values) / days_count if days_count > 0 else 0,
            "std_income": self._calculate_std(income_values),
            "std_expenses": self._calculate_std(expense_values),
            "income_by_day": daily_data["income_by_day"],
            "expenses_by_day": daily_data["expenses_by_day"],
        }

    async def _find_recurring_transactions(
            self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify recurring transactions."""
        # Group by description and amount
        transaction_groups = defaultdict(list)

        for txn in transactions:
            # Fix: Handle None description properly
            description = txn.get("description")
            if description is None or not isinstance(description, str):
                description = ""
            description = description.lower()

            key = (description, round(float(txn["amount"]), 2))
            transaction_groups[key].append(txn)

        recurring = []

        for (description, amount), group in transaction_groups.items():
            if len(group) >= 2:  # At least 2 occurrences
                # Analyze frequency
                dates = [datetime.strptime(t["date"], "%Y-%m-%d").date() for t in group]
                dates.sort()

                frequency = self._detect_frequency(dates)
                if frequency:
                    recurring.append(
                        {
                            "description": description,
                            "amount": amount,
                            "frequency": frequency,
                            "type": "income" if amount > 0 else "expense",
                            "occurrences": len(group),
                        }
                    )

        return recurring

    @staticmethod
    async def _analyze_categories(
        transactions: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze spending by category."""
        category_totals = defaultdict(float)
        category_counts = defaultdict(int)

        for txn in transactions:
            category = txn.get("category", "Uncategorized")
            amount = abs(float(txn["amount"]))

            category_totals[category] += amount
            category_counts[category] += 1

        return {
            category: {
                "total": total,
                "average": total / category_counts[category],
                "count": category_counts[category],
            }
            for category, total in category_totals.items()
        }

    @staticmethod
    def _calculate_std(values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    @staticmethod
    def _detect_frequency(dates: List[datetime.date]) -> Optional[str]:
        """Detect transaction frequency pattern."""
        if len(dates) < 2:
            return None

        # Calculate intervals
        intervals = []
        for i in range(1, len(dates)):
            interval = (dates[i] - dates[i - 1]).days
            intervals.append(interval)

        # Detect pattern
        avg_interval = sum(intervals) / len(intervals)

        if 25 <= avg_interval <= 35:
            return "monthly"
        elif 6 <= avg_interval <= 8:
            return "weekly"
        elif avg_interval <= 2:
            return "daily"

        return None

    @staticmethod
    def _empty_aggregation() -> Dict[str, Any]:
        """Return empty aggregation result."""
        return {
            "avg_income": 0,
            "avg_expenses": 0,
            "std_income": 0,
            "std_expenses": 0,
            "income_by_day": {},
            "expenses_by_day": {},
            "recurring_transactions": [],
            "category_breakdown": {},
            "count": 0,
        }
