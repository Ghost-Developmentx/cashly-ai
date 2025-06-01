"""
Detects patterns in financial data.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class PatternDetector:
    """
    A utility class for detecting patterns in financial transaction data.

    This class provides methods to analyze transaction records for recurring transactions,
    spending spikes, day-of-week patterns, and seasonal patterns. It is designed to support
    applications in financial insights, budget planning, and anomaly detection.

    Attributes
    ----------
    No attributes are defined in this class.
    """

    async def detect_patterns(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect various patterns in transactions.

        Args:
            transactions: List of transactions

        Returns:
            List of detected patterns
        """
        patterns = []

        # Detect recurring transactions
        recurring = await self._detect_recurring_transactions(transactions)
        if recurring:
            patterns.extend(recurring)

        # Detect spending spikes
        spikes = await self._detect_spending_spikes(transactions)
        if spikes:
            patterns.extend(spikes)

        # Detect day-of-week patterns
        dow_patterns = await self._detect_day_of_week_patterns(transactions)
        if dow_patterns:
            patterns.extend(dow_patterns)

        # Detect seasonal patterns
        seasonal = await self._detect_seasonal_patterns(transactions)
        if seasonal:
            patterns.extend(seasonal)

        return patterns

    async def _detect_recurring_transactions(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect recurring transactions."""
        # Group by similar description and amount
        transaction_groups = defaultdict(list)

        for txn in transactions:
            # Create key from description and rounded amount
            key = (
                self._normalize_description(txn.get("description", "")),
                round(float(txn["amount"]), 0),  # Round to nearest dollar
            )
            transaction_groups[key].append(txn)

        patterns = []

        for (description, amount), group in transaction_groups.items():
            if len(group) >= 3:  # At least 3 occurrences
                dates = [datetime.strptime(t["date"], "%Y-%m-%d") for t in group]
                dates.sort()

                # Analyze intervals
                intervals = []
                for i in range(1, len(dates)):
                    intervals.append((dates[i] - dates[i - 1]).days)

                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    frequency = self._determine_frequency(avg_interval)

                    if frequency:
                        patterns.append(
                            {
                                "type": "recurring_transaction",
                                "description": description,
                                "amount": amount,
                                "frequency": frequency,
                                "occurrences": len(group),
                                "average_interval_days": round(avg_interval, 1),
                                "confidence": self._calculate_recurrence_confidence(
                                    intervals
                                ),
                            }
                        )

        return patterns

    @staticmethod
    async def _detect_spending_spikes(
        transactions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect unusual spending spikes."""
        # Group spending by day
        daily_spending = defaultdict(float)

        for txn in transactions:
            if float(txn["amount"]) < 0:  # Expenses only
                date = txn["date"]
                daily_spending[date] += abs(float(txn["amount"]))

        if not daily_spending:
            return []

        # Calculate statistics
        amounts = list(daily_spending.values())
        mean = sum(amounts) / len(amounts)

        if len(amounts) < 2:
            return []

        variance = sum((x - mean) ** 2 for x in amounts) / len(amounts)
        std_dev = variance**0.5

        # Find spikes (> 2 standard deviations)
        patterns = []
        threshold = mean + (2 * std_dev)

        for date, amount in daily_spending.items():
            if amount > threshold:
                patterns.append(
                    {
                        "type": "spending_spike",
                        "date": date,
                        "amount": round(amount, 2),
                        "average_daily": round(mean, 2),
                        "times_average": round(amount / mean, 1) if mean > 0 else 0,
                        "severity": (
                            "high" if amount > mean + (3 * std_dev) else "medium"
                        ),
                    }
                )

        return patterns

    @staticmethod
    async def _detect_day_of_week_patterns(
        transactions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect day-of-week spending patterns."""
        dow_spending = defaultdict(lambda: {"total": 0, "count": 0})

        for txn in transactions:
            if float(txn["amount"]) < 0:  # Expenses only
                date = datetime.strptime(txn["date"], "%Y-%m-%d")
                dow = date.strftime("%A")
                amount = abs(float(txn["amount"]))

                dow_spending[dow]["total"] += amount
                dow_spending[dow]["count"] += 1

        if not dow_spending:
            return []

        # Calculate averages
        dow_averages = {}
        for day, data in dow_spending.items():
            if data["count"] > 0:
                dow_averages[day] = data["total"] / data["count"]

        if not dow_averages:
            return []

        # Find patterns
        overall_avg = sum(dow_averages.values()) / len(dow_averages)
        patterns = []

        # Find high-spending days
        high_days = [
            day
            for day, avg in dow_averages.items()
            if avg > overall_avg * 1.3  # 30% above average
        ]

        if high_days:
            patterns.append(
                {
                    "type": "day_of_week_pattern",
                    "pattern": "high_spending_days",
                    "days": high_days,
                    "average_on_these_days": round(
                        sum(dow_averages[d] for d in high_days) / len(high_days), 2
                    ),
                    "overall_daily_average": round(overall_avg, 2),
                }
            )

        return patterns

    @staticmethod
    async def _detect_seasonal_patterns(
        transactions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Detect seasonal spending patterns."""
        # Need at least 6 months of data
        dates = [datetime.strptime(t["date"], "%Y-%m-%d") for t in transactions]

        if not dates:
            return []

        date_range = (max(dates) - min(dates)).days
        if date_range < 180:  # Less than 6 months
            return []

        # Group by month
        monthly_spending = defaultdict(lambda: {"total": 0, "count": 0})

        for txn in transactions:
            if float(txn["amount"]) < 0:  # Expenses only
                date = datetime.strptime(txn["date"], "%Y-%m-%d")
                month = date.strftime("%B")
                amount = abs(float(txn["amount"]))

                monthly_spending[month]["total"] += amount
                monthly_spending[month]["count"] += 1

        # Calculate monthly averages
        month_averages = {}
        for month, data in monthly_spending.items():
            if data["count"] > 0:
                month_averages[month] = data["total"] / data["count"]

        if len(month_averages) < 6:
            return []

        # Find seasonal patterns
        patterns = []
        overall_avg = sum(month_averages.values()) / len(month_averages)

        # High spending months
        high_months = [
            month
            for month, avg in month_averages.items()
            if avg > overall_avg * 1.25  # 25% above average
        ]

        if high_months:
            patterns.append(
                {
                    "type": "seasonal_pattern",
                    "pattern": "high_spending_months",
                    "months": high_months,
                    "average_in_these_months": round(
                        sum(month_averages[m] for m in high_months) / len(high_months),
                        2,
                    ),
                    "overall_monthly_average": round(overall_avg, 2),
                }
            )

        return patterns

    @staticmethod
    def _normalize_description(description: str) -> str:
        """Normalize transaction description."""
        # Remove numbers and special characters
        import re

        normalized = re.sub(r"[0-9#*]", "", description.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _determine_frequency(avg_interval: float) -> Optional[str]:
        """Determine recurrence frequency from average interval."""
        if 0.8 <= avg_interval <= 1.2:
            return "daily"
        elif 6 <= avg_interval <= 8:
            return "weekly"
        elif 13 <= avg_interval <= 16:
            return "bi-weekly"
        elif 28 <= avg_interval <= 32:
            return "monthly"
        elif 85 <= avg_interval <= 95:
            return "quarterly"
        elif 360 <= avg_interval <= 370:
            return "yearly"
        else:
            return None

    @staticmethod
    def _calculate_recurrence_confidence(intervals: List[int]) -> float:
        """Calculate confidence in recurrence pattern."""
        if not intervals:
            return 0.0

        # Calculate consistency of intervals
        mean = sum(intervals) / len(intervals)
        if mean == 0:
            return 0.0

        # Calculate coefficient of variation
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        std_dev = variance**0.5
        cv = std_dev / mean

        # Lower CV = higher confidence
        confidence = max(0, min(1, 1 - cv))
        return round(confidence, 2)
