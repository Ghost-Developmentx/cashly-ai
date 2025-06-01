"""
Detects anomalies in financial transactions.
"""

import logging
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects statistical anomalies in transactions."""

    async def detect_anomalies(
        self, transactions: List[Dict[str, Any]], threshold: float = 2.5
    ) -> List[Dict[str, Any]]:
        """
        Detect anomalous transactions using statistical methods.

        Args:
            transactions: List of transactions
            threshold: Standard deviation threshold

        Returns:
            List of anomalous transactions
        """
        if not transactions:
            return []

        anomalies = []

        # Detect amount anomalies
        amount_anomalies = await self._detect_amount_anomalies(transactions, threshold)
        anomalies.extend(amount_anomalies)

        # Detect frequency anomalies
        frequency_anomalies = await self._detect_frequency_anomalies(transactions)
        anomalies.extend(frequency_anomalies)

        # Detect category anomalies
        category_anomalies = await self._detect_category_anomalies(transactions)
        anomalies.extend(category_anomalies)

        # Remove duplicates
        unique_anomalies = self._deduplicate_anomalies(anomalies)

        return unique_anomalies

    async def _detect_amount_anomalies(
        self, transactions: List[Dict[str, Any]], threshold: float
    ) -> List[Dict[str, Any]]:
        """Detect transactions with anomalous amounts."""
        # Separate by transaction type
        expenses = [t for t in transactions if float(t["amount"]) < 0]
        income = [t for t in transactions if float(t["amount"]) > 0]

        anomalies = []

        # Detect expense anomalies
        if len(expenses) > 10:
            expense_amounts = [abs(float(t["amount"])) for t in expenses]
            expense_anomalies = self._find_statistical_outliers(
                expenses, expense_amounts, threshold, "expense"
            )
            anomalies.extend(expense_anomalies)

        # Detect income anomalies
        if len(income) > 5:
            income_amounts = [float(t["amount"]) for t in income]
            income_anomalies = self._find_statistical_outliers(
                income, income_amounts, threshold, "income"
            )
            anomalies.extend(income_anomalies)

        return anomalies

    def _find_statistical_outliers(
        self,
        transactions: List[Dict[str, Any]],
        amounts: List[float],
        threshold: float,
        transaction_type: str,
    ) -> List[Dict[str, Any]]:
        """Find statistical outliers using z-score method."""
        if not amounts:
            return []

        # Calculate statistics
        mean = np.mean(amounts)
        std = np.std(amounts)

        if std == 0:
            return []

        anomalies = []

        for i, (txn, amount) in enumerate(zip(transactions, amounts)):
            z_score = abs((amount - mean) / std)

            if z_score > threshold:
                anomalies.append(
                    {
                        "transaction": txn,
                        "anomaly_type": f"unusual_{transaction_type}_amount",
                        "severity": self._calculate_severity(z_score),
                        "z_score": round(z_score, 2),
                        "expected_range": {
                            "min": round(max(0, mean - threshold * std), 2),
                            "max": round(mean + threshold * std, 2),
                        },
                        "actual_amount": amount,
                        "risk_score": self._calculate_risk_score(z_score, amount),
                    }
                )

        return anomalies

    async def _detect_frequency_anomalies(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect unusual transaction frequency patterns."""
        # Group by date
        daily_counts = {}

        for txn in transactions:
            date = txn["date"]
            if date not in daily_counts:
                daily_counts[date] = []
            daily_counts[date].append(txn)

        if len(daily_counts) < 7:
            return []

        # Calculate daily statistics
        counts = [len(txns) for txns in daily_counts.values()]
        mean_count = np.mean(counts)
        std_count = np.std(counts)

        if std_count == 0:
            return []

        anomalies = []
        threshold = 2.0

        for date, txns in daily_counts.items():
            count = len(txns)
            z_score = abs((count - mean_count) / std_count)

            if z_score > threshold and count > mean_count + threshold * std_count:
                anomalies.append(
                    {
                        "transaction": {
                            "date": date,
                            "description": f"{count} transactions on this day",
                            "amount": sum(float(t["amount"]) for t in txns),
                        },
                        "anomaly_type": "high_transaction_frequency",
                        "severity": self._calculate_severity(z_score),
                        "transaction_count": count,
                        "expected_count": round(mean_count, 1),
                        "risk_score": self._calculate_risk_score(z_score, count),
                    }
                )

        return anomalies

    async def _detect_category_anomalies(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect unusual spending in categories."""
        # Group by category
        category_spending = {}

        for txn in transactions:
            if float(txn["amount"]) < 0:  # Expenses only
                category = txn.get("category", "Other")
                if category not in category_spending:
                    category_spending[category] = []
                category_spending[category].append(abs(float(txn["amount"])))

        anomalies = []

        for category, amounts in category_spending.items():
            if len(amounts) > 5:
                mean = np.mean(amounts)
                std = np.std(amounts)

                if std > 0:
                    # Find outliers in this category
                    for i, amount in enumerate(amounts):
                        z_score = abs((amount - mean) / std)

                        if z_score > 2.5:
                            # Find the transaction
                            matching_txn = next(
                                (
                                    t
                                    for t in transactions
                                    if t.get("category") == category
                                    and abs(float(t["amount"])) == amount
                                ),
                                None,
                            )

                            if matching_txn:
                                anomalies.append(
                                    {
                                        "transaction": matching_txn,
                                        "anomaly_type": "category_outlier",
                                        "severity": self._calculate_severity(z_score),
                                        "category": category,
                                        "expected_range": {
                                            "min": round(max(0, mean - 2.5 * std), 2),
                                            "max": round(mean + 2.5 * std, 2),
                                        },
                                        "risk_score": self._calculate_risk_score(
                                            z_score, amount
                                        ),
                                    }
                                )

        return anomalies

    @staticmethod
    def _calculate_severity(z_score: float) -> str:
        """Calculate anomaly severity based on z-score."""
        if z_score > 4:
            return "critical"
        elif z_score > 3:
            return "high"
        elif z_score > 2.5:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _calculate_risk_score(z_score: float, amount: float) -> float:
        """Calculate risk score (0-10) based on statistical deviation and amount."""
        # Normalize z-score to 0-5 range
        z_component = min(z_score / 2, 5)

        # Add amount component (larger amounts = higher risk)
        amount_component = min(amount / 1000, 5)  # Cap at $1000

        risk_score = (z_component + amount_component) / 2
        return round(min(risk_score, 10), 1)

    @staticmethod
    def _deduplicate_anomalies(anomalies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate anomalies."""
        seen = set()
        unique = []

        for anomaly in anomalies:
            txn = anomaly["transaction"]
            key = (
                txn.get("date", ""),
                txn.get("description", ""),
                txn.get("amount", 0),
            )

            if key not in seen:
                seen.add(key)
                unique.append(anomaly)

        return unique
