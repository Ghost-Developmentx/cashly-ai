"""
Classifies and enriches detected anomalies.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AnomalyClassifier:
    """Classifies anomalies by type and risk."""

    async def classify_anomalies(
        self, anomalies: List[Dict[str, Any]], all_transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Classify and enrich anomalies.

        Args:
            anomalies: Detected anomalies
            all_transactions: All transactions for context

        Returns:
            Classified anomalies
        """
        classified = []

        for anomaly in anomalies:
            # Enrich with classification
            enriched = await self._enrich_anomaly(anomaly, all_transactions)
            classified.append(enriched)

        # Sort by risk score
        classified.sort(key=lambda x: x.get("risk_score", 0), reverse=True)

        return classified

    async def _enrich_anomaly(
        self, anomaly: Dict[str, Any], all_transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Enrich anomaly with additional context."""
        enriched = anomaly.copy()
        txn = anomaly["transaction"]

        # Add timing context
        enriched["timing_context"] = self._get_timing_context(txn, all_transactions)

        # Add merchant context
        enriched["merchant_context"] = self._get_merchant_context(txn, all_transactions)

        # Add recommendation
        enriched["recommendation"] = self._get_recommendation(anomaly)

        # Add investigation priority
        enriched["investigation_priority"] = self._calculate_priority(anomaly)

        return enriched

    @staticmethod
    def _get_timing_context(
        transaction: Dict[str, Any], all_transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get timing context for anomaly."""
        txn_date = datetime.strptime(transaction["date"], "%Y-%m-%d")

        # Check if weekend
        is_weekend = txn_date.weekday() >= 5

        # Check if end of month
        is_month_end = txn_date.day >= 28

        # Check if holiday period (simplified)
        is_holiday = txn_date.month in [11, 12]  # November, December

        return {
            "is_weekend": is_weekend,
            "is_month_end": is_month_end,
            "is_holiday_period": is_holiday,
            "day_of_week": txn_date.strftime("%A"),
            "unusual_timing": is_weekend or is_month_end,
        }

    @staticmethod
    def _get_merchant_context(
        transaction: Dict[str, Any], all_transactions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get merchant context for anomaly."""
        description = transaction.get("description", "").lower()

        # Count previous transactions with same merchant
        similar_count = sum(
            1
            for t in all_transactions
            if description in t.get("description", "").lower()
        )

        # Check if first-time merchant
        is_new_merchant = similar_count == 1

        return {
            "is_new_merchant": is_new_merchant,
            "previous_transaction_count": similar_count - 1,
            "merchant_risk": "high" if is_new_merchant else "normal",
        }

    @staticmethod
    def _get_recommendation(anomaly: Dict[str, Any]) -> str:
        """Get recommendation based on anomaly type."""
        anomaly_type = anomaly.get("anomaly_type", "")
        severity = anomaly.get("severity", "medium")

        recommendations = {
            "unusual_expense_amount": {
                "critical": "Immediately review this transaction for potential fraud",
                "high": "Verify this large transaction with your bank",
                "medium": "Check if this transaction was authorized",
                "low": "Monitor similar transactions",
            },
            "unusual_income_amount": {
                "critical": "Verify this large deposit is legitimate",
                "high": "Confirm the source of this income",
                "medium": "Track this unusual income for tax purposes",
                "low": "No action needed",
            },
            "high_transaction_frequency": {
                "high": "Review all transactions from this day for duplicates",
                "medium": "Check for any unauthorized transactions",
                "low": "Monitor transaction frequency",
            },
            "category_outlier": {
                "critical": "Investigate this unusual spending immediately",
                "high": "Review recent purchases in this category",
                "medium": "Consider if this was a one-time expense",
                "low": "Track spending in this category",
            },
        }

        type_recommendations = recommendations.get(anomaly_type, {})
        return type_recommendations.get(
            severity, "Review this transaction for accuracy"
        )

    def _calculate_priority(self, anomaly: Dict[str, Any]) -> str:
        """Calculate investigation priority."""
        risk_score = anomaly.get("risk_score", 0)
        severity = anomaly.get("severity", "medium")

        # Check additional risk factors
        txn = anomaly["transaction"]
        amount = abs(float(txn.get("amount", 0)))
        is_recent = self._is_recent_transaction(txn["date"])

        # Calculate priority score
        priority_score = risk_score

        if severity == "critical":
            priority_score += 3
        elif severity == "high":
            priority_score += 2

        if amount > 500:
            priority_score += 2
        elif amount > 200:
            priority_score += 1

        if is_recent:
            priority_score += 1

        # Map to priority level
        if priority_score >= 8:
            return "urgent"
        elif priority_score >= 5:
            return "high"
        elif priority_score >= 3:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _is_recent_transaction(date_str: str) -> bool:
        """Check if transaction is recent (within 7 days)."""
        txn_date = datetime.strptime(date_str, "%Y-%m-%d")
        days_ago = (datetime.now() - txn_date).days
        return days_ago <= 7
