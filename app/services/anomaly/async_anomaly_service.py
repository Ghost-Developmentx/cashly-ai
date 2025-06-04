"""
Async anomaly detection service for financial transactions.
"""

import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime, timedelta

from .anomaly_detector import AnomalyDetector
from .anomaly_classifier import AnomalyClassifier

logger = logging.getLogger(__name__)


class AsyncAnomalyService:
    """Async service for detecting transaction anomalies."""

    def __init__(self):
        self.detector = AnomalyDetector()
        self.classifier = AnomalyClassifier()
        self.default_threshold = 2.5  # Standard deviations

    async def detect_anomalies(
            self,
            user_id: str,
            transactions: List[Dict[str, Any]],
            threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Detect anomalous transactions.

        Args:
            user_id: User identifier
            transactions: Transaction history
            threshold: Anomaly threshold (optional)

        Returns:
            Anomaly detection results
        """
        try:
            if not transactions:
                return self._empty_anomaly_response()

            threshold = threshold or self.default_threshold

            # Detect anomalies
            anomalies = await self.detector.detect_anomalies(transactions, threshold)

            # Classify anomaly types
            classified_anomalies = await self.classifier.classify_anomalies(
                anomalies, transactions
            )

            # Generate summary
            summary = self._generate_summary(classified_anomalies, len(transactions))

            return {
                "anomalies": classified_anomalies,
                "summary": summary,
                "threshold": threshold,
                "total_transactions": len(transactions),
                "detection_method": "statistical_analysis",
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"error": f"Failed to detect anomalies: {str(e)}", "anomalies": []}

    @staticmethod
    def _generate_summary(
            anomalies: List[Dict[str, Any]], total_transactions: int
    ) -> Dict[str, Any]:
        """Generate anomaly detection summary matching AnomalySummary schema."""
        if not anomalies:
            return {
                "total_transactions": total_transactions,
                "anomalies_detected": 0,
                "anomaly_rate": 0.0,
                "by_type": {},
                "by_severity": {},
                "highest_risk_categories": [],
            }

        # Count by type
        by_type = {}
        by_severity = {}
        categories_risk = {}

        for anomaly in anomalies:
            # Count by anomaly type
            anomaly_type = anomaly.get("anomaly_type", "unknown")
            by_type[anomaly_type] = by_type.get(anomaly_type, 0) + 1

            # Count by severity
            severity = anomaly.get("severity", "medium")
            by_severity[severity] = by_severity.get(severity, 0) + 1

            # Track category risk
            transaction = anomaly.get("transaction", {})
            category = transaction.get("category", "Other")
            risk_score = anomaly.get("risk_score", 1)

            if category not in categories_risk:
                categories_risk[category] = {"count": 0, "total_risk": 0}
            categories_risk[category]["count"] += 1
            categories_risk[category]["total_risk"] += risk_score

        # Find highest risk categories
        highest_risk_categories = []
        if categories_risk:
            sorted_categories = sorted(
                categories_risk.items(),
                key=lambda x: x[1]["total_risk"] / x[1]["count"],
                reverse=True
            )
            highest_risk_categories = [cat[0] for cat in sorted_categories[:3]]

        return {
            "total_transactions": total_transactions,
            "anomalies_detected": len(anomalies),
            "anomaly_rate": round((len(anomalies) / total_transactions * 100), 2),
            "by_type": by_type,
            "by_severity": by_severity,
            "highest_risk_categories": highest_risk_categories,
        }

    @staticmethod
    def _determine_risk_level(avg_risk: float, count: int) -> str:
        """Determine overall risk level."""
        if count == 0:
            return "low"
        elif avg_risk > 7 or count > 10:
            return "high"
        elif avg_risk > 4 or count > 5:
            return "medium"
        else:
            return "low"

    @staticmethod
    def _empty_anomaly_response() -> Dict[str, Any]:
        """Return empty anomaly response."""
        return {
            "message": "No transaction data available for anomaly detection",
            "anomalies": [],
            "summary": {
                "total_transactions": 0,
                "anomalies_detected": 0,
                "anomaly_rate": 0.0,
                "by_type": {},
                "by_severity": {},
                "highest_risk_categories": [],
            },
        }
