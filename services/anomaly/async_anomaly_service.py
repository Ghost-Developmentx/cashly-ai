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
                "threshold_used": threshold,
                "total_transactions": len(transactions),
                "detection_method": "statistical_analysis",
            }

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return {"error": f"Failed to detect anomalies: {str(e)}", "anomalies": []}

    def _generate_summary(
        self, anomalies: List[Dict[str, Any]], total_transactions: int
    ) -> Dict[str, Any]:
        """Generate anomaly detection summary."""
        if not anomalies:
            return {
                "anomaly_count": 0,
                "anomaly_rate": 0,
                "categories": {},
                "risk_level": "low",
            }

        # Count by category
        categories = {}
        total_risk_score = 0

        for anomaly in anomalies:
            category = anomaly.get("anomaly_type", "unknown")
            categories[category] = categories.get(category, 0) + 1
            total_risk_score += anomaly.get("risk_score", 1)

        # Calculate risk level
        avg_risk = total_risk_score / len(anomalies)
        risk_level = self._determine_risk_level(avg_risk, len(anomalies))

        return {
            "anomaly_count": len(anomalies),
            "anomaly_rate": (len(anomalies) / total_transactions * 100),
            "categories": categories,
            "risk_level": risk_level,
            "average_risk_score": round(avg_risk, 2),
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
            "summary": {"anomaly_count": 0, "anomaly_rate": 0, "risk_level": "low"},
        }
