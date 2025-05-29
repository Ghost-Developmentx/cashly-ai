"""
Simple fallback classifier for when embeddings fail.
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class FallbackClassifier:
    """Simple keyword-based fallback classifier."""

    def __init__(self):
        self.keyword_map = {
            "transactions": [
                "transaction",
                "payment",
                "expense",
                "income",
                "spent",
                "cost",
            ],
            "accounts": ["account", "balance", "bank", "connect", "link"],
            "invoices": ["invoice", "bill", "client", "payment", "send invoice"],
            "forecasting": ["forecast", "predict", "future", "projection", "cash flow"],
            "budgets": ["budget", "limit", "spending", "allocation"],
            "insights": ["analyze", "trend", "pattern", "insight", "report"],
            "bank_connection": ["connect bank", "link bank", "add bank", "plaid"],
            "payment_processing": ["stripe", "process payment", "accept payment"],
        }

    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify query using simple keyword matching.

        Args:
            query: User query

        Returns:
            Tuple of (intent, confidence)
        """
        query_lower = query.lower()

        # Check each intent's keywords
        for intent, keywords in self.keyword_map.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Longer keywords get higher confidence
                    confidence = 0.6 if len(keyword.split()) > 1 else 0.5
                    logger.info(f"Fallback matched '{keyword}' -> {intent}")
                    return intent, confidence

        # Default to general
        return "general", 0.4
