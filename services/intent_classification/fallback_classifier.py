"""
Simple fallback classifier for when embeddings fail.
"""

import logging
import re
from typing import Tuple, List, Dict

logger = logging.getLogger(__name__)


class FallbackClassifier:
    """
    Classifies user queries into specific intents based on keyword and context pattern
    matching.

    The `FallbackClassifier` class applies keyword prioritization, assigns confidence
    scores, and uses predefined patterns to determine the user's intent. It is capable
    of handling ambiguous or general queries by using a fallback mechanism to suggest
    the most appropriate intent. The classification process is enhanced with context-based
    boosting to refine scores.

    Attributes
    ----------
    keyword_map : dict
        Intent-specific keyword mappings, including high-priority, medium-priority,
        and generic keywords, used for scoring and classification.
    context_patterns : dict
        Predefined context patterns containing question words, action words,
        and time references for score boosting.
    """

    def __init__(self):
        # Multi-level keyword mapping with priorities
        self.keyword_map = {
            "invoices": {
                "high_priority": [
                    "show.*invoices?",
                    "all.*invoices?",
                    "list.*invoices?",
                    "my invoices?",
                    "invoice.*list",
                    "invoice.*status",
                    "send.*invoice",
                    "create.*invoice",
                    "delete.*invoice",
                    "invoice.*reminder",
                    "payment.*request",
                ],
                "medium_priority": [
                    "invoices?",
                    "bills?",
                    "billing",
                    "client.*payment",
                    "invoice",
                    "bill",
                    "send.*bill",
                    "payment.*due",
                ],
                "keywords": ["invoice", "bill", "billing", "client", "send invoice"],
            },
            "transactions": {
                "high_priority": [
                    "show.*transactions?",
                    "my.*transactions?",
                    "recent.*transactions?",
                    "transaction.*history",
                    "spending.*history",
                    "what.*spent",
                    "expense.*report",
                    "income.*report",
                    "payment.*history",
                ],
                "medium_priority": [
                    "transactions?",
                    "expenses?",
                    "spending",
                    "payments?",
                    "purchases?",
                    "costs?",
                    "income",
                    "money.*spent",
                ],
                "keywords": ["transaction", "expense", "spending", "payment", "cost"],
            },
            "accounts": {
                "high_priority": [
                    "account.*balance",
                    "total.*balance",
                    "how.*much.*money",
                    "bank.*balance",
                    "my.*balance",
                    "account.*summary",
                    "connected.*accounts?",
                    "link.*account",
                    "add.*account",
                ],
                "medium_priority": [
                    "balance",
                    "accounts?",
                    "bank",
                    "money",
                    "funds",
                    "account",
                    "connected",
                    "total",
                ],
                "keywords": ["account", "balance", "bank", "money", "funds"],
            },
            "forecasting": {
                "high_priority": [
                    "cash.*flow.*forecast",
                    "predict.*expenses?",
                    "future.*spending",
                    "forecast.*income",
                    "financial.*projection",
                    "what.*if.*scenario",
                ],
                "medium_priority": [
                    "forecast",
                    "predict",
                    "future",
                    "projection",
                    "outlook",
                    "cash.*flow",
                    "trends?",
                    "next.*month",
                ],
                "keywords": [
                    "forecast",
                    "predict",
                    "future",
                    "projection",
                    "cash flow",
                ],
            },
            "budgets": {
                "high_priority": [
                    "create.*budget",
                    "budget.*planning",
                    "spending.*limit",
                    "budget.*analysis",
                    "am.*i.*over.*budget",
                    "budget.*tracking",
                ],
                "medium_priority": [
                    "budget",
                    "budgets?",
                    "limit",
                    "allocation",
                    "spending.*plan",
                    "financial.*plan",
                ],
                "keywords": ["budget", "limit", "allocation", "spending plan"],
            },
            "insights": {
                "high_priority": [
                    "analyze.*spending",
                    "financial.*analysis",
                    "spending.*trends?",
                    "expense.*analysis",
                    "financial.*report",
                    "spending.*patterns?",
                ],
                "medium_priority": [
                    "analyze",
                    "analysis",
                    "trends?",
                    "patterns?",
                    "insights?",
                    "report",
                    "summary",
                ],
                "keywords": ["analyze", "trends", "patterns", "insights", "report"],
            },
            "bank_connection": {
                "high_priority": [
                    "connect.*bank",
                    "link.*bank",
                    "add.*bank.*account",
                    "setup.*plaid",
                    "integrate.*bank",
                    "new.*bank.*connection",
                ],
                "medium_priority": [
                    "connect",
                    "link",
                    "add.*account",
                    "setup",
                    "integrate",
                    "plaid",
                    "bank.*connection",
                ],
                "keywords": ["connect bank", "link bank", "add bank", "plaid"],
            },
            "payment_processing": {
                "high_priority": [
                    "setup.*stripe",
                    "accept.*payments?",
                    "process.*payments?",
                    "payment.*processing",
                    "stripe.*connect",
                    "payment.*gateway",
                ],
                "medium_priority": [
                    "stripe",
                    "payments?",
                    "processing",
                    "gateway",
                    "merchant",
                    "credit.*card",
                    "payment.*method",
                ],
                "keywords": ["stripe", "process payment", "accept payment"],
            },
        }

        # Context patterns for disambiguation
        self.context_patterns = {
            "question_words": ["what", "how", "when", "where", "why", "show", "list"],
            "action_words": ["create", "add", "send", "delete", "update", "connect"],
            "time_references": ["today", "yesterday", "last", "next", "recent", "this"],
        }

    def classify(self, query: str) -> Tuple[str, float]:
        """
        Classify query using enhanced keyword matching.

        Args:
            query: User query

        Returns:
            Tuple of (intent, confidence)
        """
        if not query or not query.strip():
            return "transactions", 0.3  # Default fallback

        query_lower = query.lower().strip()

        # Score each intent
        intent_scores = {}

        for intent, patterns in self.keyword_map.items():
            score = self._score_intent(query_lower, patterns)
            if score > 0:
                intent_scores[intent] = score

        # Find best match
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[best_intent], 0.9)  # Cap confidence

            logger.info(
                f"Fallback classified '{query}' as {best_intent} (confidence: {confidence:.2f})"
            )
            return best_intent, confidence

        # Final fallback - use transaction assistant for general queries
        logger.info(f"No clear intent found for '{query}', defaulting to transactions")
        return "transactions", 0.4

    def _score_intent(self, query: str, patterns: Dict[str, List[str]]) -> float:
        """Score intent based on pattern matches."""
        score = 0.0

        # High-priority patterns (regex)
        for pattern in patterns.get("high_priority", []):
            if re.search(pattern, query):
                score += 0.8

        # Medium priority patterns (regex)
        for pattern in patterns.get("medium_priority", []):
            if re.search(pattern, query):
                score += 0.5

        # Simple keyword matches
        for keyword in patterns.get("keywords", []):
            if keyword in query:
                score += 0.3

        # Boost score based on context
        context_boost = self._calculate_context_boost(query)
        score += context_boost

        return min(score, 1.0)  # Cap at 1.0

    def _calculate_context_boost(self, query: str) -> float:
        """Calculate context-based score boost."""
        boost = 0.0

        # Question words boost
        for word in self.context_patterns["question_words"]:
            if word in query:
                boost += 0.1
                break

        # Action words boost
        for word in self.context_patterns["action_words"]:
            if word in query:
                boost += 0.1
                break

        # Time references boost
        for word in self.context_patterns["time_references"]:
            if word in query:
                boost += 0.05
                break

        return min(boost, 0.2)  # Cap boost at 0.2

    def get_intent_suggestions(self, query: str) -> List[Tuple[str, float]]:
        """Get top 3 intent suggestions with confidence scores."""
        query_lower = query.lower().strip()
        intent_scores = {}

        for intent, patterns in self.keyword_map.items():
            score = self._score_intent(query_lower, patterns)
            if score > 0.2:  # Only include reasonable matches
                intent_scores[intent] = score

        # Sort by score and return top 3
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_intents[:3]
