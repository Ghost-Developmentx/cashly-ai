"""
Query classifier - determines intent and suggests appropriate assistant.
Single responsibility: classification only.
"""

import logging
from typing import Dict, Any, List, Optional
from ...schemas.classification import Intent, ClassificationResult
from ...schemas.assistant import AssistantType

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classifies user queries to determine intent.
    This is a simplified version - in production, you'd use ML models.
    """

    def __init__(self):
        """Initialize with intent patterns."""
        self.intent_patterns = self._build_intent_patterns()
        self.assistant_mapping = self._build_assistant_mapping()

    @staticmethod
    def _build_intent_patterns() -> Dict[Intent, List[str]]:
        """Build keyword patterns for each intent."""
        return {
            Intent.TRANSACTION_QUERY: [
                "transaction", "spent", "purchase", "expense", "payment",
                "bought", "paid", "charge", "debit", "credit"
            ],
            Intent.TRANSACTION_CREATE: [
                "add transaction", "create transaction", "record payment",
                "log expense", "track purchase", "add expense"
            ],
            Intent.TRANSACTION_UPDATE: [
                "update transaction", "change transaction", "edit transaction",
                "modify payment", "fix transaction"
            ],
            Intent.TRANSACTION_DELETE: [
                "delete transaction", "remove transaction", "cancel payment"
            ],
            Intent.ACCOUNT_BALANCE: [
                "balance", "account", "how much", "total", "funds",
                "money left", "checking", "savings"
            ],
            Intent.ACCOUNT_CONNECT: [
                "connect bank", "link account", "add bank", "plaid",
                "sync account", "bank connection"
            ],
            Intent.INVOICE_CREATE: [
                "create invoice", "send invoice", "bill client",
                "invoice for", "generate invoice"
            ],
            Intent.INVOICE_MANAGE: [
                "invoice status", "unpaid invoices", "invoice list",
                "payment reminder", "mark paid"
            ],
            Intent.PAYMENT_SETUP: [
                "stripe connect", "accept payments", "payment processing",
                "setup payments", "merchant account"
            ],
            Intent.FORECAST: [
                "forecast", "predict", "projection", "future balance",
                "cash flow", "will I have", "next month"
            ],
            Intent.BUDGET: [
                "budget", "spending limit", "save money", "spending plan",
                "allocate", "50/30/20"
            ],
            Intent.INSIGHTS: [
                "insights", "analysis", "patterns", "anomaly", "unusual",
                "trending", "spending habits"
            ]
        }

    @staticmethod
    def _build_assistant_mapping() -> Dict[Intent, AssistantType]:
        """Map intents to appropriate assistants."""
        return {
            Intent.TRANSACTION_QUERY: AssistantType.TRANSACTION,
            Intent.TRANSACTION_CREATE: AssistantType.TRANSACTION,
            Intent.TRANSACTION_UPDATE: AssistantType.TRANSACTION,
            Intent.TRANSACTION_DELETE: AssistantType.TRANSACTION,
            Intent.ACCOUNT_BALANCE: AssistantType.ACCOUNT,
            Intent.ACCOUNT_CONNECT: AssistantType.BANK_CONNECTION,
            Intent.INVOICE_CREATE: AssistantType.INVOICE,
            Intent.INVOICE_MANAGE: AssistantType.INVOICE,
            Intent.PAYMENT_SETUP: AssistantType.PAYMENT_PROCESSING,
            Intent.FORECAST: AssistantType.FORECASTING,
            Intent.BUDGET: AssistantType.BUDGET,
            Intent.INSIGHTS: AssistantType.INSIGHTS,
            Intent.GENERAL: AssistantType.TRANSACTION  # Default
        }

    async def classify(
            self,
            query: str,
            conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ClassificationResult:
        """
        Classify a query to determine intent.

        Args:
            query: User's query
            conversation_history: Optional previous messages

        Returns:
            ClassificationResult with intent and confidence
        """
        query_lower = query.lower()

        # Check each intent pattern
        best_match = None
        best_score = 0.0
        best_keywords = []

        for intent, keywords in self.intent_patterns.items():
            found_keywords = []
            score = 0.0

            for keyword in keywords:
                if keyword in query_lower:
                    found_keywords.append(keyword)
                    # Longer keywords get higher weight
                    score += len(keyword.split())

            if score > best_score:
                best_score = score
                best_match = intent
                best_keywords = found_keywords

        # If no match, use general intent
        if not best_match:
            best_match = Intent.GENERAL
            confidence = 0.3
        else:
            # Calculate confidence based on keyword matches
            max_possible_score = sum(len(k.split()) for k in self.intent_patterns[best_match][:3])
            confidence = min(best_score / max_possible_score, 0.95) if max_possible_score > 0 else 0.5

        # Get suggested assistant
        suggested_assistant = self.assistant_mapping[best_match]

        # Consider conversation history for context
        if conversation_history and confidence < 0.7:
            # Boost confidence if recent messages are about same topic
            confidence = min(confidence + 0.1, 0.8)

        result = ClassificationResult(
            intent=best_match,
            confidence=confidence,
            suggested_assistant=suggested_assistant,
            keywords_found=best_keywords,
            method="keyword"
        )

        logger.info(f"Classified query: {best_match.value} (confidence: {confidence:.2f})")

        return result

    async def classify_with_ml(
            self,
            query: str,
            conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ClassificationResult:
        """
        Classify using ML model (placeholder for actual ML integration).

        In production, this would:
        1. Use your existing intent classification service
        2. Combine with keyword-based classification
        3. Return hybrid results
        """
        # For now, just use keyword classification
        keyword_result = await self.classify(query, conversation_history)

        # In production, you'd call your ML service here:
        # try:
        #     from app.services.intent_classification import AsyncIntentService
        #     intent_service = AsyncIntentService()
        #     ml_result = await intent_service.classify_and_route(query)
        #     # Merge results...
        # except Exception as e:
        #     logger.warning(f"ML classification failed, using keyword: {e}")

        keyword_result.method = "hybrid"
        return keyword_result
