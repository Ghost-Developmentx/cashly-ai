# app/core/pipeline/classifier.py
"""
Query classifier - integrates with existing ML classification system.
Uses AsyncIntentService for embeddings-based classification.
"""

import logging
from typing import Dict, Any, List, Optional
from ...schemas.classification import Intent, ClassificationResult
from ...schemas.assistant import AssistantType

logger = logging.getLogger(__name__)


class QueryClassifier:
    """
    Classifies user queries using the existing ML-based intent service.
    Falls back to keyword matching only when ML service is unavailable.
    """

    def __init__(self):
        """Initialize with intent service."""
        self.intent_service = None
        self._initialize_intent_service()

        # Fallback patterns if ML service unavailable
        self.fallback_patterns = self._build_fallback_patterns()
        self.assistant_mapping = self._build_assistant_mapping()

    def _initialize_intent_service(self):
        """Initialize the ML intent service."""
        try:
            from app.services.intent_classification.async_intent_service import AsyncIntentService
            self.intent_service = AsyncIntentService()
            logger.info("✅ ML Intent Service initialized for classification")
        except ImportError as e:
            logger.error(f"Failed to import AsyncIntentService: {e}")
            self.intent_service = None

    @staticmethod
    def _build_fallback_patterns() -> Dict[Intent, List[str]]:
        """Build fallback keyword patterns if ML fails."""
        return {
            Intent.TRANSACTION_QUERY: ["transaction", "spent", "expense", "purchase", "payment"],
            Intent.TRANSACTION_CREATE: ["add transaction", "new transaction", "record"],
            Intent.ACCOUNT_BALANCE: ["balance", "account", "how much", "accounts"],
            Intent.GET_ACCOUNTS: ["my accounts", "how many accounts", "accounts", "show me all my accounts"],
            Intent.ACCOUNT_CONNECT: ["connect", "link", "plaid", "bank"],
            Intent.INVOICE_CREATE: ["invoice", "bill", "send invoice"],
            Intent.PAYMENT_SETUP: ["stripe", "payment", "accept payment"],
            Intent.FORECAST: ["forecast", "predict", "future", "cash flow"],
            Intent.BUDGET: ["budget", "spending plan", "allocation"],
            Intent.INSIGHTS: ["insights", "analyze", "trends", "patterns"],
            Intent.GENERAL: ["help", "what can you do"],
        }

    @staticmethod
    def _build_assistant_mapping() -> Dict[str, AssistantType]:
        """Map intent strings to assistant types."""
        return {
            "transactions": AssistantType.TRANSACTION,
            "transaction": AssistantType.TRANSACTION,
            "accounts": AssistantType.ACCOUNT,
            "account": AssistantType.ACCOUNT,
            "invoices": AssistantType.INVOICE,
            "invoice": AssistantType.INVOICE,
            "bank_connection": AssistantType.BANK_CONNECTION,
            "payment_processing": AssistantType.PAYMENT_PROCESSING,
            "forecasting": AssistantType.FORECASTING,
            "forecast": AssistantType.FORECASTING,
            "budgets": AssistantType.BUDGET,
            "budget": AssistantType.BUDGET,
            "insights": AssistantType.INSIGHTS,
            "general": AssistantType.TRANSACTION,  # Default
        }

    async def classify(
            self,
            query: str,
            conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ClassificationResult:
        """
        Classify using ML service first, fallback to keywords if needed.

        Args:
            query: User's query
            conversation_history: Optional conversation history

        Returns:
            ClassificationResult with intent and confidence
        """
        # Try ML classification first if available
        if self.intent_service:
            try:
                return await self._classify_with_ml(query, conversation_history)
            except Exception as e:
                logger.error(f"ML classification failed: {e}", exc_info=True)
                logger.info("Falling back to keyword classification")

        # Fallback to keyword classification
        return await self._classify_with_keywords(query)

    async def _classify_with_ml(
            self,
            query: str,
            conversation_history: Optional[List[Dict[str, Any]]]
    ) -> ClassificationResult:
        """Use ML-based classification with embeddings."""
        # Call the intent service
        result = await self.intent_service.classify_and_route(
            query=query,
            user_context=None,  # Will be added by pipeline if needed
            conversation_history=conversation_history
        )

        # Extract classification info
        classification = result.get("classification", {})
        intent_str = classification.get("intent", "general")
        confidence = classification.get("confidence", 0.0)
        method = classification.get("method", "vector_search")

        # Map string intent to enum
        intent_enum = self._map_intent_to_enum(intent_str)

        # Map to assistant type
        assistant_type = self.assistant_mapping.get(intent_str, AssistantType.TRANSACTION)

        logger.info(
            f"ML Classification: {intent_str} -> {intent_enum.value} "
            f"(confidence: {confidence:.2f}, method: {method})"
        )

        return ClassificationResult(
            intent=intent_enum,
            confidence=confidence,
            suggested_assistant=assistant_type,
            method=method,
            keywords_found=[]
        )

    async def _classify_with_keywords(self, query: str) -> ClassificationResult:
        """Fallback keyword-based classification."""
        query_lower = query.lower()
        best_intent = Intent.GENERAL
        best_score = 0.0

        # Score each intent based on keyword matches
        for intent, keywords in self.fallback_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > best_score:
                best_score = score
                best_intent = intent

        # Calculate confidence based on matches
        confidence = min(best_score * 0.25, 0.9) if best_score > 0 else 0.3

        # Map to assistant
        assistant_type = self._intent_to_assistant(best_intent)

        logger.info(
            f"Keyword Classification: {best_intent.value} "
            f"(confidence: {confidence:.2f})"
        )

        return ClassificationResult(
            intent=best_intent,
            confidence=confidence,
            suggested_assistant=assistant_type,
            method="keyword",
            keywords_found=self.fallback_patterns[best_intent]
        )

    @staticmethod
    def _map_intent_to_enum(intent_str: str) -> Intent:
        """Map string intent to Intent enum."""
        # Handle direct mappings
        intent_mapping = {
            "transactions": Intent.TRANSACTION_QUERY,
            "transaction": Intent.TRANSACTION_QUERY,
            "accounts": Intent.ACCOUNT_BALANCE,
            "account": Intent.ACCOUNT_BALANCE,
            "invoices": Intent.INVOICE_MANAGE,
            "invoice": Intent.INVOICE_CREATE,
            "bank_connection": Intent.ACCOUNT_CONNECT,
            "payment_processing": Intent.PAYMENT_SETUP,
            "forecasting": Intent.FORECAST,
            "forecast": Intent.FORECAST,
            "budgets": Intent.BUDGET,
            "budget": Intent.BUDGET,
            "insights": Intent.INSIGHTS,
            "general": Intent.GENERAL,
        }

        return intent_mapping.get(intent_str, Intent.GENERAL)

    @staticmethod
    def _intent_to_assistant(intent: Intent) -> AssistantType:
        """Map Intent enum to AssistantType."""
        mapping = {
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
            Intent.GENERAL: AssistantType.TRANSACTION,
        }

        return mapping.get(intent, AssistantType.TRANSACTION)

    async def classify_with_ml(
            self,
            query: str,
            conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ClassificationResult:
        """
        Force ML classification (called when pipeline has ML enabled).
        """
        if not self.intent_service:
            logger.warning("ML classification requested but service not available")
            return await self._classify_with_keywords(query)

        return await self._classify_with_ml(query, conversation_history)
