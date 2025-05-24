import os
import logging
from typing import Any, Dict, List, Optional

import anthropic

from services.fin.prompt_builder import PromptBuilder
from services.fin.tool_registry import ToolRegistry
from services.fin.response_processor import ResponseProcessor
from services.fin.utils import normalize_transaction_dates

logger = logging.getLogger(__name__)


class FinService:
    """
    Main interface for processing user queries and financial insights via Claude + internal tools.
    """

    def __init__(
        self,
        client: Optional[anthropic.Anthropic] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        tool_registry: Optional[ToolRegistry] = None,
        response_processor: Optional[ResponseProcessor] = None,
    ):
        self.client = client or anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.tool_registry = tool_registry or ToolRegistry()
        self.response_processor = response_processor or ResponseProcessor(
            self.tool_registry.execute
        )

    def process_query(
        self,
        user_id: str,
        query: str,
        transactions: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Primary entrypoint called by app frontend to generate a financial AI response.
        """
        user_context = user_context or {}
        transactions = transactions or []
        normalized_txns = normalize_transaction_dates(transactions)

        # Extract prompt context
        financial_context = self._extract_context(normalized_txns, user_context)
        system_prompt = self.prompt_builder.build_system_prompt(
            user_id, financial_context, user_context
        )
        messages = self.prompt_builder.build_messages(query, conversation_history)

        logger.info(f"Calling Claude with query: {query}")

        try:
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                system=system_prompt,
                messages=messages,
                max_tokens=2000,
                extra_body={"tools": self.tool_registry.schemas},
            )
        except Exception as e:
            logger.error(f"Claude call failed: {e}", exc_info=True)
            return {
                "response_text": "I'm sorry, something went wrong while generating your answer.",
                "error": str(e),
            }

        return self.response_processor.process(
            response,
            user_id=user_id,
            transactions=normalized_txns,
            user_context=user_context,
            original_messages=messages,
            system_prompt=system_prompt,
            claude_client=self.client,
        )

    def _extract_context(
        self, transactions: List[Dict[str, Any]], user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Derive top-level context info like account count, income, expenses."""
        import datetime
        import pandas as pd

        context: Dict[str, Any] = {}
        accounts = user_context.get("accounts", [])
        context["account_count"] = len(accounts)
        context["total_balance"] = sum(a.get("balance", 0) for a in accounts)

        today = datetime.datetime.now().date()
        three_months_ago = today - datetime.timedelta(days=90)
        recent_txns = []

        for t in transactions:
            try:
                tx_date = t["date"]
                if isinstance(tx_date, (datetime.datetime, pd.Timestamp)):
                    tx_date = tx_date.strftime("%Y-%m-%d")
                date_obj = datetime.datetime.strptime(tx_date, "%Y-%m-%d").date()
                if date_obj >= three_months_ago:
                    recent_txns.append(t)
            except Exception:
                continue

        if not recent_txns:
            context.update(
                {
                    "monthly_income": "unknown",
                    "monthly_expenses": "unknown",
                    "top_categories": ["unknown"],
                    "recurring_expenses": "unknown",
                }
            )
            return context

        income = sum(t["amount"] for t in recent_txns if t["amount"] > 0)
        expenses = sum(abs(t["amount"]) for t in recent_txns if t["amount"] < 0)
        context["monthly_income"] = f"${income / 3:.2f}"
        context["monthly_expenses"] = f"${expenses / 3:.2f}"

        # Top categories
        category_spend = {}
        for t in recent_txns:
            if t["amount"] < 0:
                cat = t.get("category", "Uncategorized")
                category_spend[cat] = category_spend.get(cat, 0) + abs(t["amount"])

        context["top_categories"] = sorted(
            category_spend, key=category_spend.get, reverse=True
        )[:3]
        context["recurring_expenses"] = (
            f"{sum(1 for t in recent_txns if t.get('recurring'))} recurring items detected"
        )

        return context
