import datetime
from typing import Any, Dict, List, Optional


class PromptBuilder:
    """
    Responsible for generating the system prompt and chat message structure.
    """

    @staticmethod
    def build_messages(
        query: str, conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """
        Build a full list of chat messages including history and current query.

        Args:
            query: The latest user question
            conversation_history: Optional prior conversation context

        Returns:
            List of Claude-compatible message blocks
        """
        messages: List[Dict[str, str]] = []
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": query})
        return messages

    def build_system_prompt(
        self, user_id: str, financial_context: Dict[str, Any]
    ) -> str:
        """
        Construct the system prompt that defines Fin's capabilities and the user context.

        Args:
            user_id: The user's identifier
            financial_context: Computed user-specific financial snapshot

        Returns:
            A full prompt string
        """
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        thirty_days_ago = (
            datetime.datetime.now() - datetime.timedelta(days=30)
        ).strftime("%Y-%m-%d")

        return f"""
        You are Fin, an AI-powered financial assistant for the Cashly app. Today is {current_date}.

        IMPORTANT: When calculating time periods or date ranges, ALWAYS use {current_date} as the current date.
        For example, "last 30 days" means from {thirty_days_ago} to {current_date}.

        Your role is to help users understand their finances, answer questions about their spending,
        income, budgets, and provide forecasts and financial advice.

        Important financial information about this user:
        - Total accounts: {financial_context.get('account_count', 'unknown')}
        - Current balance across all accounts: {financial_context.get('total_balance', 'unknown')}
        - Monthly income (avg. last 3 months): {financial_context.get('monthly_income', 'unknown')}
        - Monthly expenses (avg. last 3 months): {financial_context.get('monthly_expenses', 'unknown')}
        - Top spending categories: {', '.join(financial_context.get('top_categories', ['unknown']))}
        - Recurring expenses detected: {financial_context.get('recurring_expenses', 'unknown')}

        ACCOUNT MANAGEMENT CAPABILITIES:
        When users ask about bank accounts, connections, or linking accounts, you should:
        1. Use the get_user_accounts tool to check their current account status
        2. If they have no accounts and want to connect one, use the initiate_plaid_connection tool
        3. If they want to disconnect an account, use the disconnect_account tool
        4. If they want account balances or details, use the get_account_details tool

        ...

        (Prompt truncated for brevity. Full instruction set should be loaded from a separate file or template in production.)
        """.strip()
