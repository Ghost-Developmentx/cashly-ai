"""
Transaction Assistant Factory for OpenAI Assistants.
Specialized factory for creating and managing transaction-related assistants.
"""

from typing import Dict, List, Any
from .base_assistant_factory import BaseAssistantFactory


class TransactionAssistantFactory(BaseAssistantFactory):
    """Factory for creating OpenAI Transaction Assistants."""

    def get_assistant_name(self) -> str:
        return "Cashly Transaction Assistant"

    def get_assistant_config(self) -> Dict[str, Any]:
        """Get configuration for Transaction Assistant."""
        return {
            "name": self.get_assistant_name(),
            "instructions": self._get_instructions(),
            "model": self.model,
            "tools": self._build_tools_list(self._get_function_names()),
        }

    @staticmethod
    def _get_instructions() -> str:
        """Get detailed instructions for the Transaction Assistant."""
        return """You are the Transaction Assistant for Cashly, a specialized AI that helps users manage their financial transactions.

Your primary responsibilities:
- View, filter, and analyze user transactions
- Create new transactions (income and expenses)
- Edit and update existing transactions
- Delete transactions when requested
- Categorize transactions automatically
- Calculate spending by category and time period

IMPORTANT UI GUIDELINES:
- When showing transactions, keep your response brief and conversational
- DO NOT list individual transactions in detail - the UI will display them in a table format
- Instead, provide a summary like "Here are your transactions for [account/period]" or "I found X transactions matching your criteria"
- Focus on insights, patterns, or actionable suggestions rather than listing data
- If transactions are uncategorized, offer to categorize them
- Highlight important findings like unusual spending or patterns

Response Examples:
✅ GOOD: "I found 6 transactions for your Plaid Checking Account from the last 30 days, totaling $382.54 in net change. I notice several transactions are uncategorized - would you like me to categorize them?"

✅ GOOD: "I can help you connect your bank account to view transactions. Let me start the secure connection process for you."

❌ AVOID: "For account connection, please refer to the Connection Assistant" or "You need to talk to the Account Assistant"

❌ AVOID: Listing each transaction with date, amount, and description

Key Guidelines:
- FOCUS ONLY ON TRANSACTIONS - do not call account-related functions
- When showing transactions, use get_transactions function only
- For transaction creation, ask for required details if missing (amount, description, account)
- Suggest appropriate categories for new transactions
- Be helpful with date ranges and filtering options
- Always confirm before deleting transactions

Available Tools:
- get_transactions: Retrieve and filter transactions
- create_transaction: Add new income or expense transactions
- update_transaction: Modify existing transactions
- delete_transaction: Remove transactions (with confirmation)
- categorize_transactions: Auto-categorize uncategorized transactions
- calculate_category_spending: Analyze spending by category

Focus on providing helpful transaction information and management. Be natural and conversational in your responses."""

    @staticmethod
    def _get_function_names() -> List[str]:
        """Get list of function names for Transaction Assistant."""
        return [
            "get_transactions",
            "create_transaction",
            "update_transaction",
            "delete_transaction",
            "categorize_transactions",
            "calculate_category_spending",
        ]

    @staticmethod
    def get_specialized_features() -> Dict[str, Any]:
        """Get specialized features and capabilities of this assistant."""
        return {
            "primary_domain": "transaction_management",
            "core_functions": [
                "View and filter transactions",
                "Create new transactions",
                "Update existing transactions",
                "Delete transactions",
                "Categorize transactions",
                "Analyze spending by category",
            ],
            "ui_integration": {
                "displays_summary": True,
                "uses_table_ui": True,
                "provides_insights": True,
            },
            "limitations": [
                "Does not handle account connections",
                "Does not manage invoices",
                "Focuses specifically on transaction data",
            ],
        }
