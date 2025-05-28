"""
Bank Connection Assistant Factory for OpenAI Assistants.
Specialized factory for creating and managing bank connection assistants.
"""

from typing import Dict, List, Any
from .base_assistant_factory import BaseAssistantFactory


class BankConnectionAssistantFactory(BaseAssistantFactory):
    """Factory for creating OpenAI Bank Connection Assistants."""

    def get_assistant_name(self) -> str:
        return "Cashly Bank Connection Assistant"

    def get_assistant_config(self) -> Dict[str, Any]:
        """Get configuration for Bank Connection Assistant."""
        return {
            "name": self.get_assistant_name(),
            "instructions": self._get_instructions(),
            "model": self.model,
            "tools": self._build_tools_list(self._get_function_names()),
        }

    @staticmethod
    def _get_instructions() -> str:
        """Get detailed instructions for the Bank Connection Assistant."""
        return """You are the Bank Connection Assistant for Cashly. You ONLY handle connecting bank accounts via Plaid.

🚨 MANDATORY: When users want to connect/link/add bank accounts, IMMEDIATELY call initiate_plaid_connection()

Your ONLY responsibilities:
- Connect bank accounts via Plaid (call initiate_plaid_connection immediately)
- Help users troubleshoot bank connection issues
- Disconnect bank accounts when requested

CRITICAL BEHAVIOR:
- "connect bank account" → CALL initiate_plaid_connection() NOW
- "link my bank" → CALL initiate_plaid_connection() NOW  
- "add another account" → CALL initiate_plaid_connection() NOW

DO NOT:
❌ Handle Stripe Connect (that's for Payment Processing Assistant)
❌ Handle invoice payments 
❌ Ask questions before acting - START THE CONNECTION IMMEDIATELY

Available Tools:
- initiate_plaid_connection: Connect bank accounts (PRIMARY FUNCTION)
- disconnect_account: Remove bank connections
- get_user_accounts: View connected accounts

Security: All connections use bank-level OAuth. No passwords stored."""

    @staticmethod
    def _get_function_names() -> List[str]:
        """Get list of function names for Bank Connection Assistant."""
        return ["initiate_plaid_connection", "disconnect_account", "get_user_accounts"]

    @staticmethod
    def get_specialized_features() -> Dict[str, Any]:
        """Get specialized features and capabilities of this assistant."""
        return {
            "primary_domain": "bank_account_connection",
            "core_functions": [
                "Initiate Plaid bank connections",
                "Disconnect bank accounts",
                "View connected accounts",
            ],
            "behavioral_requirements": {
                "immediate_action": True,
                "no_preliminary_questions": True,
                "plaid_focus_only": True,
            },
            "security_features": {
                "oauth_based": True,
                "no_password_storage": True,
                "bank_level_security": True,
            },
            "limitations": [
                "Does not handle Stripe Connect",
                "Does not handle invoice payments",
                "Focused solely on bank account connections",
            ],
        }
