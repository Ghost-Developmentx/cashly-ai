"""
Context enhancement for queries.
Adds user context and generates additional instructions.
"""

import logging
from typing import Dict, Optional
from .types import ContextEnhancement

logger = logging.getLogger(__name__)


class ContextEnhancer:
    """
    A utility class for enhancing user queries with contextual information.

    This class provides functionality to enrich original user queries with
    contextual information extracted from the user's data, such as accounts,
    payment integration status, and transaction history. It helps generate
    an enhanced query that combines original intent with relevant contextual
    content, as well as additional instructions for further guidance.

    Attributes
    ----------
    None
    """

    @staticmethod
    def enhance_query(
            query: str,
            user_context: Optional[Dict] = None
    ) -> ContextEnhancement:
        """
        Enhance query with user context.

        Args:
            query: Original user query
            user_context: User context data

        Returns:
            ContextEnhancement object
        """
        if not user_context:
            return ContextEnhancement(
                original_query=query,
                enhanced_query=query,
                context_parts=[],
                additional_instructions=""
            )

        context_parts = []

        # Add account context
        accounts = user_context.get("accounts", [])
        if accounts:
            total_balance = sum(acc.get("balance", 0) for acc in accounts)
            account_summary = (
                f"Connected accounts: {len(accounts)} accounts "
                f"with total balance of ${total_balance:,.2f}"
            )
            context_parts.append(account_summary)

        # Add Stripe Connect context
        stripe_status = user_context.get("stripe_connect", {})
        if stripe_status.get("connected"):
            if stripe_status.get("can_accept_payments"):
                context_parts.append("Stripe Connect: Active and ready for payments")
            else:
                status = stripe_status.get("status", "unknown")
                context_parts.append(f"Stripe Connect: Connected but status is {status}")
        else:
            context_parts.append("Stripe Connect: Not connected")

        # Add integration context
        integrations = user_context.get("integrations", [])
        if integrations:
            integration_names = [i.get("provider", "Unknown") for i in integrations]
            context_parts.append(f"Active integrations: {', '.join(integration_names)}")

        # Add transaction context - MODIFIED THIS PART
        transactions = user_context.get("transactions", [])
        if transactions:
            context_parts.append(
                f"Available transaction data: {len(transactions)} transactions "
                f"(accessible via tools)"  # Added this clarification
            )

        # Build enhanced query - MODIFIED THIS PART
        if context_parts:
            context_info = "User context: " + "; ".join(context_parts)
            # Add explicit instruction to use tools
            enhanced_query = f"""{context_info}

    IMPORTANT: Use the appropriate tools to access and analyze the available data. Do not assume data is unavailable without using the tools first.

    User query: {query}"""
        else:
            enhanced_query = query

        # Generate additional instructions
        instructions = ContextEnhancer._generate_instructions(user_context)

        return ContextEnhancement(
            original_query=query,
            enhanced_query=enhanced_query,
            context_parts=context_parts,
            additional_instructions=instructions
        )

    @staticmethod
    def _generate_instructions(user_context: Dict) -> str:
        """Generate additional instructions based on context."""
        instructions = []

        # Account-specific instructions
        accounts = user_context.get("accounts", [])
        if not accounts:
            instructions.append(
                "Note: User has no connected bank accounts. "
                "Suggest connecting accounts for better insights."
            )
        elif len(accounts) == 1:
            instructions.append(
                "Note: User has one connected account. "
                "Consider suggesting additional accounts for comprehensive tracking."
            )

        # Stripe Connect instructions
        stripe_status = user_context.get("stripe_connect", {})
        if stripe_status.get("connected"):
            if stripe_status.get("can_accept_payments"):
                instructions.append(
                    "Note: User has Stripe Connect set up and can accept payments. "
                    "Invoices can be created and sent with payment links."
                )
            else:
                instructions.append(
                    "Note: User has Stripe Connect but needs to complete setup. "
                    "Invoices can be created but may need manual payment processing."
                )
        else:
            instructions.append(
                "Note: User doesn't have Stripe Connect set up. "
                "For invoice queries, suggest setting up Stripe Connect."
            )

        # Transaction-based instructions - MODIFIED THIS PART
        transactions = user_context.get("transactions", [])
        if not transactions:
            instructions.append(
                "Note: No transaction history available. "
                "Financial insights will be limited."
            )
        else:
            # ADD THIS NEW INSTRUCTION
            instructions.append(
                f"IMPORTANT: User has {len(transactions)} transactions available. "
                f"Always use the appropriate tools (forecast_cash_flow, analyze_trends, etc.) "
                f"to access and analyze this data. Never say 'no transaction history' without "
                f"first attempting to use the tools."
            )

        return " ".join(instructions)