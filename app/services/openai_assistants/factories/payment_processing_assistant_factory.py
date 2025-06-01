"""
Payment Processing Assistant Factory for OpenAI Assistants.
Specialized factory for creating and managing Stripe Connect payment processing assistants.
"""

from typing import Dict, List, Any
from .base_assistant_factory import BaseAssistantFactory


class PaymentProcessingAssistantFactory(BaseAssistantFactory):
    """Factory for creating OpenAI Payment Processing Assistants (Stripe Connect)."""

    def get_assistant_name(self) -> str:
        return "Cashly Payment Processing Assistant"

    def get_assistant_config(self) -> Dict[str, Any]:
        """Get configuration for Payment Processing Assistant."""
        return {
            "name": self.get_assistant_name(),
            "instructions": self._get_instructions(),
            "model": self.model,
            "tools": self._build_tools_list(self._get_function_names()),
        }

    @staticmethod
    def _get_instructions() -> str:
        """Get detailed instructions for the Payment Processing Assistant."""
        return """You are the Payment Processing Assistant for Cashly. You ONLY handle Stripe Connect for accepting payments.

🚨 MANDATORY: When users want payment processing/Stripe setup, IMMEDIATELY call the appropriate Stripe tools.

Your ONLY responsibilities:
- Set up Stripe Connect for payment processing
- Help with incomplete/rejected Stripe accounts  
- Open Stripe dashboards
- Troubleshoot Stripe Connect issues
- Check Stripe Connect requirements and status

CRITICAL BEHAVIOR:
- "setup stripe" → CALL setup_stripe_connect() NOW
- "stripe dashboard" → CALL create_stripe_connect_dashboard_link() NOW
- "restart stripe" → CALL restart_stripe_connect_setup() NOW
- "stripe requirements" → CALL get_stripe_connect_requirements() NOW
- "stripe status" → CALL check_stripe_connect_status() NOW

DO NOT:
❌ Handle bank account connections (that's for Bank Connection Assistant)
❌ Handle transaction viewing
❌ Ask questions before acting - USE TOOLS IMMEDIATELY
❌ Write out in details the Stripe Connect requirements and process as it will be shown in a UI element

Available Tools:
- setup_stripe_connect: Set up Stripe Connect for payments
- create_stripe_connect_dashboard_link: Open Stripe dashboard
- restart_stripe_connect_setup: Start fresh Stripe setup
- get_stripe_connect_requirements: Check what Stripe needs
- check_stripe_connect_status: Get current Stripe status
- get_stripe_connect_earnings: View earnings and fees
- disconnect_stripe_connect: Disconnect Stripe (with confirmation)

Focus: Help users accept payments through invoices with Stripe Connect."""

    @staticmethod
    def _get_function_names() -> List[str]:
        """Get list of function names for Payment Processing Assistant."""
        return [
            "setup_stripe_connect",
            "create_stripe_connect_dashboard_link",
            "restart_stripe_connect_setup",
            "get_stripe_connect_requirements",
            "check_stripe_connect_status",
            "get_stripe_connect_earnings",
            "disconnect_stripe_connect",
        ]

    @staticmethod
    def get_specialized_features() -> Dict[str, Any]:
        """Get specialized features and capabilities of this assistant."""
        return {
            "primary_domain": "stripe_connect_payment_processing",
            "core_functions": [
                "Set up Stripe Connect accounts",
                "Manage Stripe onboarding process",
                "Handle incomplete/rejected accounts",
                "Provide dashboard access",
                "Check account requirements",
                "View earnings and platform fees",
            ],
            "behavioral_requirements": {
                "immediate_action": True,
                "stripe_focus_only": True,
                "no_preliminary_questions": True,
            },
            "integration_features": {
                "stripe_connect_api": True,
                "onboarding_management": True,
                "earnings_tracking": True,
                "requirement_validation": True,
            },
            "limitations": [
                "Does not handle bank account connections",
                "Does not handle transaction management",
                "Focused solely on Stripe Connect setup and management",
            ],
        }
