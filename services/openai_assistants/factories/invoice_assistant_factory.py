"""
Invoice Assistant Factory for OpenAI Assistants.
Specialized factory for creating and managing invoice-related assistants.
"""

from typing import Dict, List, Any
from .base_assistant_factory import BaseAssistantFactory


class InvoiceAssistantFactory(BaseAssistantFactory):
    """Factory for creating OpenAI Invoice Assistants."""

    def get_assistant_name(self) -> str:
        return "Cashly Invoice Assistant"

    def get_assistant_config(self) -> Dict[str, Any]:
        """Get configuration for Invoice Assistant."""
        return {
            "name": self.get_assistant_name(),
            "instructions": self._get_instructions(),
            "model": self.model,
            "tools": self._build_tools_list(self._get_function_names()),
        }

    @staticmethod
    def _get_instructions() -> str:
        """Get detailed instructions for the Invoice Assistant."""
        return """You are the Invoice Assistant for Cashly, specializing in creating, managing, and tracking invoices and payments.

Your primary responsibilities:
- Create professional invoices for clients (creates DRAFT invoices)
- Show invoice details for review before sending
- Send invoices when user confirms they're ready
- View and manage existing invoices
- Send payment reminders for overdue invoices
- Mark invoices as paid when payments are received
- Delete draft invoices when requested (ONLY for draft status invoices)

CRITICAL BEHAVIORS:

1. INVOICE CREATION:
   - When you create an invoice, you ALWAYS use create_invoice to create it before ANYTHING else
   - When presenting a draft only give MINIMAL details as the UI will display the draft in card format

2. SENDING INVOICES:
   - When user says "yes send it" after invoice creation, use the invoice_id to send the invoice
   - When a user says "Send it now", use the invoice_id and call send_invoice immediately, nothing ELSE

3. DELETING INVOICES:
   - When user says "delete invoice [ID]" or similar, IMMEDIATELY call delete_invoice with that ID
   - Do NOT call get_invoices first - use delete_invoice directly with the provided ID
   - Only delete invoices with "draft" status - the backend will validate this
   - Always confirm before deletion: "Are you sure you want to delete this draft invoice? This action cannot be undone."
   - Use delete_invoice function with the invoice_id
   - Inform user that deletion removes from both database and Stripe

When creating invoices:
1. Use the create_invoice function with the provided details
2. The function will return an action that the system will process
3. DO NOT say there was an error if you don't see an invoice_id immediately
4. Instead, say something like "I'm creating the invoice for [client_name] for $[amount]. I'll have the invoice details ready in just a moment."
5. The actual invoice ID and details will be provided by the system after creation

Available Tools:
- create_invoice: Create new DRAFT invoices (returns invoice with ID)
- send_invoice: Send a draft invoice to the client (requires invoice_id)
- delete_invoice: Delete DRAFT invoices permanently (requires invoice_id) - USE DIRECTLY, don't get_invoices first
- get_invoices: View and filter existing invoices
- send_invoice_reminder: Send payment reminders
- mark_invoice_paid: Mark invoices as paid (requires invoice_id)"""

    @staticmethod
    def _get_function_names() -> List[str]:
        """Get list of function names for Invoice Assistant."""
        return [
            "create_invoice",
            "send_invoice",
            "delete_invoice",
            "get_invoices",
            "send_invoice_reminder",
            "mark_invoice_paid",
        ]

    @staticmethod
    def get_specialized_features() -> Dict[str, Any]:
        """Get specialized features and capabilities of this assistant."""
        return {
            "primary_domain": "invoice_management",
            "core_functions": [
                "Create professional invoices",
                "Send invoices to clients",
                "Track invoice status",
                "Send payment reminders",
                "Mark invoices as paid",
                "Delete draft invoices",
            ],
            "workflow_features": {
                "draft_creation": True,
                "send_confirmation": True,
                "payment_tracking": True,
                "stripe_integration": True,
            },
            "business_logic": [
                "Only delete draft invoices",
                "Require confirmation for deletion",
                "Direct invoice ID usage for operations",
                "Minimal UI details (cards handle display)",
            ],
        }
