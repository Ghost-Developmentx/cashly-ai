"""
Async handlers for invoice-related tools.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class AsyncInvoiceHandlers:
    """
    Handles asynchronous operations related to invoices.

    Provides methods to interact with invoice services such as creating, deleting, and
    sending invoices, as well as managing Stripe connections and invoice statuses. These
    methods heavily rely on Rails API integration to perform actions asynchronously.

    Attributes
    ----------
    rails_client : Any
        Client instance to interact with the Rails API.
    """

    def __init__(self, rails_client):
        self.rails_client = rails_client

    @staticmethod
    async def connect_stripe(context: Dict[str, Any]) -> Dict[str, Any]:
        """Connect Stripe account."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        api_key = tool_args.get("api_key")

        return {
            "action": "connect_stripe",
            "user_id": user_id,
            "api_key": api_key,
            "message": "I'll connect your Stripe account for you.",
            "requires_confirmation": True,
        }

    async def create_invoice(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new invoice via Rails API."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]

        # Validate required fields
        required_fields = ["client_name", "client_email", "amount"]
        missing_fields = [
            field for field in required_fields if not tool_args.get(field)
        ]

        if missing_fields:
            return {
                "error": f"Missing required fields: {', '.join(missing_fields)}",
                "success": False,
            }

        # Prepare invoice data
        invoice_data = {
            "user_id": user_id,
            "invoice": {
                "client_name": tool_args.get("client_name"),
                "client_email": tool_args.get("client_email"),
                "amount": float(tool_args.get("amount", 0)),
                "description": tool_args.get("description", ""),
                "due_date": self._get_due_date(tool_args.get("due_date")),
                "currency": tool_args.get("currency", "USD"),
            },
        }

        # Call Rails API asynchronously
        result = await self.rails_client.post("/api/internal/invoices", invoice_data)

        if "error" in result:
            return {
                "error": result["error"],
                "message": "I couldn't create the invoice. Please try again.",
                "success": False,
            }

        return {
            "invoice_id": result["invoice_id"],
            "invoice": result["invoice"],
            "platform_fee": result.get("platform_fee"),
            "success": True,
        }

    async def delete_invoice(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Delete an invoice via Rails API."""
        tool_args = context["tool_args"]
        user_id = context["user_id"]
        invoice_id = tool_args.get("invoice_id")

        result = await self.rails_client.delete(
            f"/api/internal/invoices/{invoice_id}", {"user_id": user_id}
        )

        if "error" in result:
            return {
                "success": False,
                "error": result["error"],
                "message": f"I couldn't delete the invoice: {result['error']}",
            }

        return {
            "action": "delete_invoice_completed",
            "deleted_invoice": result["deleted_invoice"],
            "stripe_deleted": result.get("stripe_deleted", False),
            "message": result.get("message", "Invoice deleted successfully"),
            "success": True,
        }

    @staticmethod
    async def send_invoice(context: Dict[str, Any]) -> Dict[str, Any]:
        """Send an invoice."""
        tool_args = context["tool_args"]
        invoice_id = tool_args.get("invoice_id")

        return {
            "action": "send_invoice_initiated",
            "invoice_id": invoice_id,
            "user_id": context["user_id"],
            "message": f"Sending invoice {invoice_id} to the client now...",
        }

    @staticmethod
    async def get_invoices(context: Dict[str, Any]) -> Dict[str, Any]:
        """Get user invoices."""
        tool_args = context["tool_args"]

        clean_filters = {}
        for key in ["status", "client_name", "days", "invoice_id", "id"]:
            if key in tool_args:
                clean_filters[key] = tool_args[key]

        return {
            "action": "get_invoices",
            "filters": clean_filters,
            "user_id": context["user_id"],
        }

    @staticmethod
    async def send_invoice_reminder(context: Dict[str, Any]) -> Dict[str, Any]:
        """Send invoice reminder."""
        tool_args = context["tool_args"]
        invoice_id = tool_args.get("invoice_id")

        return {
            "action": "send_invoice_reminder",
            "invoice_id": invoice_id,
            "message": "I'll send a payment reminder for this invoice right away.",
        }

    @staticmethod
    async def mark_invoice_paid(context: Dict[str, Any]) -> Dict[str, Any]:
        """Mark invoice as paid."""
        tool_args = context["tool_args"]
        invoice_id = tool_args.get("invoice_id")

        return {
            "action": "mark_invoice_paid",
            "invoice_id": invoice_id,
            "user_id": context["user_id"],
            "message": f"I'll mark invoice {invoice_id} as paid.",
        }

    @staticmethod
    def _get_due_date(due_date: Optional[str]) -> str:
        """Get valid due date."""
        if not due_date:
            return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            due_date_obj = datetime.strptime(due_date, "%Y-%m-%d")
            if due_date_obj < datetime.now():
                return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            return due_date
        except ValueError:
            return (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
