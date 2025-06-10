"""
Invoice tool handlers.
Migrated from app/services/fin/tool_handlers/invoice_handlers.py
"""

import logging
from typing import Dict, Any
from datetime import datetime
from ..registry import tool_registry
from ..schemas import INVOICE_SCHEMAS
from ..helpers.invoice_helpers import get_due_date, parse_date

logger = logging.getLogger(__name__)

@tool_registry.register(
    name="connect_stripe",
    description="Connect Stripe account for payment processing",
    schema=INVOICE_SCHEMAS["CONNECT_STRIPE"],
    category="invoices"
)
async def connect_stripe(context: Dict[str, Any]) -> Dict[str, Any]:
    """Initiate Stripe connection for invoicing."""
    user_id = context.get("user_id")

    if not user_id:
        return {"error": "User ID is required for Stripe connection"}

    return {
        "action": "connect_stripe",
        "user_id": user_id,
        "message": "I'll help you connect your Stripe account to start accepting payments.",
        "instructions": "Click below to securely connect your Stripe account."
    }

@tool_registry.register(
    name="create_invoice",
    description="Create a new invoice",
    schema=INVOICE_SCHEMAS["CREATE_INVOICE"],
    category="invoices",
    requires_confirmation=True
)
async def create_invoice(context: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new invoice."""
    tool_args = context["tool_args"]
    user_id = context.get("user_id")
    rails_client = context.get("rails_client")

    # Validate required fields
    client_name = tool_args.get("client_name")
    client_email = tool_args.get("client_email")
    amount = tool_args.get("amount")

    if not all([client_name, client_email, amount]):
        return {
            "error": "Missing required fields: client_name, client_email, and amount are required"
        }

    # Validate email format
    if "@" not in client_email or "." not in client_email.split("@")[1]:
        return {"error": "Invalid email format"}

    # Parse amount
    try:
        amount_float = float(str(amount).replace("$", "").replace(",", ""))
        if amount_float <= 0:
            return {"error": "Invoice amount must be greater than zero"}
    except ValueError:
        return {"error": "Invalid amount format"}

    # Get due date
    due_date = get_due_date(tool_args.get("due_date"))

    # Prepare invoice data
    invoice_data = {
        "client_name": client_name.strip(),
        "client_email": client_email.strip().lower(),
        "amount": amount_float,
        "description": tool_args.get("description", "Professional Services"),
        "due_date": due_date,
        "status": "draft",
        "created_via_ai": True
    }

    # If we have a rails client, create invoice
    if rails_client:
        try:
            result = await rails_client.post(
                f"/api/v1/fin/invoices/create",
                json={
                    "user_id": user_id,
                    "invoice": invoice_data
                }
            )

            if result.get("invoice_id"):
                return {
                    "success": True,
                    "invoice_id": result["invoice_id"],
                    "invoice": result.get("invoice", invoice_data),
                    "message": f"Invoice for ${amount_float:,.2f} created successfully!",
                    "next_action": "You can now send this invoice to {client_name}."
                }
        except Exception as e:
            logger.error(f"Failed to create invoice via Rails: {e}")

    # Return action for frontend
    return {
        "action": "create_invoice",
        "invoice": invoice_data,
        "message": f"I'll create an invoice for ${amount_float:,.2f} to {client_name}."
    }

@tool_registry.register(
    name="send_invoice",
    description="Send an invoice to the client",
    schema=INVOICE_SCHEMAS["SEND_INVOICE"],
    category="invoices",
    requires_confirmation=True
)
async def send_invoice(context: Dict[str, Any]) -> Dict[str, Any]:
    """Send an invoice to the client."""
    tool_args = context["tool_args"]
    invoice_id = tool_args.get("invoice_id")

    if not invoice_id:
        return {"error": "Invoice ID is required"}

    return {
        "action": "send_invoice",
        "invoice_id": invoice_id,
        "message": "I'll send this invoice to your client right away.",
        "requires_confirmation": True
    }

@tool_registry.register(
    name="delete_invoice",
    description="Delete an invoice",
    schema=INVOICE_SCHEMAS["DELETE_INVOICE"],
    category="invoices",
    requires_confirmation=True
)
async def delete_invoice(context: Dict[str, Any]) -> Dict[str, Any]:
    """Delete an invoice."""
    tool_args = context["tool_args"]
    invoice_id = tool_args.get("invoice_id")

    if not invoice_id:
        return {"error": "Invoice ID is required"}

    return {
        "action": "delete_invoice",
        "invoice_id": invoice_id,
        "message": "Are you sure you want to delete this invoice? This action cannot be undone.",
        "requires_confirmation": True
    }

@tool_registry.register(
    name="get_invoices",
    description="Get list of invoices",
    schema=INVOICE_SCHEMAS["GET_INVOICES"],
    category="invoices"
)
async def get_invoices(context: Dict[str, Any]) -> Dict[str, Any]:
    """Get invoices with optional status filter."""
    tool_args = context["tool_args"]
    user_context = context.get("user_context", {})

    status_filter = tool_args.get("status", "all")
    invoices = user_context.get("invoices", [])

    if not invoices:
        return {
            "invoices": [],
            "count": 0,
            "total_amount": 0,
            "message": "No invoices found. Would you like to create one?"
        }

    # Filter by status
    filtered_invoices = invoices
    if status_filter != "all":
        if status_filter == "overdue":
            today = datetime.now().date()
            filtered_invoices = [
                inv for inv in invoices
                if inv.get("status") == "unpaid" and
                   parse_date(inv.get("due_date")) < today
            ]
        else:
            filtered_invoices = [
                inv for inv in invoices
                if inv.get("status") == status_filter
            ]

    # Calculate totals
    total_amount = sum(float(inv.get("amount", 0)) for inv in filtered_invoices)

    # Group by status
    status_counts = {
        "paid": len([i for i in filtered_invoices if i.get("status") == "paid"]),
        "unpaid": len([i for i in filtered_invoices if i.get("status") == "unpaid"]),
        "draft": len([i for i in filtered_invoices if i.get("status") == "draft"])
    }

    return {
        "invoices": filtered_invoices,
        "count": len(filtered_invoices),
        "total_amount": round(total_amount, 2),
        "status_counts": status_counts,
        "filter_applied": status_filter,
        "message": f"Found {len(filtered_invoices)} invoice{'s' if len(filtered_invoices) != 1 else ''} totaling ${total_amount:,.2f}"
    }

@tool_registry.register(
    name="send_invoice_reminder",
    description="Send a reminder for an unpaid invoice",
    schema=INVOICE_SCHEMAS["SEND_INVOICE_REMINDER"],
    category="invoices"
)
async def send_invoice_reminder(context: Dict[str, Any]) -> Dict[str, Any]:
    """Send reminder for unpaid invoice."""
    tool_args = context["tool_args"]
    invoice_id = tool_args.get("invoice_id")

    if not invoice_id:
        return {"error": "Invoice ID is required"}

    return {
        "action": "send_invoice_reminder",
        "invoice_id": invoice_id,
        "message": "I'll send a friendly payment reminder for this invoice."
    }

@tool_registry.register(
    name="mark_invoice_paid",
    description="Mark an invoice as paid",
    schema=INVOICE_SCHEMAS["MARK_INVOICE_PAID"],
    category="invoices"
)
async def mark_invoice_paid(context: Dict[str, Any]) -> Dict[str, Any]:
    """Mark invoice as paid."""
    tool_args = context["tool_args"]
    invoice_id = tool_args.get("invoice_id")

    if not invoice_id:
        return {"error": "Invoice ID is required"}

    return {
        "action": "mark_invoice_paid",
        "invoice_id": invoice_id,
        "message": "Great! I'll mark this invoice as paid.",
        "payment_date": datetime.now().strftime("%Y-%m-%d")
    }
