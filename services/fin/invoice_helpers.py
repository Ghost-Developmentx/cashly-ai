from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def connect_stripe_account(user_id: str, api_key: str) -> Dict[str, Any]:
    """
    Initiate Stripe account connection
    """
    return {
        "action": "connect_stripe",
        "user_id": user_id,
        "api_key": api_key,
        "message": "I'll connect your Stripe account for you. This will allow you to send invoices and track payments directly through our chat.",
        "requires_confirmation": True,
    }


def create_invoice(user_id: str, invoice_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new invoice with proper due date validation
    """
    logger.info(f"📧 Creating invoice with data: {invoice_data}")

    # Validate required fields
    required_fields = ["client_name", "client_email", "amount"]
    missing_fields = [field for field in required_fields if not invoice_data.get(field)]

    if missing_fields:
        return {
            "action": "create_invoice",
            "error": f"Missing required fields: {', '.join(missing_fields)}",
            "message": f"Cannot create invoice: missing {', '.join(missing_fields)}",
        }

    # Handle due date - if not provided or in the past, use 30 days from now
    due_date = invoice_data.get("due_date")
    if not due_date:
        # Default to 30 days from now
        due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        logger.info(f"No due date provided, using: {due_date}")
    else:
        # Check if the provided date is in the past
        try:
            due_date_obj = datetime.strptime(due_date, "%Y-%m-%d")
            if due_date_obj < datetime.now():
                due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                logger.info(f"Due date was in the past, updated to: {due_date}")
        except ValueError:
            due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            logger.info(f"Invalid due date format, using: {due_date}")

    # Prepare the cleaned invoice data
    cleaned_invoice_data = {
        "client_name": invoice_data.get("client_name"),
        "client_email": invoice_data.get("client_email"),
        "amount": float(invoice_data.get("amount", 0)),
        "description": invoice_data.get("description", ""),
        "due_date": due_date,
    }

    return {
        "action": "create_invoice",
        "invoice": cleaned_invoice_data,
        "message": f"Creating invoice for {cleaned_invoice_data['client_name']} for ${cleaned_invoice_data['amount']} (due: {due_date})",
    }


def send_invoice(user_id: str, invoice_id: str) -> Dict[str, Any]:
    """
    Send a draft invoice to the client.
    """
    logger.info(f"📧 Sending invoice {invoice_id} for user {user_id}")

    return {
        "action": "send_invoice",
        "invoice_id": invoice_id,
        "user_id": user_id,
        "message": f"Sending invoice {invoice_id} to the client now...",
    }


def get_invoices(user_id: str, **filters) -> Dict[str, Any]:
    """
    Retrieve user invoices with optional filters.
    This will trigger the Rails backend to fetch real invoices.
    """
    logger.info(f"📋 Getting invoices for user {user_id} with filters: {filters}")

    return {"action": "get_invoices", "filters": filters, "user_id": user_id}


def send_invoice_reminder(user_id: str, invoice_id: str) -> Dict[str, Any]:
    """
    Send a payment reminder for an invoice
    """
    return {
        "action": "send_invoice_reminder",
        "invoice_id": invoice_id,
        "message": "I'll send a payment reminder for this invoice right away.",
    }


def mark_invoice_paid(user_id: str, invoice_id: str) -> Dict[str, Any]:
    """
    Mark an invoice as paid.
    """
    logger.info(f"✅ Marking invoice {invoice_id} as paid for user {user_id}")

    return {
        "action": "mark_invoice_paid",
        "invoice_id": invoice_id,
        "user_id": user_id,
        "message": f"I'll mark invoice {invoice_id} as paid.",
    }
