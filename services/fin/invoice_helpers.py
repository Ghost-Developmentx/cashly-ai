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
    Create a new invoice
    """
    return {
        "action": "create_invoice",
        "invoice": invoice_data,
        "message": f"I'll create an invoice for {invoice_data.get('client_name')} for {invoice_data.get('amount')}.",
    }


def get_invoices(user_id: str, **filters) -> Dict[str, Any]:
    """
    Retrieve user invoices with optional filters
    """
    # This would normally query the database
    # For now, return a structure for the Rails backend to handle
    return {"action": "get_invoices", "filters": filters}


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
    Mark an invoice as paid
    """
    return {
        "action": "mark_invoice_paid",
        "invoice_id": invoice_id,
        "message": "I'll mark this invoice as paid.",
    }
