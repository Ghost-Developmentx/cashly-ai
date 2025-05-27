from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging
import requests
import os

logger = logging.getLogger(__name__)

RAILS_API_URL = os.getenv("RAILS_API_URL")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "your-secure-internal-api-key")


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
    Create a new invoice synchronously by calling Rails API
    """
    logger.info(f"📧 Creating invoice with data: {invoice_data}")

    # Validate required fields
    required_fields = ["client_name", "client_email", "amount"]
    missing_fields = [field for field in required_fields if not invoice_data.get(field)]

    if missing_fields:
        return {
            "error": f"Missing required fields: {', '.join(missing_fields)}",
            "message": f"Cannot create invoice: missing {', '.join(missing_fields)}",
        }

    # Handle due date
    due_date = invoice_data.get("due_date")
    if not due_date:
        due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        logger.info(f"No due date provided, using: {due_date}")
    else:
        try:
            due_date_obj = datetime.strptime(due_date, "%Y-%m-%d")
            if due_date_obj < datetime.now():
                due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
                logger.info(f"Due date was in the past, updated to: {due_date}")
        except ValueError:
            due_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            logger.info(f"Invalid due date format, using: {due_date}")

    # Prepare the invoice data for Rails
    rails_invoice_data = {
        "user_id": user_id,
        "invoice": {
            "client_name": invoice_data.get("client_name"),
            "client_email": invoice_data.get("client_email"),
            "amount": float(invoice_data.get("amount", 0)),
            "description": invoice_data.get("description", ""),
            "due_date": due_date,
            "currency": invoice_data.get("currency", "USD"),
        },
    }

    # Make synchronous call to Rails
    try:
        response = requests.post(
            f"{RAILS_API_URL}/api/internal/invoices",
            json=rails_invoice_data,
            headers={
                "X-Internal-Api-Key": INTERNAL_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=10,
        )

        if response.status_code == 200:
            result = response.json()
            logger.info(
                f"✅ Invoice created successfully with ID: {result.get('invoice_id')}"
            )

            return {
                "invoice_id": result["invoice_id"],
                "invoice": result["invoice"],
                "platform_fee": result.get("platform_fee"),
                "message": f"I've created invoice #{result['invoice_id']} for {result['invoice']['client_name']} "
                f"for ${result['invoice']['amount']}. It's currently a draft. "
                f"When you're ready to send it, just say 'send invoice {result['invoice_id']}'.",
                "success": True,
            }
        else:
            error_data = response.json()
            logger.error(f"❌ Failed to create invoice: {error_data}")
            return {
                "error": error_data.get("error", "Failed to create invoice"),
                "message": "I couldn't create the invoice. Please check your Stripe Connect setup or try again.",
                "success": False,
            }

    except requests.exceptions.Timeout:
        logger.error("❌ Rails API timeout")
        return {
            "error": "Request timeout",
            "message": "The invoice creation is taking too long. Please try again.",
            "success": False,
        }
    except Exception as e:
        logger.error(f"❌ Error calling Rails API: {e}")
        return {
            "error": str(e),
            "message": "I encountered an error creating the invoice. Please try again.",
            "success": False,
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
