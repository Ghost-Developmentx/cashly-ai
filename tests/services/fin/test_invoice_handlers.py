import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timedelta

from app.services.fin.tool_handlers.invoice_handlers import AsyncInvoiceHandlers


def test_get_due_date():
    result = AsyncInvoiceHandlers._get_due_date(None)
    due = datetime.strptime(result, "%Y-%m-%d").date()
    diff = (due - datetime.now().date()).days
    assert 29 <= diff <= 31

    future = (datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d")
    assert AsyncInvoiceHandlers._get_due_date(future) == future

    invalid = AsyncInvoiceHandlers._get_due_date("bad-date")
    invalid_due = datetime.strptime(invalid, "%Y-%m-%d").date()
    diff = (invalid_due - datetime.now().date()).days
    assert 29 <= diff <= 31


@pytest.mark.asyncio
async def test_create_invoice_success():
    rails = AsyncMock()
    rails.post = AsyncMock(return_value={"invoice_id": "inv1", "invoice": {"id": "inv1"}})
    handler = AsyncInvoiceHandlers(rails)
    context = {
        "user_id": "u1",
        "tool_args": {"client_name": "John", "client_email": "j@x.com", "amount": "10"},
    }
    result = await handler.create_invoice(context)
    rails.post.assert_awaited_once()
    assert result["success"] is True
    assert result["invoice_id"] == "inv1"


@pytest.mark.asyncio
async def test_create_invoice_error():
    rails = AsyncMock()
    rails.post = AsyncMock(return_value={"error": "fail"})
    handler = AsyncInvoiceHandlers(rails)
    context = {
        "user_id": "u1",
        "tool_args": {"client_name": "John", "client_email": "j@x.com", "amount": "10"},
    }
    result = await handler.create_invoice(context)
    assert result["success"] is False
    assert "error" in result