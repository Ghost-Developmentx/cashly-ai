import pytest
from datetime import datetime
from app.services.fin.tool_handlers.transaction_handlers import AsyncTransactionHandlers


@pytest.mark.asyncio
async def test_filter_transactions():
    handler = AsyncTransactionHandlers()
    txns = [
        {"id": 1, "date": "2024-01-10", "amount": 100, "account_id": "a1", "category": "Salary"},
        {"id": 2, "date": "2024-01-11", "amount": -20, "account_id": "a1", "category": "Food"},
        {"id": 3, "date": "2023-12-25", "amount": -5, "account_id": "a2", "category": "Food"},
    ]
    user_ctx = {"accounts": [{"id": "a1", "name": "Checking"}, {"id": "a2", "name": "Savings"}]}
    filtered = await handler._filter_transactions(
        transactions=txns,
        user_context=user_ctx,
        start_date="2024-01-01",
        end_date="2024-01-31",
        account_name="Checking",
        category="Food",
        type="expense",
    )
    assert len(filtered) == 1
    assert filtered[0]["id"] == 2


@pytest.mark.asyncio
async def test_prepare_updates():
    handler = AsyncTransactionHandlers()
    args = {"amount": "50", "description": "Lunch", "date": "2024-01-15"}
    updates = await handler._prepare_updates(args)
    assert updates["amount"] == 50
    assert updates["description"] == "Lunch"
    assert updates["date"] == "2024-01-15"

    bad = await handler._prepare_updates({"amount": "bad"})
    assert "error" in bad


def test_get_transaction_date():
    handler = AsyncTransactionHandlers()
    valid = handler._get_transaction_date("2024-01-01")
    assert valid == "2024-01-01"
    assert handler._get_transaction_date("bad") == ""
    today = datetime.now().strftime("%Y-%m-%d")
    assert handler._get_transaction_date(None) == today