import pytest
from app.services.fin.tool_handlers.account_handlers import AsyncAccountHandlers


@pytest.mark.asyncio
async def test_get_user_accounts():
    context = {"user_context": {"accounts": [{"id": "1", "balance": 100}]}}
    result = await AsyncAccountHandlers.get_user_accounts(context)
    assert result["account_count"] == 1
    assert result["total_balance"] == 100
    assert result["has_accounts"] is True


@pytest.mark.asyncio
async def test_get_account_details():
    context = {
        "tool_args": {"account_id": "1"},
        "user_context": {"accounts": [{"id": "1", "balance": 50, "name": "Test", "account_type": "checking", "institution": "Bank"}]},
    }
    result = await AsyncAccountHandlers.get_account_details(context)
    assert result["account"]["id"] == "1"
    assert result["balance"] == 50
    assert result["name"] == "Test"
    assert result["institution"] == "Bank"


@pytest.mark.asyncio
async def test_disconnect_account():
    context = {"tool_args": {"account_id": "1"}, "user_id": "u1"}
    result = await AsyncAccountHandlers.disconnect_account(context)
    assert result["action"] == "disconnect_account"
    assert result["account_id"] == "1"
    assert result["requires_confirmation"] is True