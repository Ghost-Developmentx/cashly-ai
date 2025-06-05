import pytest
from app.services.fin.tool_handlers.stripe_handlers import AsyncStripeHandlers
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_setup_stripe_new_connection():
    handler = AsyncStripeHandlers(AsyncMock())
    context = {"tool_args": {}, "user_context": {"stripe_connect": {}}, "user_id": "u"}
    result = await handler.setup_stripe_connect(context)
    assert result["action"] == "setup_stripe_connect"
    assert result["user_id"] == "u"


@pytest.mark.asyncio
async def test_setup_stripe_already_connected():
    handler = AsyncStripeHandlers(AsyncMock())
    context = {
        "tool_args": {},
        "user_context": {"stripe_connect": {"connected": True, "can_accept_payments": True, "status": "active"}},
        "user_id": "u",
    }
    result = await handler.setup_stripe_connect(context)
    assert result["already_connected"] is True
    assert result["can_accept_payments"] is True


@pytest.mark.asyncio
async def test_check_stripe_status_not_connected():
    handler = AsyncStripeHandlers(AsyncMock())
    context = {"user_context": {"stripe_connect": {}}, "user_id": "u"}
    result = await handler.check_stripe_connect_status(context)
    assert result["connected"] is False
    assert result["setup_recommended"] is True


@pytest.mark.asyncio
async def test_check_stripe_status_active():
    handler = AsyncStripeHandlers(AsyncMock())
    context = {
        "user_context": {
            "stripe_connect": {"connected": True, "status": "active", "charges_enabled": True}
        },
        "user_id": "u",
    }
    result = await handler.check_stripe_connect_status(context)
    assert result["connected"] is True
    assert result["message"].startswith("Your Stripe Connect account is active")


@pytest.mark.asyncio
async def test_disconnect_stripe_connect():
    context = {"user_context": {"stripe_connect": {"connected": True}}, "user_id": "u"}
    result = await AsyncStripeHandlers.disconnect_stripe_connect(context)
    assert result["action"] == "disconnect_stripe_connect"
    assert result["requires_confirmation"] is True


@pytest.mark.asyncio
async def test_disconnect_stripe_connect_no_account():
    context = {"user_context": {"stripe_connect": {}}, "user_id": "u"}
    result = await AsyncStripeHandlers.disconnect_stripe_connect(context)
    assert "error" in result