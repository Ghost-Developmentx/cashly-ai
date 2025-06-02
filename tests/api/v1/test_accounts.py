"""
Account management endpoint tests.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_account_status(client: AsyncClient):
    """Test account status endpoint."""
    request_data = {
        "user_id": "test_user",
        "user_context": {
            "accounts": [
                {
                    "id": "acc_123",
                    "name": "Checking",
                    "account_type": "checking",
                    "balance": 5000.0,
                    "institution": "Test Bank",
                },
                {
                    "id": "acc_456",
                    "name": "Savings",
                    "account_type": "savings",
                    "balance": 10000.0,
                    "institution": "Test Bank",
                },
            ]
        },
    }

    response = await client.post("/api/v1/fin/accounts/status", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert data["has_accounts"] is True
    assert data["account_count"] == 2
    assert data["total_balance"] == 15000.0
    assert len(data["accounts"]) == 2


@pytest.mark.asyncio
async def test_initiate_plaid_connection(client: AsyncClient):
    """Test Plaid connection initiation."""
    request_data = {"user_id": "test_user", "institution_preference": "major_bank"}

    response = await client.post("/api/v1/fin/accounts/connect", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert data["action"] == "initiate_plaid_connection"
    assert data["user_id"] == "test_user"
    assert "message" in data
    assert "next_step" in data


@pytest.mark.asyncio
async def test_disconnect_account(client: AsyncClient):
    """Test account disconnection."""
    request_data = {
        "user_id": "test_user",
        "account_id": "acc_123",
        "reason": "No longer needed",
    }

    response = await client.post("/api/v1/fin/accounts/disconnect", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert data["action"] == "disconnect_account"
    assert data["account_id"] == "acc_123"
    assert data["requires_confirmation"] is True


@pytest.mark.asyncio
async def test_list_accounts(client: AsyncClient):
    """Test listing all accounts."""
    response = await client.get(
        "/api/v1/fin/accounts/", params={"user_id": "test_user"}
    )

    assert response.status_code == 200
    data = response.json()

    assert isinstance(data, list)
