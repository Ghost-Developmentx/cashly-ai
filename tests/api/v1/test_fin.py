"""
Fin conversational AI endpoint tests.
"""
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_process_query(client, mocker):
    """Test natural language query processing."""
    from app.main import app
    from app.core.dependencies import get_openai_service

    # Create the mock service instance
    mock_service_instance = AsyncMock()
    mock_service_instance.process_financial_query = AsyncMock(return_value={
        "success": True,
        "message": "Here are your recent transactions",
        "response_text": "Here are your recent transactions",
        "actions": [],
        "tool_results": [],
        "classification": {
            "intent": "transactions",
            "confidence": 0.8,
            "assistant_used": "transaction",
            "method": "vector",
            "rerouted": False
        },
        "routing": {},
        "metadata": {}
    })

    # Override the dependency for this test
    def mock_get_openai_service():
        return mock_service_instance

    app.dependency_overrides[get_openai_service] = mock_get_openai_service

    try:
        request_data = {
            "user_id": "test_user",
            "query": "Show me my recent transactions",
            "user_context": {
                "user_id": "test_user",
                "accounts": [{"id": "acc_123", "name": "Checking", "balance": 5000.0}],
                "transactions": [
                    {
                        "date": "2024-01-20",
                        "amount": -50.0,
                        "description": "Coffee Shop",
                        "category": "Food",
                    }
                ],
                # Add missing stripe_connect to prevent the AttributeError
                "stripe_connect": {
                    "connected": False,
                    "can_accept_payments": False
                }
            },
        }

        response = await client.post("/api/v1/fin/conversations/query", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "recent transactions" in data["message"].lower()

    finally:
        # Clean up the override
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_query_with_conversation_history(client, mocker):
    """Test query with conversation history."""
    from app.main import app
    from app.core.dependencies import get_openai_service

    # Create the mock service instance
    mock_service_instance = AsyncMock()
    mock_service_instance.process_financial_query = AsyncMock(return_value={
        "success": True,
        "message": "Last month you spent $1,200 across 38 transactions.",
        "response_text": "Last month you spent $1,200 across 38 transactions.",
        "actions": [],
        "tool_results": [],
        "classification": {
            "intent": "general",
            "confidence": 0.8,
            "assistant_used": "general",
            "method": "conversation_context",
            "rerouted": False
        },
        "routing": {},
        "metadata": {},
        "conversation_id": "conv_test_user_123"
    })

    # Override the dependency for this test
    def mock_get_openai_service():
        return mock_service_instance

    app.dependency_overrides[get_openai_service] = mock_get_openai_service

    try:
        request_data = {
            "user_id": "test_user",
            "query": "What about last month?",
            "conversation_history": [
                {"role": "user", "content": "Show me my spending this month"},
                {
                    "role": "assistant",
                    "content": "You've spent $1,500 this month across 45 transactions.",
                },
            ],
        }

        response = await client.post("/api/v1/fin/conversations/query", json=request_data)

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "conversation_id" in data

    finally:
        # Clean up the override
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_fin_health_check(client):
    """Test OpenAI Assistant health check."""
    response = await client.get("/api/v1/fin/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert "components" in data
    assert "available_assistants" in data
    assert "missing_assistants" in data


@pytest.mark.asyncio
async def test_get_analytics(client):
    """Test analytics endpoint."""
    request_data = {
        "user_id": "test_user",
        "recent_queries": [
            "Show me my balance",
            "What did I spend on food?",
            "Create a budget for me",
        ],
    }

    response = await client.post("/api/v1/fin/analytics", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "intent_analytics" in data
    assert "assistant_usage" in data
    assert "performance_metrics" in data
    assert data["total_queries"] == 3


@pytest.mark.asyncio
async def test_clear_conversation(client):
    """Test clearing conversation history."""
    response = await client.delete("/api/v1/fin/conversations/test_user")

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["data"]["user_id"] == "test_user"
    assert "cleared_at" in data["data"]
