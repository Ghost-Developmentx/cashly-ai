"""
Fin conversational AI endpoint tests.
"""

import pytest


@pytest.mark.asyncio
async def test_process_query(client):
    """Test natural language query processing."""
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
        },
    }

    response = await client.post("/api/v1/fin/conversations/query", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "message" in data
    assert "response_text" in data
    assert "classification" in data
    assert "actions" in data
    assert data["success"] is True


@pytest.mark.asyncio
async def test_query_with_conversation_history(client):
    """Test query with conversation history."""
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
