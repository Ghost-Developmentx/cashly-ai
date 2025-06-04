"""
Anomaly detection endpoint tests.
"""
from unittest.mock import AsyncMock

import pytest
from datetime import datetime, timedelta

from app.core.dependencies import get_anomaly_service


@pytest.mark.asyncio
async def test_detect_anomalies(client):
    """Test anomaly detection."""
    transactions = []

    # Normal transactions
    for i in range(20):
        transactions.append(
            {
                "id": f"txn_{i}",
                "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                "amount": -50.0 - (i % 10),
                "description": f"Normal transaction {i}",
                "category": "Food",
            }
        )

    # Add anomalies once
    transactions.extend(
        [
            {
                "id": "txn_anomaly_1",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "amount": -1500.0,
                "description": "Large purchase",
                "category": "Shopping",
            },
            {
                "id": "txn_anomaly_2",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "amount": -50.0,
                "description": "Duplicate transaction",
                "category": "Food",
            },
            {
                "id": "txn_anomaly_3",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "amount": -50.0,
                "description": "Duplicate transaction",
                "category": "Food",
            },
        ]
    )

    request_data = {
        "user_id": "test_user",
        "transactions": transactions,
        "threshold": 2.0,
        "check_duplicates": True,
    }

    response = await client.post("/api/v1/anomaly/detect", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "anomalies" in data
    assert len(data["anomalies"]) > 0
    assert "summary" in data
    assert data["summary"]["total_transactions"] == len(transactions)
    assert "recommendations" in data


@pytest.mark.asyncio
async def test_anomaly_summary(client):
    """Test anomaly summary endpoint."""
    response = await client.get(
        "/api/v1/anomaly/summary", params={"user_id": "test_user", "days": 30}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == "test_user"
    assert "summary" in data
    assert "recent_anomalies" in data


@pytest.mark.asyncio
async def test_mark_anomalies_reviewed(client):
    """Test marking anomalies as reviewed."""
    request_data = {
        "anomaly_ids": ["anomaly_1", "anomaly_2"],
        "user_id": "test_user",
        "action": "acknowledged"
    }

    response = await client.post(
        "/api/v1/anomaly/mark_reviewed",
        json=request_data,
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["data"]["updated_count"] == 2
    assert data["data"]["action"] == "acknowledged"


@pytest.mark.asyncio
async def test_anomaly_response_format(client, mocker):
    """Test anomaly detection handles a response format correctly."""
    from app.main import app

    # Create the mock service instance
    mock_service_instance = AsyncMock()
    mock_service_instance.detect_anomalies = AsyncMock(return_value={
        "anomalies": [{
            "transaction": {
                "id": "123",
                "date": "2024-01-01",
                "amount": -100.0,
                "description": "Test"
            },
            "anomaly_type": "unusual_amount",
            "severity": "high",
            "confidence": 0.9,
            "reason": "Test reason"
        }],
        "summary": {
            "total_transactions": 1,
            "anomalies_detected": 1,
            "anomaly_rate": 100.0,
            "by_type": {"unusual_amount": 1},
            "by_severity": {"high": 1},
            "highest_risk_categories": []
        },
        "threshold": 2.0
    })

    # Override the dependency for this test
    def mock_get_anomaly_service():
        return mock_service_instance

    app.dependency_overrides[get_anomaly_service] = mock_get_anomaly_service

    try:
        response = await client.post("/api/v1/anomaly/detect", json={
            "user_id": "test",
            "transactions": [{
                "date": "2024-01-01",
                "amount": -100.0,
                "description": "Test transaction",
                "category": "Food"
            }]
        })

        assert response.status_code == 200
        data = response.json()
        assert len(data["anomalies"]) == 1
        assert data["anomalies"][0]["transaction_date"] == "2024-01-01"

    finally:
        # Clean up the override
        app.dependency_overrides.clear()
