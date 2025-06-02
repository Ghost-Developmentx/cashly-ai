"""
Anomaly detection endpoint tests.
"""

import pytest
from datetime import datetime, timedelta


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
    }

    response = await client.post(
        "/api/v1/anomaly/mark_reviewed",
        json=request_data,
        params={"action": "acknowledged"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["success"] is True
    assert data["data"]["updated_count"] == 2
    assert data["data"]["action"] == "acknowledged"
