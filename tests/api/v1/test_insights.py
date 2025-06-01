"""
Insights endpoint tests.
"""

import pytest
from httpx import AsyncClient
from datetime import datetime, timedelta


@pytest.mark.asyncio
async def test_analyze_trends(client: AsyncClient):
    """Test trend analysis endpoint."""
    # Create test transactions
    transactions = []
    base_date = datetime.now() - timedelta(days=90)

    for i in range(90):
        transactions.append(
            {
                "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "amount": -50.0 - (i % 30),  # Increasing spending
                "category": "Food" if i % 3 == 0 else "Transport",
                "description": f"Transaction {i}",
            }
        )

    request_data = {
        "user_id": "test_user",
        "transactions": transactions,
        "period": "3m",
    }

    response = await client.post("/api/v1/insights/trends", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "spending_trends" in data
    assert "category_trends" in data
    assert "insights" in data
    assert len(data["category_trends"]) > 0


@pytest.mark.asyncio
async def test_financial_summary(client: AsyncClient):
    """Test financial summary endpoint."""
    transactions = [
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "amount": 5000.0,
            "category": "Salary",
        },
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "amount": -1500.0,
            "category": "Rent",
        },
    ]

    request_data = {
        "user_id": "test_user",
        "transactions": transactions,
        "include_insights": True,
    }

    response = await client.post("/api/v1/insights/summary", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert data["summary"]["total_income"] == 5000.0
    assert data["summary"]["total_expenses"] == 1500.0
    assert data["net_cash_flow"] == 3500.0
    assert "financial_health_score" in data
    assert "recommendations" in data
