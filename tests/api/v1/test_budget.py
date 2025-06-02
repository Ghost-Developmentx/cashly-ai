"""
Budget endpoint tests.
"""

import pytest
from datetime import datetime, timedelta


@pytest.mark.asyncio
async def test_generate_budget(client):
    """Test budget generation."""
    # Create test transactions
    transactions = []
    base_date = datetime.now() - timedelta(days=60)

    # Add income
    for i in range(2):
        transactions.append(
            {
                "date": (base_date + timedelta(days=i * 30)).strftime("%Y-%m-%d"),
                "amount": 5000.0,
                "category": "Income",
                "description": "Salary",
            }
        )

    # Add expenses
    categories = ["Food", "Transport", "Entertainment", "Bills"]
    for i in range(20):
        transactions.append(
            {
                "date": (base_date + timedelta(days=i * 3)).strftime("%Y-%m-%d"),
                "amount": -100.0 - (i * 10),
                "category": categories[i % len(categories)],
                "description": f"Expense {i}",
            }
        )

    request_data = {"user_id": "test_user", "transactions": transactions}

    response = await client.post("/api/v1/budget/generate", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert "monthly_income" in data
    assert "budget_allocations" in data
    assert "recommendations" in data
    assert "savings_potential" in data
    assert data["monthly_income"] > 0
    assert len(data["budget_allocations"]) > 0


@pytest.mark.asyncio
async def test_budget_validation(client):
    """Test budget data validation."""
    # Test with no transactions
    response = await client.post(
        "/api/v1/budget/generate", json={"user_id": "test_user", "transactions": []}
    )
    assert response.status_code == 422  # Validation error

    # Test with invalid transaction data
    response = await client.post(
        "/api/v1/budget/generate",
        json={
            "user_id": "test_user",
            "transactions": [{"date": "invalid-date", "amount": 0}],
        },
    )
    assert response.status_code == 422
