"""
Forecast endpoint tests.
"""

import pytest
from datetime import datetime, timedelta


@pytest.mark.asyncio
async def test_cash_flow_forecast(client):
    """Test basic cash flow forecast."""
    transactions = []
    base_date = datetime.now() - timedelta(days=30)

    # Create regular pattern
    for i in range(30):
        # Weekly income
        if i % 7 == 0:
            transactions.append(
                {
                    "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "amount": 1000.0,
                    "category": "Income",
                }
            )
        # Daily expenses
        transactions.append(
            {
                "date": (base_date + timedelta(days=i)).strftime("%Y-%m-%d"),
                "amount": -50.0 - (i % 20),
                "category": "Food",
            }
        )

    request_data = {
        "user_id": "test_user",
        "transactions": transactions,
        "forecast_days": 7,
    }

    response = await client.post("/api/v1/forecast/cash_flow", json=request_data)

    assert response.status_code == 200
    data = response.json()

    assert data["forecast_days"] == 7
    assert len(data["daily_forecast"]) == 7
    assert "summary" in data
    assert "historical_context" in data

    # Check daily forecast structure
    first_day = data["daily_forecast"][0]
    assert "predicted_income" in first_day
    assert "predicted_expenses" in first_day
    assert "net_change" in first_day
    assert "confidence" in first_day
    assert "running_balance" in first_day


@pytest.mark.asyncio
async def test_scenario_forecast(client):
    """Test scenario-based forecast."""
    transactions = [
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "amount": 1000.0,
            "category": "Income",
        }
    ]

    request_data = {
        "user_id": "test_user",
        "transactions": transactions,
        "forecast_days": 30,
        "adjustments": {"income_adjustment": 500.0, "expense_adjustment": -200.0},
    }

    response = await client.post(
        "/api/v1/forecast/cash_flow/scenario", json=request_data
    )

    assert response.status_code == 200
    data = response.json()
    assert "scenario" in data
    assert data["scenario"]["adjustments_applied"]["income_adjustment"] == 500.0
