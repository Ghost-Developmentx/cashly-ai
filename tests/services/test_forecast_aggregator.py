# tests/services/test_forecast_aggregator.py
import pytest
from app.services.forecast.forecast_aggregator import ForecastAggregator

@pytest.mark.asyncio
async def test_aggregator_handles_none_category():
    """Test aggregator handles None categories gracefully."""
    aggregator = ForecastAggregator()

    transactions = [
        {"date": "2024-01-01", "amount": 100, "category": None},
        {"date": "2024-01-02", "amount": -50, "category": "Food"}
    ]

    result = await aggregator.aggregate_transactions(transactions)
    assert result is not None
    assert "error" not in result