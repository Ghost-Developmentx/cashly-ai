import pytest
from datetime import datetime

from app.services.fin.tool_handlers.analytics_handlers import AsyncAnalyticsHandlers


@pytest.mark.asyncio
async def test_filter_by_date_range():
    txns = [
        {"date": "2024-01-10", "amount": -10},
        {"date": "2024-02-05", "amount": -20},
    ]
    start = datetime(2024, 1, 1).date()
    end = datetime(2024, 1, 31).date()
    result = await AsyncAnalyticsHandlers._filter_by_date_range(txns, start, end)
    assert len(result) == 1
    assert result[0]["date"] == "2024-01-10"


@pytest.mark.asyncio
async def test_filter_by_categories():
    txns = [{"category": "Food"}, {"category": "Entertainment"}]
    result = await AsyncAnalyticsHandlers._filter_by_categories(txns, ["food"])
    assert result == [{"category": "Food"}]


@pytest.mark.asyncio
async def test_calculate_category_totals():
    txns = [
        {"amount": "-10", "category": "Food"},
        {"amount": "-5", "category": "Food"},
        {"amount": "20", "category": "Salary"},
    ]
    totals = await AsyncAnalyticsHandlers._calculate_category_totals(txns)
    assert totals["Food"] == 15
    assert "Salary" not in totals