import pytest
from unittest.mock import AsyncMock

from app.services.fin.async_tool_registry import AsyncToolRegistry


def _convert_transactions(df, n=30):
    records = df.head(n).to_dict('records')
    txns = []
    for r in records:
        txns.append({
            'date': r['date'].strftime('%Y-%m-%d') if hasattr(r['date'], 'strftime') else str(r['date']),
            'amount': float(r['amount']),
            'category': str(r.get('category', '')),
            'description': str(r.get('description', '')),
        })
    return txns


@pytest.mark.asyncio
async def test_forecast_with_existing_transactions(transaction_data):
    transactions = _convert_transactions(transaction_data)

    registry = AsyncToolRegistry()
    registry.rails_client.get = AsyncMock(return_value={'transactions': transactions})

    user_context = {'transactions': transactions, 'user_id': 'user123'}

    result = await registry.execute(
        'forecast_cash_flow',
        {'days': 7},
        user_id='user123',
        transactions=user_context['transactions'],
        user_context=user_context,
    )

    assert 'daily_forecast' in result
    registry.rails_client.get.assert_not_awaited()
    await registry.close()


@pytest.mark.asyncio
async def test_forecast_fetches_transactions_when_missing(transaction_data):
    transactions = _convert_transactions(transaction_data)

    registry = AsyncToolRegistry()
    registry.rails_client.get = AsyncMock(return_value={'transactions': transactions})

    user_context = {'transactions': [], 'user_id': 'user123'}

    result = await registry.execute(
        'forecast_cash_flow',
        {'days': 7},
        user_id='user123',
        transactions=user_context['transactions'],
        user_context=user_context,
    )

    assert 'daily_forecast' in result
    registry.rails_client.get.assert_awaited_once()
    await registry.close()