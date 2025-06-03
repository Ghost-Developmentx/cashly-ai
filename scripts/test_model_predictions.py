#!/usr/bin/env python
"""
Quick test of model predictions.
"""

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.services.categorize.async_categorize_service import AsyncCategorizationService
from app.services.forecast.async_forecast_service import AsyncForecastService
from app.services.anomaly.async_anomaly_service import AsyncAnomalyService
from app.utils.synthetic_data.transaction_generator import TransactionGenerator


async def test_all_services():
    """Test all ML services."""

    # Generate test data
    gen = TransactionGenerator()
    test_transactions = gen.generate_transactions(num_days=30)
    transaction_list = test_transactions.to_dict('records')

    print("🧪 Testing ML Services\n")

    # Test categorization
    print("1️⃣ Testing Categorization Service:")
    cat_service = AsyncCategorizationService()

    test_txn = transaction_list[0]
    result = await cat_service.categorize_transaction(
        description=test_txn['description'],
        amount=test_txn['amount']
    )
    print(f"   Input: {test_txn['description']} (${test_txn['amount']})")
    print(f"   Predicted: {result['category']} (confidence: {result['confidence']:.2f})")
    print(f"   Method: {result['method']}\n")

    # Test forecasting
    print("2️⃣ Testing Forecast Service:")
    forecast_service = AsyncForecastService()

    forecast = await forecast_service.forecast_cash_flow(
        user_id='test_user',
        transactions=transaction_list,
        forecast_days=7
    )
    print(f"   Forecast days: {forecast['forecast_days']}")
    print(f"   Confidence: {forecast['summary']['confidence_score']:.2f}")
    print(f"   Projected net: ${forecast['summary']['projected_net']:.2f}\n")

    # Test anomaly detection
    print("3️⃣ Testing Anomaly Detection:")
    anomaly_service = AsyncAnomalyService()

    anomalies = await anomaly_service.detect_anomalies(
        user_id='test_user',
        transactions=transaction_list
    )
    print(f"   Transactions analyzed: {anomalies['total_transactions']}")
    print(f"   Anomalies detected: {anomalies['summary']['anomaly_count']}")
    print(f"   Risk level: {anomalies['summary']['risk_level']}\n")

    print("✅ All services tested successfully!")


if __name__ == "__main__":
    asyncio.run(test_all_services())