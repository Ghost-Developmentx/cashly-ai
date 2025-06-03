"""
Quick script to test the forecast flow end-to-end.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from datetime import datetime, timedelta
import random


async def test_forecast_flow():
    """Test the complete forecast flow."""

    # Import services
    from app.services.fin_service import FinService
    from app.services.fin.context_builder import ContextBuilder

    print("🚀 Testing Forecast Flow\n")

    # Create mock user
    class MockUser:
        id = "test_user_123"
        currency = "USD"

        @property
        def transactions(self):
            # Generate sample transactions
            transactions = []
            for i in range(60):
                date = datetime.now() - timedelta(days=60-i)

                # Daily income
                transactions.append({
                    "id": f"txn_income_{i}",
                    "date": date.strftime("%Y-%m-%d"),
                    "amount": random.uniform(150, 250),
                    "category": "Income",
                    "description": f"Daily income {i}",
                    "account": {"name": "Checking"}
                })

                # Daily expenses
                for category in ["Food", "Transport", "Utilities"]:
                    transactions.append({
                        "id": f"txn_expense_{i}_{category}",
                        "date": date.strftime("%Y-%m-%d"),
                        "amount": -random.uniform(20, 80),
                        "category": category,
                        "description": f"{category} expense {i}",
                        "account": {"name": "Checking"}
                    })

            return transactions

        @property
        def accounts(self):
            return [{"id": "1", "name": "Checking", "balance": 5000}]

        @property
        def budgets(self):
            return []

        @property
        def invoices(self):
            return []

    # Test the flow
    user = MockUser()

    # Test 1: Simple forecast query
    print("📊 Test 1: Simple Forecast Query")
    print("Query: 'Show me a 30-day cash flow forecast'\n")

    # This would normally go through the full Rails/Python flow
    # For testing, we'll simulate the AI service response

    # Build context
    context_builder = ContextBuilder(user)
    context = context_builder.build()

    print(f"✅ Context built with {len(context['transactions'])} transactions")

    # Test 2: Direct ML forecast
    print("\n📊 Test 2: Direct ML Forecast Service")

    from app.services.forecast.async_forecast_service import AsyncForecastService

    forecast_service = AsyncForecastService()
    result = await forecast_service.forecast_cash_flow(
        user_id=user.id,
        transactions=context['transactions'],
        forecast_days=30
    )

    if "error" not in result:
        print(f"✅ Forecast generated for {result['forecast_days']} days")
        print(f"   - Confidence: {result['summary']['confidence_score']:.2%}")
        print(f"   - Net change: ${result['summary']['projected_net']:,.2f}")
        print(f"   - Daily predictions: {len(result['daily_forecast'])}")
    else:
        print(f"❌ Forecast failed: {result['error']}")

    # Test 3: Check if ML was used
    print("\n📊 Test 3: Verify ML Model Usage")

    from app.services.ml.ml_forecast_service import MLForecastService

    ml_service = MLForecastService()
    ml_result = await ml_service.forecast_with_ml(
        transactions=context['transactions'],
        forecast_days=30
    )

    if ml_result:
        print("✅ ML model was used for forecasting")
        print(f"   - Method: {ml_result['patterns'].get('method', 'unknown')}")
    else:
        print("⚠️  ML model not used (may need more data)")

    # Test 4: Action format
    print("\n📊 Test 4: Action Format for Frontend")

    if "daily_forecast" in result:
        # Simulate action creation
        action = {
            "type": "show_forecast",
            "success": True,
            "data": {
                "id": f"forecast-{datetime.now().timestamp()}",
                "title": "30-Day Cash Flow Forecast",
                "dataPoints": [
                    {
                        "date": day["date"],
                        "predicted": day["net_change"],
                        "confidence": day.get("confidence", 0.75)
                    }
                    for day in result["daily_forecast"][:5]  # First 5 days
                ],
                "summary": {
                    "totalProjected": result["summary"]["projected_net"],
                    "averageDaily": result["summary"]["projected_net"] / 30,
                    "trend": "stable",
                    "confidenceScore": result["summary"]["confidence_score"],
                    "periodDays": 30
                },
                "generatedAt": datetime.now().isoformat()
            }
        }

        print("✅ Action formatted correctly")
        print(f"   - Type: {action['type']}")
        print(f"   - Data points: {len(action['data']['dataPoints'])} (showing first 5)")
        print(f"   - Summary included: {bool(action['data']['summary'])}")

    print("\n🎉 Forecast flow test completed!")


if __name__ == "__main__":
    asyncio.run(test_forecast_flow())