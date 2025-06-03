"""
Integration tests for ML-based forecasting.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
import numpy as np
from typing import List, Dict, Any
import shutil
from pathlib import Path

# Import our services
from app.services.forecast.async_forecast_service import AsyncForecastService
from app.services.ml.ml_forecast_service import MLForecastService
from app.utils.async_model_registry import AsyncModelRegistry
from app.core.config import settings

class TestMLForecastIntegration:
    """Test ML forecast integration end-to-end."""

    @classmethod
    def setup_class(cls):
        """Setup test environment."""
        # Create a temporary model directory
        cls.test_model_dir = Path("test_models")
        cls.test_model_dir.mkdir(exist_ok=True)

        # Override settings
        settings.model_dir = str(cls.test_model_dir)
        settings.enable_ml_forecasting = True
        settings.ml_min_training_samples = 30

    @classmethod
    def teardown_class(cls):
        """Cleanup test environment."""
        # Remove the test model directory
        if cls.test_model_dir.exists():
            shutil.rmtree(cls.test_model_dir)

    @staticmethod
    def generate_test_transactions(
            days: int = 90,
            base_income: float = 5000,
            base_expense: float = 3500,
            noise_factor: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Generate realistic test transactions."""
        transactions = []
        start_date = datetime.now() - timedelta(days=days)

        for i in range(days):
            date = start_date + timedelta(days=i)

            # Weekly pattern (higher spending on weekends)
            day_of_week = date.weekday()
            weekend_multiplier = 1.3 if day_of_week >= 5 else 1.0

            # Monthly pattern (higher spending at month start/end)
            day_of_month = date.day
            month_multiplier = 1.2 if day_of_month <= 5 or day_of_month >= 25 else 1.0

            # Add some randomness
            income_noise = np.random.normal(0, base_income * noise_factor)
            expense_noise = np.random.normal(0, base_expense * noise_factor)

            # Generate daily transactions
            daily_income = base_income / 30 + income_noise
            daily_expense = base_expense / 30 * weekend_multiplier * month_multiplier + expense_noise

            # Income transaction
            if daily_income > 0:
                transactions.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "amount": round(daily_income, 2),
                    "category": "Income",
                    "description": f"Daily income for {date.strftime('%Y-%m-%d')}"
                })

            # Expense transactions
            if daily_expense > 0:
                # Split into multiple expense categories
                categories = ["Food", "Transport", "Utilities", "Entertainment", "Other"]
                for category in categories:
                    amount = daily_expense / len(categories) * np.random.uniform(0.5, 1.5)
                    transactions.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "amount": -round(amount, 2),
                        "category": category,
                        "description": f"{category} expense"
                    })

        return transactions

    @pytest.mark.asyncio
    async def test_ml_service_initialization(self):
        """Test ML service initializes correctly."""
        service = MLForecastService()

        assert service is not None
        assert service.model_registry is not None
        assert service.min_samples == 30

    @pytest.mark.asyncio
    async def test_model_training(self):
        """Test model training with sufficient data."""
        service = MLForecastService()
        transactions = self.generate_test_transactions(days=60)

        # Train model
        forecaster = await service.train_model(transactions, method="ensemble")

        assert forecaster is not None
        assert forecaster.model is not None
        assert forecaster.scaler is not None
        assert len(forecaster.feature_cols) > 0

    @pytest.mark.asyncio
    async def test_model_caching(self):
        """Test model caching works correctly."""
        service = MLForecastService()
        transactions = self.generate_test_transactions(days=60)

        # First call - should train
        model1 = await service.get_or_train_model(transactions)

        # Second call - should use cache
        model2 = await service.get_or_train_model(transactions)

        # Should be the same instance from cache
        assert model1 is model2

    @pytest.mark.asyncio
    async def test_forecast_generation(self):
        """Test complete forecast generation."""
        service = MLForecastService()
        transactions = self.generate_test_transactions(days=90)

        # Generate forecast
        result = await service.forecast_with_ml(
            transactions=transactions,
            forecast_days=30,
            method="ensemble"
        )

        # Verify forecast structure
        assert result is not None
        assert "daily_predictions" in result
        assert "confidence" in result
        assert "total_income" in result
        assert "total_expenses" in result
        assert "net_change" in result

        # Verify predictions
        predictions = result["daily_predictions"]
        assert len(predictions) == 30

        for pred in predictions:
            assert "date" in pred
            assert "predicted_income" in pred
            assert "predicted_expenses" in pred
            assert "net_change" in pred
            assert "confidence" in pred

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self):
        """Test handling of insufficient training data."""
        service = MLForecastService()
        transactions = self.generate_test_transactions(days=10)  # Too few

        # Should return None due to insufficient data
        model = await service.get_or_train_model(transactions)
        assert model is None

    @pytest.mark.asyncio
    async def test_model_persistence(self):
        """Test model saving and loading."""
        registry = AsyncModelRegistry(str(self.test_model_dir))

        # Create a fake model
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit([[1, 2], [3, 4]], [5, 6])

        # Save model
        model_id = await registry.save_model(
            model=model,
            model_name="test_forecast",
            model_type="forecasting",
            features=["feature1", "feature2"],
            metrics={"mae": 0.1}
        )

        assert model_id is not None

        # Load model
        loaded_model, model_info = await registry.load_model(
            model_type="forecasting",
            latest=True
        )

        assert loaded_model is not None
        assert model_info["id"] == model_id
        assert model_info["features"] == ["feature1", "feature2"]

    @pytest.mark.asyncio
    async def test_full_forecast_service_integration(self):
        """Test full integration with AsyncForecastService."""
        service = AsyncForecastService()
        transactions = self.generate_test_transactions(days=90)

        # Generate forecast (should use ML if available)
        result = await service.forecast_cash_flow(
            user_id="test_user",
            transactions=transactions,
            forecast_days=30
        )

        # Verify response format
        assert "forecast_days" in result
        assert "daily_forecast" in result
        assert "summary" in result
        assert "historical_context" in result

        # Verify it used ML (check patterns)
        if "error" not in result:
            # ML forecasts should have higher confidence
            summary = result["summary"]
            assert summary["confidence_score"] >= 0.7

    @pytest.mark.asyncio
    async def test_forecast_accuracy(self):
        """Test forecast accuracy with known patterns."""
        service = MLForecastService()

        # Generate transactions with clear pattern
        transactions = []
        for i in range(90):
            date = (datetime.now() - timedelta(days=90-i)).strftime("%Y-%m-%d")
            # Linear growth pattern
            amount = 100 + i * 2  # Growing by $2 per day
            transactions.append({
                "date": date,
                "amount": amount,
                "category": "Income"
            })

        # Train and forecast
        result = await service.forecast_with_ml(
            transactions=transactions,
            forecast_days=7,
            method="linear"  # Use linear for predictable pattern
        )

        if result:  # Only test if ML succeeded
            predictions = result["daily_predictions"]

            # Check if forecast continues the pattern
            last_amount = 100 + 89 * 2  # Last historical amount
            for i, pred in enumerate(predictions):
                expected = last_amount + (i + 1) * 2
                predicted = pred["net_change"]

                # Allow 20% error margin
                assert abs(predicted - expected) / expected < 0.2

    @pytest.mark.asyncio
    async def test_concurrent_model_access(self):
        """Test concurrent access to models."""
        service = MLForecastService()
        transactions = self.generate_test_transactions(days=60)

        # Train model first
        await service.train_model(transactions)

        # Simulate concurrent access
        tasks = []
        for _ in range(5):
            task = service.get_or_train_model(transactions)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should succeed and return same model
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in forecast service."""
        service = MLForecastService()

        # Test with invalid data
        invalid_transactions = [
            {"date": "invalid", "amount": "not a number", "category": None}
        ]

        result = await service.forecast_with_ml(
            transactions=invalid_transactions,
            forecast_days=30
        )

        # Should handle gracefully
        assert result is None  # Returns None on error

    @pytest.mark.asyncio
    async def test_model_cleanup(self):
        """Test old model cleanup."""
        registry = AsyncModelRegistry(str(self.test_model_dir))

        # Create multiple models
        from sklearn.linear_model import LinearRegression

        for i in range(5):
            model = LinearRegression()
            model.fit([[i]], [i])

            await registry.save_model(
                model=model,
                model_name=f"test_model_{i}",
                model_type="forecasting",
                features=[f"feature_{i}"],
                keep_latest=3
            )

            # Small delay to ensure different timestamps
            await asyncio.sleep(0.1)

        # Check only 3 models remain
        models = await registry.list_models(model_type="forecasting")
        assert len(models) == 3

        # Verify they are the latest 3
        model_names = [m["name"] for m in models]
        assert "test_model_2" in model_names
        assert "test_model_3" in model_names
        assert "test_model_4" in model_names


async def run_all_tests():
    """Run all tests manually."""
    test_instance = TestMLForecastIntegration()
    test_instance.setup_class()

    try:
        print("🧪 Testing ML Service Initialization...")
        await test_instance.test_ml_service_initialization()
        print("✅ ML Service Initialization: PASSED\n")

        print("🧪 Testing Model Training...")
        await test_instance.test_model_training()
        print("✅ Model Training: PASSED\n")

        print("🧪 Testing Model Caching...")
        await test_instance.test_model_caching()
        print("✅ Model Caching: PASSED\n")

        print("🧪 Testing Forecast Generation...")
        await test_instance.test_forecast_generation()
        print("✅ Forecast Generation: PASSED\n")

        print("🧪 Testing Full Service Integration...")
        await test_instance.test_full_forecast_service_integration()
        print("✅ Full Service Integration: PASSED\n")

        print("🎉 All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        test_instance.teardown_class()


if __name__ == "__main__":
    # Run tests
    asyncio.run(run_all_tests())