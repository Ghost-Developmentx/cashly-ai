"""
Test configuration and fixtures.
"""
import tempfile
from unittest.mock import AsyncMock

import pytest
from typing import AsyncGenerator

import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession
import sys
from pathlib import Path
import pandas as pd
from app.utils.synthetic_data.transaction_generator import TransactionGenerator

from app.main import app
from app.core.dependencies import get_db

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Setup test environment variables."""
    # Create temporary SQLite databases for testing
    test_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    test_mlflow_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)

    monkeypatch.setenv("ENVIRONMENT", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{test_db.name}")
    monkeypatch.setenv("ASYNC_DATABASE_URL", f"sqlite+aiosqlite:///{test_db.name}")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{test_mlflow_db.name}")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test-access-key")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-secret-key")
    monkeypatch.setenv("RAILS_API_URL", "http://test-rails-api")
    monkeypatch.setenv("INTERNAL_API_KEY", "test-internal-key")

@pytest.fixture
def mock_openai_service():
    """Mock OpenAI service for tests."""
    mock_service = AsyncMock()
    mock_service.process_financial_query = AsyncMock(return_value={
        "success": True,
        "message": "Test response",
        "response_text": "Test response",
        "actions": [],
        "tool_results": [],
        "classification": {
            "intent": "general",
            "confidence": 0.8,
            "assistant_used": "test",
            "method": "test",
            "rerouted": False
        },
        "routing": {},
        "metadata": {}
    })
    mock_service.health_check = AsyncMock(return_value={
        "status": "healthy",
        "components": {},
        "summary": {
            "available_assistants": ["test"],
            "missing_assistants": []
        }
    })
    return mock_service

@pytest_asyncio.fixture
async def client(mock_openai_service):
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(autouse=True)
def mock_external_services(monkeypatch):
    """Mock external services for testing."""

    # Mock OpenAI
    class MockOpenAI:
        def __init__(self, *args, **kwargs):
            pass

        class CHAT:
            class COMPLETIONS:
                @staticmethod
                def create(*args, **kwargs):
                    class MockResponse:
                        class MockChoice:
                            class MockMessage:
                                content = '{"insights": [], "summary": "Test summary"}'

                        choices = [MockChoice()]

                    return MockResponse()

    monkeypatch.setattr("openai.OpenAI", MockOpenAI)

    # Mock MLflow with proper return values
    class MockModelInfo:
        def __init__(self):
            self.model_uri = "test://mock_model_uri"
            self.run_id = "mock_run_id"

    class MockMLflow:
        @staticmethod
        def log_model(*args, **kwargs):
            return MockModelInfo()  # Return proper mock object instead of None

        @staticmethod
        def load_model(*args, **kwargs):
            return None

    # Mock MLflow modules
    monkeypatch.setattr("mlflow.sklearn.log_model", MockMLflow.log_model)
    monkeypatch.setattr("mlflow.sklearn.load_model", MockMLflow.load_model)
    monkeypatch.setattr("mlflow.pyfunc.log_model", MockMLflow.log_model)
    monkeypatch.setattr("mlflow.pyfunc.load_model", MockMLflow.load_model)




@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async for session in get_db():
        yield session

@pytest.fixture
def transaction_data():
    """Generate synthetic transaction data for testing."""
    generator = TransactionGenerator()
    data = generator.generate_transactions(
        num_days=90,
        transactions_per_day=(8, 20),  # Increased significantly
        anomaly_rate=0.05
    )

    # Ensure we have meaningful descriptions and amounts
    if 'description' not in data.columns:
        data['description'] = data['merchant'] + ' ' + data['category']

    # Ensure no empty or null descriptions
    data['description'] = data['description'].fillna('Transaction').astype(str)
    data = data[data['description'].str.len() > 0]

    return data


@pytest.fixture
def categorized_data(transaction_data):
    """Transaction data with categories and proper distribution."""
    data = transaction_data.copy()

    # Ensure we have multiple categories with sufficient samples
    categories = ['Food', 'Transportation', 'Entertainment', 'Shopping', 'Bills', 'Income']

    # Assign categories to ensure each has at least 3 samples
    total_rows = len(data)
    samples_per_category = max(3, total_rows // len(categories))

    category_list = []
    for i, category in enumerate(categories):
        start_idx = i * samples_per_category
        end_idx = min((i + 1) * samples_per_category, total_rows)
        category_list.extend([category] * (end_idx - start_idx))

    # Fill remaining rows if any
    while len(category_list) < total_rows:
        category_list.append(categories[len(category_list) % len(categories)])

    data['category'] = category_list[:total_rows]
    return data


@pytest.fixture
def time_series_data(transaction_data):
    """Aggregated time series data for forecasting."""
    df = transaction_data.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Create a proper daily aggregation that matches model expectations
    daily_agg = df.groupby(df['date'].dt.date).agg({
        'amount': ['sum', 'count', 'mean', 'std']
    }).fillna(0)

    # Flatten column names
    daily_agg.columns = ['daily_sum', 'transaction_count', 'avg_amount', 'amount_std']
    daily_agg = daily_agg.reset_index()
    daily_agg['date'] = pd.to_datetime(daily_agg['date'])

    # Add required columns for compatibility
    daily_agg['amount'] = daily_agg['daily_sum']

    return daily_agg

