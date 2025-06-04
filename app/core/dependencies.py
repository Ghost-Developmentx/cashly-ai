"""
Dependency injection container.
Provides shared dependencies for FastAPI routes.
"""

from typing import AsyncGenerator, Optional
from functools import lru_cache
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import AsyncMock
import os

from app.core.config import settings
from app.db.async_db.connection import AsyncDatabaseConnection
from app.db.singleton_registry import registry

# Service imports
from app.services.categorize import AsyncCategorizationService
from app.services.forecast import AsyncForecastService
from app.services.budget import AsyncBudgetService
from app.services.insights import AsyncInsightService
from app.services.anomaly import AsyncAnomalyService
from app.services.openai_assistants import OpenAIIntegrationService


# Database Dependencies
async def get_db_connection() -> AsyncDatabaseConnection:
    """Get database connection from a singleton registry."""

    async def create_connection():
        from app.core.config import get_settings
        return AsyncDatabaseConnection(get_settings())

    return await registry.get_or_create("async_db_connection", create_connection)



async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Database session dependency.
    Yields an async session and handles cleanup.
    """
    db_conn = await get_db_connection()
    async with db_conn.get_session() as session:
        yield session


# Service Dependencies (using lru_cache for singletons)
@lru_cache()
def get_categorization_service() -> AsyncCategorizationService:
    """Get categorization service singleton."""
    return AsyncCategorizationService()


@lru_cache()
def get_forecast_service() -> AsyncForecastService:
    """Get forecast service singleton."""
    return AsyncForecastService()


@lru_cache()
def get_budget_service() -> AsyncBudgetService:
    """Get budget service singleton."""
    return AsyncBudgetService()


@lru_cache()
def get_insight_service() -> AsyncInsightService:
    """Get insight service singleton."""
    return AsyncInsightService()


@lru_cache()
def get_anomaly_service() -> AsyncAnomalyService:
    """Get anomaly service singleton."""
    return AsyncAnomalyService()


def get_openai_service() -> OpenAIIntegrationService:
    """Get OpenAI integration service (mocked in tests)."""
    if settings.testing or os.getenv("TESTING", "false").lower() == "true":
        # Return mock for tests
        mock = AsyncMock()
        mock.health_check = AsyncMock(return_value={
            "status": "healthy",
            "components": {"assistant": {"status": "healthy"}},
            "summary": {
                "available_assistants": ["test_assistant"],
                "missing_assistants": []
            }
        })
        mock.process_financial_query = AsyncMock(return_value={
            "success": True,
            "message": "Test response",
            "response_text": "Test response",
            "actions": [],
            "tool_results": [],
            "classification": {
                "intent": "test_intent",
                "confidence": 0.9,
                "assistant_used": "test_assistant",
                "method": "test",
                "rerouted": False
            },
            "routing": {},
            "metadata": {}
        })
        return mock

    return OpenAIIntegrationService()


# Optional Dependencies
async def get_current_user_id(
    # In production, extract from JWT/Auth header
    user_id: Optional[str] = None,
) -> str:
    """
    Get current user ID from request.
    In production, this would extract from auth token.
    """
    if not user_id:
        # For now, allow passing user_id in request
        # Later: Extract from Clerk JWT or session
        return "default_user"
    return user_id
