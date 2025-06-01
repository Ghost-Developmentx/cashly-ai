"""
Health endpoint tests.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_basic_health_check(client: AsyncClient):
    """Test basic health endpoint."""
    response = await client.get("/api/v1/health/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "cashly-ai-service"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_detailed_health_check(client: AsyncClient):
    """Test detailed health endpoint."""
    response = await client.get("/api/v1/health/detailed")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    assert "components" in data
    assert "database" in data["components"]
    assert "uptime" in data
