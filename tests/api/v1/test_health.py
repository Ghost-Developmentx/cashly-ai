"""
Health endpoint tests.
"""

import pytest


@pytest.mark.asyncio
async def test_basic_health_check(client):
    """Test basic health endpoint."""
    response = await client.get("/api/v1/health/")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Cashly AI Service"
    assert "timestamp" in data


@pytest.mark.asyncio
async def test_detailed_health_check(client):
    """Test detailed health endpoint."""
    response = await client.get("/api/v1/health/detailed")

    assert response.status_code == 200
    # Add a debug log here
    print(f"Raw response content: {response.content}")
    print(f"Type: {type(response.content)}")

    try:
        data = response.json()
    except Exception as e:
        print(f"⚠️ Failed to parse JSON: {e}")
        raise
    assert data["status"] in ["healthy", "degraded", "unhealthy"]
    assert "components" in data
    assert "database" in data["components"]
    assert "uptime" in data
