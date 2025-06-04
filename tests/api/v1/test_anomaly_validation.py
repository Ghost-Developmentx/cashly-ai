# tests/api/v1/test_anomaly_validation.py
import pytest

@pytest.mark.asyncio
async def test_anomaly_detect_missing_date(client):
    """Test that missing date field returns 422."""
    request_data = {
        "user_id": "test_user",
        "transactions": [
            {
                # Missing 'date' field
                "amount": -100.0,
                "description": "Test transaction",
                "category": "Food"
            }
        ]
    }

    response = await client.post("/api/v1/anomaly/detect", json=request_data)
    assert response.status_code == 422
    assert "date" in response.text.lower()