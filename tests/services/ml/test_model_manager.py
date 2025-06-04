import pytest
import pandas as pd
import asyncio
from app.services.ml.model_manager import ModelManager, model_manager

@pytest.fixture
def model_manager_instance():
    """Create a fresh model manager instance."""
    manager = ModelManager()
    return manager

@pytest.mark.asyncio
async def test_model_manager_get_model(model_manager_instance):
    """Test getting a model from the manager."""
    # Test each model type
    model_types = ['forecaster', 'categorizer', 'trend_analyzer', 'budget_recommender', 'anomaly_detector']

    for model_type in model_types:
        model = await model_manager_instance.get_model(model_type)
        assert model is not None

        # Check model is cached
        assert model_type in model_manager_instance._models
        assert model_type in model_manager_instance._model_timestamps

@pytest.mark.asyncio
async def test_model_manager_force_reload(model_manager_instance):
    """Test force reloading a model."""
    # First load
    model1 = await model_manager_instance.get_model('categorizer')

    # Force reload
    model2 = await model_manager_instance.get_model('categorizer', force_reload=True)

    # Should be different instances
    assert model1 is not model2

@pytest.mark.asyncio
async def test_model_manager_clear_cache(model_manager_instance):
    """Test clearing the model cache."""
    # Load a model
    await model_manager_instance.get_model('forecaster')
    assert 'forecaster' in model_manager_instance._models

    # Clear cache
    model_manager_instance.clear_cache()
    assert 'forecaster' not in model_manager_instance._models
    assert 'forecaster' not in model_manager_instance._model_timestamps

@pytest.mark.asyncio
async def test_model_manager_train_model(model_manager_instance, transaction_data):
    """Test training a model through the manager."""
    # Train categorizer
    result = await model_manager_instance.train_model(
        'categorizer',
        transaction_data
    )

    assert result['success'] is True
    assert result['model_type'] == 'categorizer'
    assert 'metrics' in result