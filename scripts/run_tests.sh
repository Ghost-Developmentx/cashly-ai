#!/bin/bash

echo "🧪 Running ML Forecast Integration Tests"
echo "======================================"

# Set test environment
export ENABLE_ML_FORECASTING=true
export ML_MIN_TRAINING_SAMPLES=30
export MODEL_DIR=test_models

# Run pytest
pytest tests/api/v1/test_ml_forecast_integration.py -v

# Cleanup
rm -rf test_models

echo ""
echo "✅ Tests completed"