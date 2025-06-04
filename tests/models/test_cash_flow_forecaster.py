import pytest
import pandas as pd
import numpy as np
from app.models.forecasting.cash_flow_forecaster import CashFlowForecaster

def test_forecaster_init():
    """Test initialization of cash flow forecaster."""
    forecaster = CashFlowForecaster()
    assert forecaster.model_name == "cash_flow_forecaster_ensemble"
    assert forecaster.forecast_horizon == 30
    assert forecaster.method == "ensemble"

def test_forecaster_preprocess(time_series_data):
    """Test preprocessing of time series data."""
    forecaster = CashFlowForecaster()
    processed_data = forecaster.preprocess(time_series_data)

    # Check required columns exist
    assert 'date' in processed_data.columns
    assert 'daily_sum' in processed_data.columns

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(processed_data['date'])

def test_forecaster_feature_extraction(time_series_data):
    """Test feature extraction."""
    forecaster = CashFlowForecaster()
    processed_data = forecaster.preprocess(time_series_data)
    features = forecaster.extract_features(processed_data)

    # Check feature columns exist
    assert len(features.columns) > 3  # Should have multiple features

def test_forecaster_train(time_series_data):
    """Test training the forecaster."""
    forecaster = CashFlowForecaster()
    processed_data = forecaster.preprocess(time_series_data)

    # Create target variable (next day's value)
    X = processed_data.copy()
    y = X['daily_sum'].shift(-1).dropna()
    X = X.iloc[:-1]  # Remove last row as it has no target

    forecaster.train(X, y)

    # Check model is fitted
    assert forecaster.is_fitted
    assert forecaster.model is not None

    # Check metrics
    assert 'mae' in forecaster.metrics
    assert 'rmse' in forecaster.metrics

def test_forecaster_predict(time_series_data):
    """Test cash flow prediction."""
    forecaster = CashFlowForecaster()
    processed_data = forecaster.preprocess(time_series_data)

    # Create target variable (next day's value)
    X = processed_data.copy()
    y = X['daily_sum'].shift(-1).dropna()
    X = X.iloc[:-1]  # Remove last row as it has no target

    forecaster.train(X, y)

    # Test on same data
    predictions = forecaster.predict(X)

    # Check predictions format
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X)

def test_forecaster_forecast(time_series_data):
    """Test forecasting future cash flows."""
    forecaster = CashFlowForecaster()
    processed_data = forecaster.preprocess(time_series_data)

    # Create target variable (next day's value)
    X = processed_data.copy()
    y = X['daily_sum'].shift(-1).dropna()
    X = X.iloc[:-1]  # Remove last row as it has no target

    forecaster.train(X, y)

    # Test forecasting
    forecast_result = forecaster.forecast(processed_data, horizon=7)

    # Check forecast structure
    assert 'forecast' in forecast_result
    assert 'dates' in forecast_result
    assert len(forecast_result['forecast']) == 7  # 7-day forecast