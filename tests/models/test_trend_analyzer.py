import pytest
import pandas as pd
import numpy as np
from app.models.trend_analysis.trend_analyzer import TrendAnalyzer

def test_trend_analyzer_init():
    """Test initialization of trend analyzer."""
    analyzer = TrendAnalyzer()
    assert analyzer.model_name == "trend_analyzer"
    assert analyzer.window_size == 30

def test_trend_analyzer_preprocess(time_series_data):
    """Test preprocessing of time series data."""
    analyzer = TrendAnalyzer()
    processed_data = analyzer.preprocess(time_series_data)

    # Check required columns exist
    assert 'date' in processed_data.columns
    assert 'daily_sum' in processed_data.columns

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(processed_data['date'])

def test_trend_analyzer_feature_extraction(time_series_data):
    """Test feature extraction."""
    analyzer = TrendAnalyzer()
    processed_data = analyzer.preprocess(time_series_data)
    features = analyzer.extract_features(processed_data)

    # Check feature columns exist
    assert len(features.columns) > 3  # Should have multiple features

def test_trend_analyzer_train(time_series_data):
    """Test training the trend analyzer."""
    analyzer = TrendAnalyzer()
    processed_data = analyzer.preprocess(time_series_data)
    analyzer.train(processed_data)

    # Check model is fitted
    assert analyzer.is_fitted
    assert analyzer.model is not None
    assert 'trends' in analyzer.model

    # Check metrics
    assert 'trend_strength' in analyzer.metrics
    assert 'r_squared' in analyzer.metrics

def test_trend_analyzer_predict(time_series_data):
    """Test trend prediction."""
    analyzer = TrendAnalyzer()
    processed_data = analyzer.preprocess(time_series_data)
    analyzer.train(processed_data)

    # Test on same data
    predictions = analyzer.predict(processed_data)

    # Check predictions format
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(processed_data)