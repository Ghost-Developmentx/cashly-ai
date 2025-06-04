import pytest
import pandas as pd
import numpy as np
from app.models.anomaly.anomaly_detector import AnomalyDetector

def test_anomaly_detector_init():
    """Test initialization of anomaly detector."""
    detector = AnomalyDetector()
    assert detector.model_name == "anomaly_detector"
    assert detector.contamination == 0.05
    assert detector.method == "isolation_forest"

def test_anomaly_detector_preprocess(transaction_data):
    """Test preprocessing of transaction data."""
    detector = AnomalyDetector()
    processed_data = detector.preprocess(transaction_data)

    # Check required columns exist - update to match actual columns
    required_cols = ['date', 'amount', 'category', 'merchant', 'type']
    for col in required_cols:
        assert col in processed_data.columns

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(processed_data['date'])

def test_anomaly_detector_feature_extraction(transaction_data):
    """Test feature extraction."""
    detector = AnomalyDetector()
    processed_data = detector.preprocess(transaction_data)
    features = detector.extract_features(processed_data)

    # Check feature columns exist
    assert len(features.columns) > 5  # Should have multiple features
    assert len(detector.feature_names) > 0

def test_anomaly_detector_train(transaction_data):
    """Test training the anomaly detector."""
    detector = AnomalyDetector()
    processed_data = detector.preprocess(transaction_data)
    detector.train(processed_data)

    # Check model is fitted
    assert detector.is_fitted
    assert detector.model is not None
    assert 'detector' in detector.model
    assert 'scaler' in detector.model

    # Check metrics
    assert 'contamination_rate' in detector.metrics
    assert 'training_samples' in detector.metrics

def test_anomaly_detector_predict(transaction_data):
    """Test anomaly prediction."""
    detector = AnomalyDetector()
    processed_data = detector.preprocess(transaction_data)
    detector.train(processed_data)

    # Test on same data
    predictions = detector.predict(processed_data)

    # Check predictions format
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(processed_data)
    assert set(np.unique(predictions)).issubset({-1, 1})  # -1 for anomalies, 1 for normal

def test_anomaly_detector_predict_with_details(transaction_data):
    """Test detailed anomaly prediction."""
    detector = AnomalyDetector()
    processed_data = detector.preprocess(transaction_data)
    detector.train(processed_data)

    # Test detailed predictions
    results = detector.predict_with_details(processed_data)

    # Check results structure - update to match actual format
    assert isinstance(results, list)
    if len(results) > 0:
        anomaly = results[0]
        assert 'anomaly_score' in anomaly
        assert 'confidence' in anomaly