import numpy as np
import pandas as pd
import pytest

from app.models.categorization.transaction_categorizer import TransactionCategorizer

def test_categorizer_init():
    """Test initialization of transaction categorizer."""
    categorizer = TransactionCategorizer()
    assert categorizer.model_name == "transaction_categorizer"
    assert categorizer.confidence_threshold == 0.7

def test_categorizer_preprocess(transaction_data):
    """Test preprocessing of transaction data."""
    categorizer = TransactionCategorizer()
    processed_data = categorizer.preprocess(transaction_data)

    # Check required columns exist
    assert 'description' in processed_data.columns
    assert 'amount' in processed_data.columns

    # Check text preprocessing
    assert processed_data['description'].notna().all()

def test_categorizer_feature_extraction(transaction_data):
    """Test feature extraction."""
    categorizer = TransactionCategorizer()
    processed_data = categorizer.preprocess(transaction_data)
    features = categorizer.extract_features(processed_data)

    # Check feature columns exist
    assert len(features.columns) > 3  # Should have multiple features

def test_categorizer_train(categorized_data):
    """Test training the categorizer."""
    categorizer = TransactionCategorizer()

    # Ensure we have valid text data
    categorized_data = categorized_data.dropna(subset=['description'])
    categorized_data = categorized_data[categorized_data['description'].str.len() > 0]

    # Ensure minimum samples per category
    category_counts = categorized_data['category'].value_counts()
    valid_categories = category_counts[category_counts >= 3].index
    categorized_data = categorized_data[categorized_data['category'].isin(valid_categories)]

    processed_data = categorizer.preprocess(categorized_data)
    target = processed_data['category']

    categorizer.train(processed_data, target)
    assert categorizer.is_fitted



def test_categorizer_predict(categorized_data):
    """Test category prediction."""
    categorizer = TransactionCategorizer()
    processed_data = categorizer.preprocess(categorized_data)

    # Use categorized data with proper distribution
    target = processed_data['category']

    categorizer.train(processed_data, target)

    # Test on same data
    predictions = categorizer.predict(processed_data)

    # Check predictions format
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(processed_data)


def test_categorizer_predict_with_confidence(categorized_data):
    """Test prediction with confidence scores."""
    categorizer = TransactionCategorizer()
    processed_data = categorizer.preprocess(categorized_data)

    # Use categorized data with proper distribution
    target = processed_data['category']

    categorizer.train(processed_data, target)

    # Test confidence predictions
    results = categorizer.predict_with_confidence(processed_data)

    # Check results structure
    assert 'category' in results.columns
    assert 'confidence' in results.columns
    assert (results['confidence'] >= 0).all() and (results['confidence'] <= 1).all()

def test_categorizer_handles_empty_features(categorized_data):
    """Test categorizer handles empty feature extraction gracefully."""
    categorizer = TransactionCategorizer()

    # Create data that would result in empty features
    empty_data = pd.DataFrame({
        'description': ['', '', ''],
        'amount': [0, 0, 0],
        'category': ['A', 'B', 'C']
    })

    with pytest.raises(ValueError, match="No features extracted"):
        processed = categorizer.preprocess(empty_data)
        categorizer.extract_features(processed)
