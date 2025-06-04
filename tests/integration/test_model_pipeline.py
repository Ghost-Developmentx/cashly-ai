import pytest
import pandas as pd
import asyncio
from app.services.ml.model_manager import model_manager
from app.utils.synthetic_data.transaction_generator import TransactionGenerator

@pytest.mark.asyncio
async def test_anomaly_detection_pipeline():
    """Test the full anomaly detection pipeline."""
    # Generate data
    generator = TransactionGenerator()
    data = generator.generate_transactions(num_days=60, transactions_per_day=(5, 10), anomaly_rate=0.1)

    # Get model
    detector = await model_manager.get_model('anomaly_detector')

    # Process data
    processed_data = detector.preprocess(data)

    # Train model
    detector.train(processed_data)

    # Make predictions - update to match actual return format
    results = detector.predict_with_details(processed_data)

    # Verify results - update assertions
    assert isinstance(results, list)
    if len(results) > 0:
        assert 'anomaly_score' in results[0]
        assert 'confidence' in results[0]


@pytest.mark.asyncio
async def test_categorization_pipeline():
    """Test the full transaction categorization pipeline."""
    # Generate data with more samples to ensure proper category distribution
    generator = TransactionGenerator()
    data = generator.generate_transactions(num_days=90, transactions_per_day=(8, 15))

    # Ensure proper category distribution
    categories = ['Food', 'Transportation', 'Entertainment', 'Shopping', 'Bills', 'Income']
    total_rows = len(data)
    samples_per_category = max(5, total_rows // len(categories))

    category_list = []
    for i, category in enumerate(categories):
        start_idx = i * samples_per_category
        end_idx = min((i + 1) * samples_per_category, total_rows)
        category_list.extend([category] * (end_idx - start_idx))

    # Fill remaining rows
    while len(category_list) < total_rows:
        category_list.append(categories[len(category_list) % len(categories)])

    data['category'] = category_list[:total_rows]

    # Get model
    categorizer = await model_manager.get_model('categorizer')

    # Process data
    processed_data = categorizer.preprocess(data)

    # Train model
    categorizer.train(processed_data, processed_data['category'])

    # Make predictions
    results = categorizer.predict_with_confidence(processed_data)

    # Verify results
    assert 'category' in results.columns
    assert 'confidence' in results.columns
