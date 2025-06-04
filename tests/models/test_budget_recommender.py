import pytest
import pandas as pd
import numpy as np
from app.models.budgeting.budget_recommender import BudgetRecommender

def test_budget_recommender_init():
    """Test initialization of budget recommender."""
    recommender = BudgetRecommender()
    assert recommender.model_name == "budget_recommender"
    assert recommender.allocation_method == "50-30-20"

def test_budget_recommender_preprocess(transaction_data):
    """Test preprocessing of transaction data."""
    recommender = BudgetRecommender()
    processed_data = recommender.preprocess(transaction_data)

    # Check required columns exist
    assert 'category' in processed_data.columns
    assert 'amount' in processed_data.columns

    # Check data types
    assert pd.api.types.is_datetime64_any_dtype(processed_data['date'])

def test_budget_recommender_feature_extraction(transaction_data):
    """Test feature extraction."""
    recommender = BudgetRecommender()
    processed_data = recommender.preprocess(transaction_data)
    features = recommender.extract_features(processed_data)

    # Check feature columns exist - update to match actual columns
    expected_cols = ['total_income', 'total_expenses', 'num_categories', 'expense_volatility']
    for col in expected_cols:
        assert col in features.columns

def test_budget_recommender_train(transaction_data):
    """Test training the budget recommender."""
    recommender = BudgetRecommender()
    processed_data = recommender.preprocess(transaction_data)
    recommender.train(processed_data)

    # Check model is fitted
    assert recommender.is_fitted
    assert recommender.model is not None

    # Check metrics
    assert 'num_categories' in recommender.metrics
    assert 'total_monthly_spend' in recommender.metrics

def test_budget_recommender_generate_recommendations():
    """Test budget recommendation generation."""
    recommender = BudgetRecommender()

    # Mock the model components
    recommender.is_fitted = True
    recommender.model = {
        'category_clusters': {'Needs': ['Food', 'Housing'], 'Wants': ['Entertainment'], 'Savings': ['Investments']},
        'monthly_spend': {'Food': 500, 'Housing': 1000, 'Entertainment': 300, 'Investments': 200},
        'allocations': {'Needs': 0.5, 'Wants': 0.3, 'Savings': 0.2}
    }

    # Test recommendation generation
    recommendations = recommender.generate_recommendations(monthly_income=3000)

    # Check recommendations structure - update to match actual format
    assert 'allocations' in recommendations
    assert 'needs' in recommendations['allocations']
    assert 'wants' in recommendations['allocations']
    assert 'savings' in recommendations['allocations']
    assert recommendations['allocations']['needs']['amount'] == 1500.0